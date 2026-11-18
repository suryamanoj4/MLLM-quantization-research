"""Latency / memory / energy measurement for real-kernel rows (scripts/15_latency.py).

What `evaluate.measure_latency` (05_latency.py) gets wrong, and this fixes:
  * its "decode throughput" divides a whole generate() call -- which re-runs the
    prefill -- by the number of new tokens, so TPOT includes a prefill;
  * generation stops at EOS, so the decode length (and the KV length it ends at)
    differs between configurations;
  * host timers around unsynchronised work.

Here, at a fixed image resolution:
  vision_ms     vision encoder + projector only (get_image_features)
  ttft_ms       prefill forward (vision + LLM) through the first token's argmax
  prefill_ms    LLM prefill = ttft - vision, per sample
  tpot_ms       decode-only time per token: the n-1 decode steps after the prefill of
                a FORCED-length run (EOS ignored), no host sync inside the loop
  e2e_forced_ms prefill + exactly n new tokens
  e2e_natural_ms the eval path (phoenix.acab.generate, stops at EOS), with its length
All GPU spans are timed with CUDA events after torch.cuda.synchronize(); on CPU (tests)
perf_counter is used. Energy comes from NVML's total-energy counter (mJ, Volta+).
"""
from __future__ import annotations

import statistics as st
import time

import numpy as np
import torch

from .acab import _needs_cache_position, generate
from .loras import steer_mask


# --------------------------------------------------------------------------- #
# clocks, energy, sizes
# --------------------------------------------------------------------------- #
class Clock:
    """CUDA events when on GPU, perf_counter otherwise. ms(a, b) synchronises."""

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def mark(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def ms(self, a, b) -> float:
        if self.cuda:
            b.synchronize()
            return float(a.elapsed_time(b))
        return (b - a) * 1000.0


class Energy:
    """NVML total-energy counter of the GPU torch uses. read() -> joules or None."""

    def __init__(self, device="cuda"):
        self.handle, self.why = None, ""
        if not (torch.device(device).type == "cuda" and torch.cuda.is_available()):
            self.why = "no CUDA device"
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            h = None
            uuid = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", None)
            if uuid is not None:
                try:
                    h = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
                except Exception:                                   # noqa: BLE001
                    h = None
            if h is None:
                h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)           # raises if unsupported
            self.handle, self._nvml = h, pynvml
        except Exception as e:                                      # noqa: BLE001
            self.why = f"NVML energy counter unavailable: {e}"

    @property
    def ok(self) -> bool:
        return self.handle is not None

    def read(self) -> float | None:
        if self.handle is None:
            return None
        return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0


def weight_bytes(model: torch.nn.Module) -> int:
    """Parameter + buffer bytes (each storage once), plus bitsandbytes' absmax, which
    lives in a quant_state rather than a parameter or buffer."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t is None or t.device.type == "meta":
            continue
        key = (t.data_ptr(), t.numel(), t.dtype)
        if key in seen:
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
    for m in model.modules():
        qs = getattr(getattr(m, "weight", None), "quant_state", None)
        for name in ("absmax", "code"):
            a = getattr(qs, name, None) if qs is not None else None
            if isinstance(a, torch.Tensor):
                total += a.numel() * a.element_size()
        nested = getattr(qs, "state2", None) if qs is not None else None
        a = getattr(nested, "absmax", None) if nested is not None else None
        if isinstance(a, torch.Tensor):
            total += a.numel() * a.element_size()
    return total


def summarize(xs) -> dict:
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return {"median": None, "p90": None, "mean": None, "n": 0}
    return {"median": float(np.median(xs)), "p90": float(np.percentile(xs, 90)),
            "mean": float(np.mean(xs)), "n": len(xs)}


# --------------------------------------------------------------------------- #
# the measured spans
# --------------------------------------------------------------------------- #
@torch.no_grad()
def vision_forward(lm, batch: dict):
    """Vision encoder + projector only, the way the model's own forward calls them."""
    m = lm.model
    if _needs_cache_position(lm):                                   # Qwen2.5-VL
        return m.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
    cfg = m.config
    return m.get_image_features(
        pixel_values=batch["pixel_values"],
        vision_feature_layer=getattr(cfg, "vision_feature_layer", -2),
        vision_feature_select_strategy=getattr(cfg, "vision_feature_select_strategy", "default"))


@torch.no_grad()
def time_vision(lm, batch: dict, clock: Clock) -> float:
    clock.sync()
    a = clock.mark()
    vision_forward(lm, batch)
    b = clock.mark()
    return clock.ms(a, b)


@torch.no_grad()
def forced_decode(lm, batch: dict, n_new: int, clock: Clock, energy: Energy | None = None) -> dict:
    """Prefill + exactly n_new greedy tokens, EOS ignored. Returns ttft / decode / tpot
    (ms), the tokens, and the energy (J) of the whole run."""
    if n_new < 2:
        raise ValueError("n_new must be >= 2 (TPOT needs at least one decode step)")
    model = lm.model
    ids = batch["input_ids"]
    B, T = ids.shape
    vis = ids == lm.image_token_id
    am = batch.get("attention_mask")
    am = torch.ones_like(ids) if am is None else am
    ones = torch.ones(B, 1, dtype=am.dtype, device=am.device)
    qwen = _needs_cache_position(lm)
    toks = []
    clock.sync()
    e0 = energy.read() if energy is not None else None
    t0 = clock.mark()
    with steer_mask(vis):
        out = model(**batch, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1)
        t1 = clock.mark()
        past = out.past_key_values
        toks.append(nxt)
        for i in range(n_new - 1):
            am = torch.cat([am, ones], dim=1)
            step = dict(input_ids=nxt[:, None], attention_mask=am, past_key_values=past,
                        use_cache=True)
            if qwen:                       # same positions generate() passes
                step["cache_position"] = torch.arange(T + i, T + i + 1, device=ids.device)
            o = model(**step)
            past = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1)
            toks.append(nxt)
        t2 = clock.mark()
    ttft, dec = clock.ms(t0, t1), clock.ms(t1, t2)
    e1 = energy.read() if energy is not None else None
    return {"ttft_ms": ttft, "decode_ms": dec, "tpot_ms": dec / (n_new - 1),
            "e2e_ms": ttft + dec, "tokens": torch.stack(toks, 1),
            "energy_j": (e1 - e0) if e0 is not None and e1 is not None else None}


def natural_length(seqs: torch.Tensor, eos_id: int) -> list[int]:
    """Generated tokens before the first EOS (all of them if there is none)."""
    out = []
    for row in seqs.tolist():
        out.append(row.index(eos_id) if eos_id in row else len(row))
    return out


@torch.no_grad()
def natural_decode(lm, batch: dict, max_new: int, clock: Clock, energy: Energy | None = None) -> dict:
    """The evaluation path (generate(), stops at EOS), timed end to end."""
    vis = batch["input_ids"] == lm.image_token_id
    clock.sync()
    e0 = energy.read() if energy is not None else None
    t0 = clock.mark()
    with steer_mask(vis):
        r = generate(lm, batch, None, max_new_tokens=max_new)
    t1 = clock.mark()
    ms = clock.ms(t0, t1)
    e1 = energy.read() if energy is not None else None
    lens = natural_length(r["sequences"], lm.processor.tokenizer.eos_token_id)
    return {"e2e_ms": ms, "len": lens, "text": r["text"],
            "energy_j": (e1 - e0) if e0 is not None and e1 is not None else None}


def reset_peak(device):
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gb(device) -> float | None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 30
    return None


# --------------------------------------------------------------------------- #
# one row
# --------------------------------------------------------------------------- #
def measure_row(lm, batches: list[dict], *, n_new: int = 128, n_warmup: int = 10,
                repeats: int = 5, sweep: dict[int, list[dict]] | None = None,
                sweep_repeats: int = 3, sweep_warmup: int = 2, device="cuda",
                energy: Energy | None = None, log=print) -> dict:
    """All batch-1 measurements on `batches` (one prepared image each), then the batch
    sweep on `sweep` ({B: [batches of B images]}). Returns summaries + raw samples."""
    clock = Clock(device)
    for k in range(n_warmup):
        b = batches[k % len(batches)]
        time_vision(lm, b, clock)
        forced_decode(lm, b, n_new, clock)
        natural_decode(lm, b, n_new, clock)
    reset_peak(device)
    raw = {k: [] for k in ("vision_ms", "ttft_ms", "prefill_ms", "tpot_ms", "e2e_forced_ms",
                           "e2e_natural_ms", "len_natural", "j_caption", "j_token_forced",
                           "j_token_natural")}
    t_start = time.perf_counter()
    for r in range(repeats):
        for b in batches:
            v = time_vision(lm, b, clock)
            f = forced_decode(lm, b, n_new, clock, energy)
            nat = natural_decode(lm, b, n_new, clock, energy)
            raw["vision_ms"].append(v)
            raw["ttft_ms"].append(f["ttft_ms"])
            raw["prefill_ms"].append(f["ttft_ms"] - v)
            raw["tpot_ms"].append(f["tpot_ms"])
            raw["e2e_forced_ms"].append(f["e2e_ms"])
            raw["e2e_natural_ms"].append(nat["e2e_ms"])
            raw["len_natural"].append(nat["len"][0])
            if f["energy_j"] is not None:
                raw["j_token_forced"].append(f["energy_j"] / n_new)
            if nat["energy_j"] is not None:
                raw["j_caption"].append(nat["energy_j"])
                raw["j_token_natural"].append(nat["energy_j"] / max(nat["len"][0], 1))
        log(f"    repeat {r + 1}/{repeats}: TTFT {st.median(raw['ttft_ms']):.1f} ms, "
            f"TPOT {st.median(raw['tpot_ms']):.2f} ms ({time.perf_counter() - t_start:.0f}s)")
    out = {k: summarize(v) for k, v in raw.items()}
    out["len_natural"]["median_int"] = int(np.median(raw["len_natural"])) if raw["len_natural"] else None
    out["peak_alloc_gb"] = peak_gb(device)
    out["raw"] = raw
    out["example_caption"] = natural_decode(lm, batches[0], n_new, clock)["text"][0]
    if sweep:
        out["sweep"] = {}
        for B, bl in sorted(sweep.items()):
            reset_peak(device)
            for k in range(sweep_warmup):
                forced_decode(lm, bl[k % len(bl)], n_new, clock)
            tp, thr, e2e = [], [], []
            for _ in range(sweep_repeats):
                for b in bl:
                    f = forced_decode(lm, b, n_new, clock)
                    tp.append(f["tpot_ms"])
                    thr.append(B * (n_new - 1) / (f["decode_ms"] / 1000))
                    e2e.append(B * n_new / (f["e2e_ms"] / 1000))
            out["sweep"][str(B)] = {"tpot_ms": summarize(tp), "decode_tok_s": summarize(thr),
                                    "e2e_tok_s": summarize(e2e), "peak_alloc_gb": peak_gb(device),
                                    "n_batches": len(bl)}
            log(f"    batch {B:>2}: TPOT {st.median(tp):.2f} ms, "
                f"decode {st.median(thr):.0f} tok/s, e2e {st.median(e2e):.0f} tok/s")
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _mp(d, fmt="{:.1f}"):
    if not d or d.get("median") is None:
        return "-"
    return f"{fmt.format(d['median'])} ({fmt.format(d['p90'])})"


def markdown_table(rows: dict, model: str, n_new: int) -> str:
    sweep_bs = sorted({int(b) for r in rows.values() for b in r.get("sweep", {})})
    hdr = (["row", "backend", "LoRAS-T", "weights GB", "peak GB", "vision ms", "LLM prefill ms",
            "TTFT ms", "TPOT ms", f"e2e ms @{n_new} (forced)", "e2e ms (natural)",
            "len mean / median", "J/caption", "J/token"]
           + [f"tok/s @B={b}" for b in sweep_bs])
    lines = [f"### {model}: latency (median (p90)), batch 1 unless noted", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for name, r in rows.items():
        ln = r["len_natural"]
        cells = [name, r["backend"], r.get("lorast_mode", "off"),
                 f"{r['weights_gb']:.2f}", f"{r['peak_alloc_gb']:.2f}" if r.get("peak_alloc_gb") else "-",
                 _mp(r["vision_ms"]), _mp(r["prefill_ms"]), _mp(r["ttft_ms"]),
                 _mp(r["tpot_ms"], "{:.2f}"), _mp(r["e2e_forced_ms"], "{:.0f}"),
                 _mp(r["e2e_natural_ms"], "{:.0f}"),
                 f"{ln['mean']:.1f} / {ln.get('median_int')}" if ln.get("mean") is not None else "-",
                 _mp(r["j_caption"], "{:.1f}"), _mp(r["j_token_forced"], "{:.3f}")]
        for b in sweep_bs:
            s = r.get("sweep", {}).get(str(b))
            cells.append(f"{s['decode_tok_s']['median']:.0f}" if s else "-")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"TPOT = decode-only ms per token over the {n_new - 1} decode steps of a "
              f"forced {n_new}-token run (EOS ignored); LLM prefill = TTFT - vision per sample; "
              "J/token = energy of the forced run / "
              f"{n_new}; tok/s @B = decode tokens/s at batch B (forced length). "
              "Only fp16 / bnb / packed rows: simulated rows are never timed."]
    return "\n".join(lines)
