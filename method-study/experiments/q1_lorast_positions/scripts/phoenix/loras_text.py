"""LoRAS-T: text-side low-rank steering of the residual stream.

Gate 2 localised the grounding loss to *text-position* computation: W3 weights used
only at image positions do nothing; used only at text positions they reproduce the
whole coverage loss. So the correction belongs where the text tokens are -- the
prompt and, above all, every generated token -- not on the image K/V (the original
LoRAS site).

At the output of decoder layer l, for text positions t:

    h_q  = quantized hidden state (with every earlier corrector already live)
    h_fp = FP16 hidden state for the same tokens
    h_q <- h_q + (h_q A_l) B_l + b_l         (rank r)

(A_l, B_l, b_l) is the closed-form reduced-rank ridge solution (loras.RRRStats),
fitted block by block so each layer sees the corrected upstream it will see at
inference. Because it corrects the residual stream, it compensates the error
*accumulated* up to layer l, not just layer l's own.

Calibration targets are teacher-forced captions on the calibration images. By
default the captions are FP16's own (generated once with quantization switched off),
so the text positions match what the model actually emits.

Massive activations (a few tokens such as BOS carry norms 100x the rest) would
dominate the regression; they are excluded from the fit and, through a calibrated
norm gate, left uncorrected at inference.

Decode overhead: two skinny matmuls per layer per token (2 * d * r MACs).
"""
from __future__ import annotations

import time
from typing import Sequence

import torch
import torch.nn as nn

from .loras import RRRStats, SteerContext, steer_enabled, steer_mask
from .quant import quant_mode

MASSIVE_FACTOR = 10.0      # a token with norm > 10x the batch median is "massive"


class HiddenCorrector(nn.Module):
    def __init__(self, A, B, bias, max_norm: float, scale: float = 1.0):
        super().__init__()
        self.register_buffer("A", A, persistent=True)
        self.register_buffer("B", B, persistent=True)
        self.register_buffer("bias", bias, persistent=True)
        self.max_norm = float(max_norm)
        self.scale = float(scale)

    def forward(self, h: torch.Tensor, image_mask: torch.Tensor | None) -> torch.Tensor:
        if self.scale == 0.0 or not SteerContext.enabled:
            return h
        if image_mask is not None and h.dim() == 3 and h.shape[:2] == image_mask.shape:
            sel = ~image_mask                    # prefill: every non-image position
        else:
            sel = torch.ones(h.shape[:-1], dtype=torch.bool, device=h.device)  # decode
        sel = sel & (h.float().norm(dim=-1) <= self.max_norm)
        if not bool(sel.any()):
            return h
        x = h[sel].float()
        corr = (x @ self.A.float()) @ self.B.float() + self.bias.float()
        out = h.clone()
        out[sel] = (x + self.scale * corr).to(h.dtype)
        return out


def _layer_hook(corrector: HiddenCorrector):
    def hook(module, args, output):
        if isinstance(output, (tuple, list)):
            h = corrector(output[0], SteerContext.mask)
            return (h,) + tuple(output[1:])
        return corrector(output, SteerContext.mask)
    return hook


def attach_hidden(layers: nn.ModuleList, correctors: dict[int, HiddenCorrector]) -> None:
    for li, c in correctors.items():
        layer = layers[li]
        old = getattr(layer, "_loras_t_handle", None)
        if old is not None:
            old.remove()
        layer._loras_t_handle = layer.register_forward_hook(_layer_hook(c))
        layer._loras_t = c


def detach_hidden(layers: nn.ModuleList) -> None:
    for layer in layers:
        h = getattr(layer, "_loras_t_handle", None)
        if h is not None:
            h.remove()
            layer._loras_t_handle = None
            layer._loras_t = None


def set_scale(layers: nn.ModuleList, scale: float) -> None:
    for layer in layers:
        c = getattr(layer, "_loras_t", None)
        if c is not None:
            c.scale = float(scale)


def save(path, correctors: dict[int, HiddenCorrector], meta: dict) -> None:
    torch.save({"state": {str(li): {"A": c.A.cpu(), "B": c.B.cpu(), "bias": c.bias.cpu(),
                                     "max_norm": c.max_norm}
                          for li, c in correctors.items()},
                "meta": meta}, path)


def load(path, device, dtype) -> tuple[dict[int, HiddenCorrector], dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = {int(li): HiddenCorrector(t["A"].to(device, dtype), t["B"].to(device, dtype),
                                    t["bias"].to(device, dtype), t["max_norm"])
           for li, t in blob["state"].items()}
    return out, blob["meta"]


# --------------------------------------------------------------------------- #
# calibration data: teacher-forced captions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def caption_sequences(lm, root, n: int, seed: int = 0, source: str = "gen",
                      max_new_tokens: int = 96, batch_size: int = 8) -> tuple[list[dict], list[str]]:
    """[image + caption prompt + caption] inputs (batch 1) from the calibration pool.

    source="gen": FP16's own greedy captions (quantization off), the trajectory the
    corrected model should follow. source="ref": two COCO reference captions.
    """
    import random
    from .acab import generate
    from .data import build_calibration_set, load_captions, open_image
    from .model import CAPTION_PROMPT, build_prompt, prepare_batch
    samples = build_calibration_set(root, "train2014", n_images=n, seed=seed + 29,
                                    fallback_subset="val2014")
    caps: list[str] = []
    t0 = time.perf_counter()
    if source == "gen":
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            imgs = [open_image(s.image_path) for s in chunk]
            batch = prepare_batch(lm, imgs, [CAPTION_PROMPT] * len(chunk))
            with quant_mode(lm.model, False), steer_enabled(False), \
                    steer_mask(batch["input_ids"] == lm.image_token_id):
                caps += generate(lm, batch, None, max_new_tokens=max_new_tokens)["text"]
            print(f"[loras-t] FP16 captions {len(caps)}/{len(samples)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    else:
        refs: dict = {}
        for sub in ("train2014", "val2014"):
            try:
                refs.update(load_captions(root, sub))
            except FileNotFoundError:
                pass
        rng = random.Random(seed)
        for s in samples:
            cs = refs.get(s.image_id) or ["A photo."]
            caps.append(" ".join(rng.sample(cs, min(2, len(cs)))))
    out = []
    for s, cap in zip(samples, caps):
        text = build_prompt(CAPTION_PROMPT) + " " + cap.strip()
        enc = lm.processor(images=[open_image(s.image_path)], text=[text], return_tensors="pt")
        out.append({k: (v.to(lm.device, lm.dtype) if k == "pixel_values" else v.to(lm.device))
                    for k, v in enc.items()})
    return out, caps


@torch.no_grad()
def qa_sequences(lm, root, n: int, per_image: int = 2, seed: int = 0,
                 batch_size: int = 8) -> tuple[list[dict], list[dict], list[dict]]:
    """Yes/no object questions on the calibration images, answered by FP16.

    Caption-only calibration never shows the corrector a question prompt, and the
    POPE answer is decided at the last prompt token. Half the questions name an object
    annotated in the image, half an absent one (alternating a random and a
    co-occurring category, as POPE's random/adversarial splits do). Answers are
    FP16's own; no labels are used. Only calibration images are touched.

    Returns (teacher-forced batches for calibration, prompt-only batches for the
    held-out P(yes) check, per-question records).
    """
    import random
    from collections import Counter, defaultdict
    from .acab import generate
    from .data import POPE_TEMPLATE, build_calibration_set, load_instances, open_image
    from .model import build_prompt, prepare_batch
    samples = build_calibration_set(root, "train2014", n_images=n, seed=seed + 29,
                                    fallback_subset="val2014")
    objs: dict = {}
    for sub in ("train2014", "val2014"):
        try:
            objs.update(load_instances(root, sub)[0])
        except FileNotFoundError:
            pass
    cats = sorted({c for v in objs.values() for c in v})
    co: dict = defaultdict(Counter)
    for v in objs.values():
        for a in v:
            for b in v:
                if a != b:
                    co[a][b] += 1
    rng = random.Random(seed + 7)
    items = []
    for k, s in enumerate(samples):
        present = sorted(objs.get(s.image_id, ()))
        if not present:
            continue
        absent = [c for c in cats if c not in present]
        n_pos = max(1, per_image // 2)
        for i in range(n_pos):
            items.append((s, rng.choice(present)))
            if (k + i) % 2:
                score = Counter()
                for c in present:
                    score.update(co[c])
                ranked = [c for c, _ in score.most_common() if c not in present] or absent
                items.append((s, ranked[min(i, len(ranked) - 1)]))
            else:
                items.append((s, rng.choice(absent)))
    calib, probe, records = [], [], []
    t0 = time.perf_counter()
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        imgs = [open_image(s.image_path) for s, _ in chunk]
        qs = [POPE_TEMPLATE.format(obj=o) for _, o in chunk]
        batch = prepare_batch(lm, imgs, qs)
        with quant_mode(lm.model, False), steer_enabled(False), \
                steer_mask(batch["input_ids"] == lm.image_token_id):
            ans = generate(lm, batch, None, max_new_tokens=4)["text"]
        texts = [build_prompt(q) + " " + a.strip() for q, a in zip(qs, ans)]
        enc = lm.processor(images=imgs, text=texts, return_tensors="pt", padding=True)
        calib.append({k: (v.to(lm.device, lm.dtype) if k == "pixel_values" else v.to(lm.device))
                      for k, v in enc.items()})
        probe.append(batch)
        records += [{"image_id": s.image_id, "object": o, "fp16_answer": a}
                    for (s, o), a in zip(chunk, ans)]
    print(f"[loras-t] {len(records)} FP16 yes/no answers on {len(samples)} calibration images "
          f"({time.perf_counter() - t0:.0f}s, "
          f"{sum(r['fp16_answer'].lower().startswith('yes') for r in records)} yes)", flush=True)
    return calib, probe, records


@torch.no_grad()
def heldout_pyes(lm, probe: list[dict], idx, steer: bool) -> dict:
    """Mean |P_q(yes) - P_fp(yes)| and mean shift at the answer position."""
    from .metrics import yes_no_probability
    tok = lm.processor.tokenizer
    diffs = []
    for j in idx:
        b = probe[j]
        img = b["input_ids"] == lm.image_token_id
        with steer_mask(img), steer_enabled(False), quant_mode(lm.model, False):
            pf = yes_no_probability(lm.model(**b, use_cache=False).logits[:, -1], tok)
        with steer_mask(img), steer_enabled(steer), quant_mode(lm.model, True):
            pq = yes_no_probability(lm.model(**b, use_cache=False).logits[:, -1], tok)
        diffs.append(pq - pf)
    d = torch.cat(diffs) if diffs else torch.zeros(1)
    return {"abs": float(d.abs().mean()), "shift": float(d.mean()), "n": int(d.numel())}


# --------------------------------------------------------------------------- #
# sequential closed-form calibration
# --------------------------------------------------------------------------- #
class _Stop(Exception):
    pass


def _text_sel(lm, s: dict) -> torch.Tensor:
    img = s["input_ids"] == lm.image_token_id
    am = s.get("attention_mask")
    am = torch.ones_like(img) if am is None else am.bool()
    return (~img) & am


@torch.no_grad()
def _run_capture(lm, s: dict, layer_ids: Sequence[int], quant: bool, stop_after: int | None):
    """Hidden states at text positions after each layer in layer_ids ([N, d] each)."""
    layers, sel = lm.layers, _text_sel(lm, s)
    store, hs = {}, []
    for li in layer_ids:
        def hook(m, a, out, li=li):
            h = out[0] if isinstance(out, (tuple, list)) else out
            store[li] = h[sel].detach()
            if stop_after is not None and li == stop_after:
                raise _Stop
        hs.append(layers[li].register_forward_hook(hook))
    img = s["input_ids"] == lm.image_token_id
    try:
        with steer_mask(img), steer_enabled(quant), quant_mode(lm.model, quant):
            lm.model(**s, use_cache=False)
    except _Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    return store


@torch.no_grad()
def calibrate(lm, seqs: list[dict], layer_ids: Sequence[int], rank: int = 64,
              ridge: float = 1e-2, layers_per_pass: int = 1, energy: float | None = None,
              val_every: int = 5, verbose: bool = True) -> tuple[dict, dict]:
    """Fit one HiddenCorrector per layer, front to back.

    FP16 targets are captured once (CPU, fp16). Each block then re-runs the quantized
    model -- with every earlier corrector live -- up to the block's last layer only.
    layers_per_pass=1 is exact sequential fitting (each corrector sees the corrected
    upstream it will see at inference); larger values trade that for speed.
    """
    layers = lm.layers
    layer_ids = sorted(layer_ids)
    t0 = time.perf_counter()
    fp_cache: list[dict] = []
    for j, s in enumerate(seqs):
        st = _run_capture(lm, s, layer_ids, quant=False, stop_after=layer_ids[-1])
        fp_cache.append({li: v.to("cpu", torch.float16) for li, v in st.items()})
    if verbose:
        print(f"[loras-t] FP16 targets for {len(seqs)} sequences, {len(layer_ids)} layers "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    blocks = [layer_ids[i:i + layers_per_pass] for i in range(0, len(layer_ids), layers_per_pass)]
    correctors: dict[int, HiddenCorrector] = {}
    diag: dict[int, dict] = {}
    for bi, block in enumerate(blocks):
        t0 = time.perf_counter()
        fit: dict[int, RRRStats] = {}
        val: dict[int, RRRStats] = {}
        kept_norm = {li: 0.0 for li in block}
        n_massive = {li: 0 for li in block}
        base_err = {li: [0.0, 0.0] for li in block}      # sum ||y-x||^2, sum ||y||^2
        for j, s in enumerate(seqs):
            q = _run_capture(lm, s, block, quant=True, stop_after=block[-1])
            store = val if (val_every and j % val_every == 0) else fit
            for li in block:
                y = fp_cache[j][li].to(lm.device).float()
                x = q[li].float()
                ny = y.norm(dim=-1)
                ok = ny <= MASSIVE_FACTOR * ny.median().clamp(min=1e-6)
                n_massive[li] += int((~ok).sum())
                if not bool(ok.any()):
                    continue
                x, y = x[ok], y[ok]
                kept_norm[li] = max(kept_norm[li], float(x.norm(dim=-1).max()))
                base_err[li][0] += float((y - x).pow(2).sum())
                base_err[li][1] += float(y.pow(2).sum())
                if li not in store:
                    store[li] = RRRStats(x.shape[-1], y.shape[-1], x.device)
                store[li].update(x, y)
        new = {}
        for li in block:
            st = fit[li]
            sol = st.solve(rank=rank, ridge=ridge, use_bias=True, energy=energy)
            new[li] = HiddenCorrector(sol["A"].to(lm.device, lm.dtype),
                                      sol["B"].to(lm.device, lm.dtype),
                                      sol["bias"].to(lm.device, lm.dtype),
                                      max_norm=1.25 * kept_norm[li])
            d = {k: sol[k] for k in ("rank", "rel_mse_reduction", "ceiling", "n_tokens")}
            if li in val:
                d["val_red"] = val[li].evaluate(sol["A"], sol["B"], sol["bias"])
                val[li].free()
            d["massive_excluded"] = n_massive[li]
            d["rel_err_before"] = (base_err[li][0] / max(base_err[li][1], 1e-12)) ** 0.5
            if d.get("val_red", 1.0) <= 0.0:      # does not generalise: leave this layer alone
                d["skipped"] = True
                new.pop(li)
            diag[li] = d
            st.free()
        attach_hidden(layers, new)                # live for the next block
        correctors.update(new)
        for li in block:                          # targets no longer needed
            for c in fp_cache:
                c.pop(li, None)
        if verbose:
            for li in block:
                d = diag[li]
                print(f"[loras-t] L{li:02d} rel_err={d['rel_err_before']:.4f} "
                      f"rank={d['rank']:3d} fit={d['rel_mse_reduction']:.3f} "
                      f"val={d.get('val_red', float('nan')):.3f} ceiling={d['ceiling']:.3f} "
                      f"tokens={d['n_tokens']} massive={d['massive_excluded']}"
                      f"{' SKIPPED (val<=0)' if d.get('skipped') else ''} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return correctors, diag
