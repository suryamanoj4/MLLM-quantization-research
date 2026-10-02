"""Evaluation drivers: POPE, CHAIR, latency."""
from __future__ import annotations

import time
import torch

from .acab import ACABConfig, ACABController, generate, install_acab
from .data import batches_with_samples
from .loras import steer_mask
from .metrics import (expected_calibration_error, fluency_report, pope_metrics,
                      parse_yes_no, yes_no_probability)
from .utils import peak_vram_gb, reset_vram_stats


def _vis(lm, input_ids):
    return input_ids == lm.image_token_id


def _bar(total, desc, unit):
    """tqdm on stderr. Redraws at most every 30 s when stderr is a pipe/log file, so a
    `| tee run.log` stays readable instead of filling with carriage returns."""
    import sys
    from tqdm import tqdm
    tty = sys.stderr.isatty()
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True,
                mininterval=0.5 if tty else 30.0, file=sys.stderr)


@torch.no_grad()
def run_pope(lm, samples, ctrl: ACABController | None = None,
             batch_size: int = 8, max_new_tokens: int = 8,
             collect_probs: bool = True, verbose: bool = True,
             desc: str = "pope") -> dict:
    preds, labels, probs, corrects, texts, p_yes = [], [], [], [], [], []
    t0 = time.perf_counter()
    with _bar(len(samples), f"[{desc}]", "q") as bar:
        for chunk, batch in batches_with_samples(samples, lm, batch_size):
            mask = _vis(lm, batch["input_ids"])
            with steer_mask(mask):
                # generate() returns the prefill logits, so P(yes) for the ECE comes
                # free -- no second prefill (which is most of the cost of a POPE batch)
                r = generate(lm, batch, ctrl=ctrl, max_new_tokens=max_new_tokens)
            if collect_probs:
                p = yes_no_probability(r["first_logits"], lm.processor.tokenizer)
            for j, s in enumerate(chunk):
                pred = parse_yes_no(r["text"][j])
                preds.append(pred)
                labels.append(s.label)
                texts.append(r["text"][j])
                if collect_probs:
                    pv = float(p[j])
                    p_yes.append(pv)
                    probs.append(pv if s.label == "yes" else 1 - pv)
                    corrects.append((pred or "no") == s.label)
            bar.update(len(chunk))
            if preds:
                acc = sum((q or "no") == l for q, l in zip(preds, labels)) / len(preds)
                bar.set_postfix_str(f"acc {acc:.3f}")

    m = pope_metrics(preds, labels)
    if collect_probs:
        m["ece"] = expected_calibration_error(probs, corrects)
    m["wall_s"] = time.perf_counter() - t0
    m["samples"] = [{"q": s.question, "label": s.label, "pred": p, "raw": t}
                    for s, p, t in zip(samples[:20], preds[:20], texts[:20])]
    # per-question vectors, for paired comparisons between configurations
    m["preds"] = preds
    m["labels"] = labels
    m["image_ids"] = [int(s.image_id) for s in samples]
    m["p_yes"] = p_yes                   # P(yes) at the answer token, for AUROC
    return m


@torch.no_grad()
def run_captions(lm, samples, ctrl: ACABController | None = None,
                 batch_size: int = 4, max_new_tokens: int = 128,
                 do_sample: bool = False, record_trace: bool = False,
                 verbose: bool = True, desc: str = "captions") -> dict:
    records, traces = [], []
    t0 = time.perf_counter()
    with _bar(len(samples), f"[{desc}]", "cap") as bar:
        for chunk, batch in batches_with_samples(samples, lm, batch_size):
            mask = _vis(lm, batch["input_ids"])
            with steer_mask(mask):
                r = generate(lm, batch, ctrl=ctrl, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, record=record_trace)
            for j, s in enumerate(chunk):
                records.append({"image_id": s.image_id, "caption": r["text"][j]})
            if record_trace:
                traces.append(r["trace"])
            bar.update(len(chunk))
    return {"records": records, "traces": traces,
            "fluency": fluency_report([r["caption"] for r in records]),
            "wall_s": time.perf_counter() - t0}


@torch.no_grad()
def measure_latency(lm, samples, ctrl: ACABController | None = None,
                    n_warmup: int = 2, n_trials: int = 8,
                    max_new_tokens: int = 64) -> dict:
    """TTFT, decode throughput and peak VRAM. Batch size 1, the deployment case."""
    from .data import open_image
    from .model import prepare_batch
    it = list(samples)[: n_warmup + n_trials]
    ttfts, tputs = [], []
    reset_vram_stats()
    for i, s in enumerate(it):
        batch = prepare_batch(lm, [open_image(s.image_path)], [s.question])
        mask = _vis(lm, batch["input_ids"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with steer_mask(mask):
            out = lm.model(**batch, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
        del out
        t1 = time.perf_counter()
        with steer_mask(mask):
            r = generate(lm, batch, ctrl=ctrl, max_new_tokens=max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        n_new = int(r["sequences"].shape[1])
        if i >= n_warmup:
            ttfts.append(ttft * 1000)
            tputs.append(n_new / max(dt, 1e-6))
    import statistics as st
    return {
        "ttft_ms_median": st.median(ttfts) if ttfts else float("nan"),
        "decode_tok_per_s_median": st.median(tputs) if tputs else float("nan"),
        "peak_vram_gb": peak_vram_gb(),
        "n_trials": len(ttfts),
    }


class acab_session:
    """`with acab_session(lm, cfg) as ctrl: ...` -- installs and removes the patch."""

    def __init__(self, lm, cfg: ACABConfig | None):
        self.lm, self.cfg, self.ctrl, self._undo = lm, cfg, None, None

    def __enter__(self):
        if self.cfg is None or self.cfg.mode == "off":
            return None
        n_heads = self.lm.model.config.text_config.num_attention_heads \
            if hasattr(self.lm.model.config, "text_config") \
            else self.lm.model.config.num_attention_heads
        self.ctrl = ACABController(self.cfg, self.lm.n_layers, n_heads)
        self._undo = install_acab(self.lm.model, self.lm.layers, self.ctrl)
        return self.ctrl

    def __exit__(self, *a):
        if self._undo:
            self._undo()
        self._undo = None
