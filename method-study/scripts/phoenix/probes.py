"""Mechanistic probes: layer-depth drift, KV error, visual attention mass/entropy.

These feed two decisions that the proposal currently makes a priori:
  * *where* to put LoRAS (the proposal asserts l >= 0.75L; drift profiling usually
    says otherwise -- cross-modal representation drift in LLaVA-style decoders
    tends to peak in the middle third, and correcting only the top quartile can be
    too late to matter),
  * *which* layers/heads A-CAB should touch (the "visual heads").
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loras import _attn_of, steer_enabled, steer_mask
from .quant import quant_mode


# --------------------------------------------------------------------------- #
# generic activation capture
# --------------------------------------------------------------------------- #
class Capture:
    """Capture the outputs of named submodules, restricted to a boolean mask."""

    def __init__(self):
        self.store: dict = {}
        self._handles: list = []
        self._mask: torch.Tensor | None = None

    def hook(self, key, module: nn.Module):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if self._mask is not None and t.dim() == 3 and t.shape[:2] == self._mask.shape:
                t = t[self._mask]
            self.store[key] = t.detach()
        self._handles.append(module.register_forward_hook(fn))

    def set_mask(self, mask):
        self._mask = mask

    def clear(self):
        self.store = {}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()


def kv_capture(layers: nn.ModuleList, layer_ids: Sequence[int],
               sites: Sequence[str] = ("k", "v")) -> Capture:
    """Hook k_proj / v_proj outputs (pre-RoPE). Works through LoRASProj wrappers."""
    cap = Capture()
    for li in layer_ids:
        attn = _attn_of(layers[li])
        for s in sites:
            cap.hook((li, s), getattr(attn, f"{s}_proj"))
    return cap


# --------------------------------------------------------------------------- #
# layer-depth drift between FP16 and quantized
# --------------------------------------------------------------------------- #
@torch.no_grad()
def drift_profile(lm, batches: Iterable[dict], visual_mask_fn: Callable,
                  max_batches: int = 16) -> dict:
    """Cosine similarity and relative MSE, FP16 vs quantized, at visual positions."""
    model = lm.model
    n_layers = lm.n_layers
    acc = defaultdict(lambda: {"cos": 0.0, "rel_mse": 0.0, "n": 0})

    layer_ids = list(range(n_layers))
    for bi, batch in enumerate(batches):
        if bi >= max_batches:
            break
        mask = visual_mask_fn(batch["input_ids"])

        cap_fp, cap_q = kv_capture(lm.layers, layer_ids), kv_capture(lm.layers, layer_ids)
        cap_fp.set_mask(mask)
        cap_q.set_mask(mask)

        with steer_enabled(False), quant_mode(model, False):
            out_fp = model(**batch, output_hidden_states=True, use_cache=False)
        hs_fp = [h.detach() for h in out_fp.hidden_states]
        kv_fp = dict(cap_fp.store)
        cap_fp.remove()
        del out_fp

        with steer_mask(mask), steer_enabled(False), quant_mode(model, True):
            out_q = model(**batch, output_hidden_states=True, use_cache=False)
        hs_q = [h.detach() for h in out_q.hidden_states]
        kv_q = dict(cap_q.store)
        cap_q.remove()
        del out_q

        for li in range(len(hs_fp)):
            a, b = hs_fp[li][mask].float(), hs_q[li][mask].float()
            _accum(acc[("hidden", li)], a, b)
        for key in kv_fp:
            _accum(acc[("kv", key)], kv_fp[key].float(), kv_q[key].float())

        del hs_fp, hs_q, kv_fp, kv_q
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out = {"hidden": [], "k": [], "v": []}
    for li in range(n_layers + 1):
        e = acc.get(("hidden", li))
        if e and e["n"]:
            out["hidden"].append({"layer": li, "cos": e["cos"] / e["n"],
                                  "rel_mse": e["rel_mse"] / e["n"]})
    for site in ("k", "v"):
        for li in range(n_layers):
            e = acc.get(("kv", (li, site)))
            if e and e["n"]:
                out[site].append({"layer": li, "cos": e["cos"] / e["n"],
                                  "rel_mse": e["rel_mse"] / e["n"]})
    return out


def _accum(entry, fp: torch.Tensor, q: torch.Tensor):
    entry["cos"] += F.cosine_similarity(fp, q, dim=-1).mean().item()
    entry["rel_mse"] += ((fp - q).pow(2).sum() / fp.pow(2).sum().clamp(min=1e-12)).item()
    entry["n"] += 1


def select_layers(drift: dict, k: int = 8, site: str = "k",
                  strategy: str = "max_drift", n_layers: int | None = None) -> list[int]:
    """Pick LoRAS layers from the drift profile."""
    if strategy == "upper_quartile":               # the proposal's a priori rule
        n = n_layers or (len(drift[site]))
        return list(range(int(0.75 * n), n))
    if strategy == "all":
        return [d["layer"] for d in drift[site]]
    rows = sorted(drift[site], key=lambda d: -d["rel_mse"])[:k]
    return sorted(r["layer"] for r in rows)


# --------------------------------------------------------------------------- #
# visual attention mass and predictive entropy
# --------------------------------------------------------------------------- #
@torch.no_grad()
def visual_attention_stats(attentions: Sequence[torch.Tensor],
                           vis_cols: torch.Tensor) -> dict:
    """attentions: tuple of [B, H, q, kv] per layer. vis_cols: bool [kv]."""
    mass, ent = [], []
    for A in attentions:
        a = A.float()
        m = a[..., vis_cols].sum(-1)                      # [B,H,q]
        e = -(a.clamp(min=1e-12) * a.clamp(min=1e-12).log()).sum(-1)
        mass.append(m.mean(dim=(0, 2)).cpu())             # per head
        ent.append(e.mean(dim=(0, 2)).cpu())
    return {"mass": torch.stack(mass), "entropy": torch.stack(ent)}   # [L, H]


def rank_visual_heads(mass: torch.Tensor, top_frac: float = 0.25,
                      layers: Sequence[int] | None = None) -> dict[int, list[int]]:
    """Return {layer: [head ids]} for the top `top_frac` heads by visual mass."""
    L, H = mass.shape
    cand = layers if layers is not None else range(L)
    flat = [(float(mass[l, h]), l, h) for l in cand for h in range(H)]
    flat.sort(reverse=True)
    keep = flat[: max(1, int(top_frac * len(flat)))]
    out: dict[int, list[int]] = defaultdict(list)
    for _, l, h in keep:
        out[l].append(h)
    return {l: sorted(v) for l, v in out.items()}


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of the next-token distribution. logits: [B, V]."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


@torch.no_grad()
def kl_to_blind(model, inputs_with_image: dict, inputs_blind: dict) -> float:
    """D_KL(P_Q(.|y<t,v) || P_blind(.|y<t)) -- the proposal's Eq. 2, teacher forced."""
    lv = model(**inputs_with_image, use_cache=False).logits
    lb = model(**inputs_blind, use_cache=False).logits
    n = min(lv.shape[1], lb.shape[1])
    p = torch.log_softmax(lv[:, -n:].float(), -1)
    q = torch.log_softmax(lb[:, -n:].float(), -1)
    return float((p.exp() * (p - q)).sum(-1).mean())


# --------------------------------------------------------------------------- #
# language-model health without an image
# --------------------------------------------------------------------------- #
BLIND_PROMPTS = [
    "Describe what a busy city street usually looks like.",
    "What do people usually keep in a kitchen?",
    "Explain how to make a cup of tea.",
    "Describe a typical day at the beach.",
    "What animals might you see on a farm?",
    "Describe the inside of a classroom.",
    "What happens at a football match?",
    "Describe a quiet park in the morning.",
]


def blind_captions(coco_root, n: int = 40, subset: str = "val2014") -> list[str]:
    """Reference captions from the *eval* pool (calibration never sees them)."""
    from .data import load_captions, split_pool
    pool = split_pool(coco_root, "eval")
    caps = []
    for iid, cs in sorted(load_captions(coco_root, subset).items()):
        if cs and (pool is None or iid in pool):
            caps.append(cs[0])
        if len(caps) >= n:
            break
    return caps


@torch.no_grad()
def blind_fluency(lm, captions: list[str], max_new_tokens: int = 64) -> dict:
    """Text-only generation + teacher-forced caption perplexity, no image anywhere.

    If a configuration is fluent here but fails on images, the failure needs the image.
    If it is broken here too, the language model itself is damaged."""
    import math
    from .acab import generate
    from .metrics import fluency_report
    tok = lm.processor.tokenizer
    prompts = [f"USER: {q} ASSISTANT:" for q in BLIND_PROMPTS]
    enc = tok(prompts, return_tensors="pt", padding=True).to(lm.device)
    r = generate(lm, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
                 None, max_new_tokens=max_new_tokens)
    nll, ntok = 0.0, 0
    head = "USER: Write a one-sentence description of a photo. ASSISTANT:"
    ids_head = tok(head, return_tensors="pt")["input_ids"].to(lm.device)
    for c in captions:
        ids_all = tok(head + " " + c.strip(), return_tensors="pt")["input_ids"].to(lm.device)
        k = ids_all.shape[1] - ids_head.shape[1]
        if k <= 0:
            continue
        logits = lm.model(input_ids=ids_all, use_cache=False).logits[:, :-1].float()
        tgt = ids_all[:, 1:]
        lp = torch.log_softmax(logits, -1)[0, -k:].gather(-1, tgt[0, -k:, None]).squeeze(-1)
        nll -= float(lp.sum())
        ntok += k
    return {"fluency": fluency_report(r["text"]), "answers": r["text"],
            "blind_ppl": math.exp(nll / ntok) if ntok else float("nan"),
            "scored_tokens": ntok}
