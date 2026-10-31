"""Cross-precision causal patching: where in the network is grounding lost?

Study 2 showed *that* low-bit weights make the model omit objects, and that the
loss comes from computation at text positions. This module locates it by layer,
component and position, with two complementary interventions on one teacher-forced
FP16 caption:

  activation patching   run the quantized model, but overwrite the output of one
                        site (layer l x {residual stream, attention output, MLP
                        output}) at one position set ({text, image, all}) with the
                        FP16 model's activation at that site.
  weight patching       "restore": everything quantized except layer l's attention
                        (or MLP) weights; "damage": everything FP16 except them.

The measured quantity is the log-probability the model assigns to each token of
FP16's own greedy caption. Tokens are split into
  obj     first sub-token of a mention of a COCO object that is in the image
  hall    first sub-token of a mention of an object that is not
  other   every other caption token (function words, attributes, punctuation ...)
so that any effect on *grounding* can be told apart from generic language damage.

For a token class k and an intervention p,
    gap_k = sum (lp_fp - lp_q)            (how much quantization costs)
    R_k(p) = sum (lp_p - lp_q) / gap_k    (fraction of it the intervention restores)
    D_k(p) = sum (lp_fp - lp_p) / gap_k   (fraction a single-site damage reproduces)
with sums over all tokens of class k in all captions (ratio of sums, so captions with
small gaps do not dominate), and bootstrap CIs by resampling captions.

Efficiency: the vision tower and projector are never quantized, so the decoder's
input embeddings (and Qwen's M-RoPE position ids) are identical in every run. They
are captured once per caption and every intervention then runs the decoder stack
alone, batched over interventions.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .quant import FakeQuantLinear, quant_mode

SITES = ("resid", "attn", "mlp")
POSSETS = ("text", "image", "all")
CLASSES = ("obj", "hall", "other")


# --------------------------------------------------------------------------- #
# sites
# --------------------------------------------------------------------------- #
def site_module(lm, site: str, layer: int) -> nn.Module:
    blk = lm.layers[layer]
    if site == "resid":
        return blk
    if site == "attn":
        return blk.self_attn
    if site == "mlp":
        return blk.mlp
    raise ValueError(site)


def _out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def _swap(o, h):
    if isinstance(o, tuple):
        return (h,) + tuple(o[1:])
    if isinstance(o, list):
        return [h] + list(o[1:])
    return h


def fq_modules(lm, comp: str, layers: Iterable[int]) -> list[FakeQuantLinear]:
    """FakeQuantLinears of the attention ('attn') or MLP ('mlp') of the given layers."""
    out = []
    for l in layers:
        root = lm.layers[l].self_attn if comp == "attn" else lm.layers[l].mlp
        out += [m for m in root.modules() if isinstance(m, FakeQuantLinear)]
    return out


@contextmanager
def weights_override(mods: Sequence[FakeQuantLinear], enabled: bool):
    prev = [m.quant_enabled for m in mods]
    try:
        for m in mods:
            m.quant_enabled = enabled
        yield
    finally:
        for m, p in zip(mods, prev):
            m.quant_enabled = p


# --------------------------------------------------------------------------- #
# one caption, prepared once
# --------------------------------------------------------------------------- #
class _Stop(Exception):
    pass


@dataclass
class Prepared:
    embeds: torch.Tensor                 # [1, T, d] decoder input embeddings
    position_ids: torch.Tensor | None    # [1, T] / [3, 1, T] (M-RoPE) / None
    image_mask: torch.Tensor             # [T] bool
    tgt_pos: torch.Tensor                # [n] positions of caption tokens
    targets: torch.Tensor                # [n] the caption token ids
    tags: list[str]                      # [n] obj / hall / other / cont
    fp_acts: dict = field(default_factory=dict)   # (site, l) -> [T, d]
    lp_fp: torch.Tensor | None = None    # [n]
    lp_q: torch.Tensor | None = None     # [n]
    top_fp: torch.Tensor | None = None   # [n] argmax agreement with the target (bool)
    top_q: torch.Tensor | None = None


@torch.no_grad()
def capture_decoder_inputs(lm, inputs: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the full model up to the decoder and grab its inputs_embeds / position_ids."""
    got = {}

    def pre(mod, args, kwargs):
        got["embeds"] = kwargs.get("inputs_embeds")
        got["pos"] = kwargs.get("position_ids")
        raise _Stop

    h = lm.lm.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        with quant_mode(lm.model, False):
            lm.model(**inputs, use_cache=False)
    except _Stop:
        pass
    finally:
        h.remove()
    if got.get("embeds") is None:          # text-only causal LMs (tests): embed ourselves
        ids = inputs["input_ids"]
        got["embeds"] = lm.lm.get_input_embeddings()(ids)
    return got["embeds"].detach(), (got["pos"].detach() if got.get("pos") is not None else None)


def _rep_pos(pos: torch.Tensor | None, B: int):
    if pos is None:
        return None
    if pos.dim() == 3:                     # [3, 1, T]
        return pos.expand(pos.shape[0], B, pos.shape[2])
    return pos.expand(B, pos.shape[-1])


@torch.no_grad()
def decoder_logprobs(lm, prep: Prepared, B: int = 1, plan: dict | None = None,
                     capture: bool = False, sites_layers: Sequence | None = None):
    """Run the decoder on B copies of the caption. `plan[(site, l)] = [(row, mask)]`
    overwrites that site's output at `mask` positions of `row` with the FP16 activation.
    Returns (log p(target) [B, n], argmax==target [B, n])."""
    T = prep.embeds.shape[1]
    hooks, store = [], {}
    keys = set(plan) if plan else set()
    if capture:
        keys |= set(sites_layers)
    for (site, l) in keys:
        mod = site_module(lm, site, l)

        def hook(m, a, o, key=(site, l)):
            h = _out(o)
            if capture:
                store[key] = h[0].detach().clone()
                return None
            todo = plan.get(key)
            if not todo:
                return None
            h = h.clone()
            src = prep.fp_acts[key].to(h.dtype)
            for row, mask in todo:
                h[row, mask] = src[mask]
            return _swap(o, h)
        hooks.append(mod.register_forward_hook(hook))
    try:
        out = lm.lm(inputs_embeds=prep.embeds.expand(B, T, prep.embeds.shape[-1]),
                    position_ids=_rep_pos(prep.position_ids, B),
                    attention_mask=torch.ones(B, T, dtype=torch.long, device=prep.embeds.device),
                    use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    hid = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    head = lm.model.get_output_embeddings()
    logits = head(hid[:, prep.tgt_pos - 1, :]).float()          # [B, n, V]
    lp = torch.log_softmax(logits, -1)
    tgt = prep.targets.to(lp.device)
    lpt = lp.gather(-1, tgt.view(1, -1, 1).expand(B, -1, 1)).squeeze(-1)
    top = logits.argmax(-1) == tgt.view(1, -1)
    if capture:
        prep.fp_acts = store
    return lpt, top


# --------------------------------------------------------------------------- #
# object-token tagging
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-zA-Z]+")


def object_mentions(text: str, mapping: dict, double_words: Sequence[str]) -> list[tuple[int, int, str]]:
    """(char_start, char_end, coco_class) for each object mention, CHAIR's matching rules."""
    words = [(m.start(), m.end(), m.group().lower()) for m in _WORD.finditer(text)]
    dbl = set(double_words)
    out, i = [], 0
    while i < len(words):
        s, e, w = words[i]
        if i + 1 < len(words) and f"{w} {words[i+1][2]}" in dbl:
            pair = f"{w} {words[i+1][2]}"
            c = mapping.get(pair.replace(" ", "_")) or mapping.get(pair)
            if c:
                out.append((s, words[i + 1][1], c))
            i += 2
            continue
        c = mapping.get(w)
        if c:
            out.append((s, e, c))
        i += 1
    return out


def token_char_spans(tok, ids: Sequence[int]) -> tuple[str, list[tuple[int, int]]]:
    """Character span of every token in the decoded text (incremental decoding)."""
    texts = [tok.decode(list(ids[:i + 1]), skip_special_tokens=True) for i in range(len(ids))]
    spans, prev = [], 0
    for t in texts:
        cur = len(t)
        spans.append((min(prev, cur), cur))
        prev = cur
    return (texts[-1] if texts else ""), spans


def tag_tokens(tok, ids: Sequence[int], gt: set[str], mapping: dict,
               double_words: Sequence[str]) -> tuple[list[str], str]:
    text, spans = token_char_spans(tok, ids)
    tags = ["other"] * len(ids)
    for (s, e, c) in object_mentions(text, mapping, double_words):
        first = True
        for j, (a, b) in enumerate(spans):
            if b <= s or a >= e or a == b:
                continue
            if first:
                tags[j] = "obj" if c in gt else "hall"
                first = False
            else:
                tags[j] = "cont"
    return tags, text


# --------------------------------------------------------------------------- #
# interventions
# --------------------------------------------------------------------------- #
def act_configs(n_layers: int, layers: Sequence[int] | None = None,
                sites=SITES, possets=("text", "image")) -> list[dict]:
    """Activation-patching configurations. 'all' positions only for the residual stream
    (it yields the cumulative curve: everything up to l exact, everything after quantized)."""
    layers = list(range(n_layers)) if layers is None else list(layers)
    out = []
    for site in sites:
        for ps in possets:
            for l in layers:
                out.append({"mode": "act", "site": site, "pos": ps, "layer": l})
        if site == "resid":
            for l in layers:
                out.append({"mode": "act", "site": site, "pos": "all", "layer": l})
    return out


def weight_configs(n_layers: int, layers: Sequence[int] | None = None,
                   comps=("attn", "mlp"), kinds=("restore", "damage")) -> list[dict]:
    layers = list(range(n_layers)) if layers is None else list(layers)
    return [{"mode": "w", "kind": k, "comp": c, "layer": l}
            for k in kinds for c in comps for l in layers]


def config_key(c: dict) -> str:
    if c["mode"] == "act":
        return f"act:{c['site']}:{c['pos']}:{c['layer']}"
    return f"w:{c['kind']}:{c['comp']}:{c['layer']}"


def _posmask(prep: Prepared, ps: str) -> torch.Tensor:
    if ps == "image":
        return prep.image_mask
    if ps == "text":
        return ~prep.image_mask
    return torch.ones_like(prep.image_mask)


@torch.no_grad()
def run_caption(lm, prep: Prepared, configs: Sequence[dict], batch: int = 8) -> dict[str, dict]:
    """lp for every configuration on one prepared caption. Returns key -> {lp, top}."""
    res: dict[str, dict] = {}
    acts = [c for c in configs if c["mode"] == "act"]
    for i in range(0, len(acts), batch):
        chunk = acts[i:i + batch]
        plan: dict = {}
        for r, c in enumerate(chunk):
            plan.setdefault((c["site"], c["layer"]), []).append((r, _posmask(prep, c["pos"])))
        with quant_mode(lm.model, True):
            lp, top = decoder_logprobs(lm, prep, B=len(chunk), plan=plan)
        for r, c in enumerate(chunk):
            res[config_key(c)] = {"lp": lp[r].cpu(), "top": top[r].cpu()}
    for c in (c for c in configs if c["mode"] == "w"):
        mods = fq_modules(lm, c["comp"], [c["layer"]])
        base = c["kind"] == "damage"            # damage: FP16 everywhere but here
        with quant_mode(lm.model, not base), weights_override(mods, base):
            lp, top = decoder_logprobs(lm, prep, B=1)
        res[config_key(c)] = {"lp": lp[0].cpu(), "top": top[0].cpu()}
    return res


@torch.no_grad()
def prepare(lm, inputs: dict, prompt_len: int, tags: list[str],
            sites_layers: Sequence) -> Prepared:
    """inputs: teacher-forced prompt+caption (batch 1); prompt_len: index of the first
    caption token. Captures embeddings, FP16 activations and both baselines."""
    ids = inputs["input_ids"][0]
    emb, pos = capture_decoder_inputs(lm, inputs)
    n = ids.shape[0] - prompt_len
    prep = Prepared(embeds=emb, position_ids=pos,
                    image_mask=(ids == lm.image_token_id),
                    tgt_pos=torch.arange(prompt_len, prompt_len + n, device=ids.device),
                    targets=ids[prompt_len:], tags=tags)
    with quant_mode(lm.model, False):
        lp_fp, top_fp = decoder_logprobs(lm, prep, capture=True, sites_layers=sites_layers)
    with quant_mode(lm.model, True):
        lp_q, top_q = decoder_logprobs(lm, prep)
    prep.lp_fp, prep.top_fp = lp_fp[0].cpu(), top_fp[0].cpu()
    prep.lp_q, prep.top_q = lp_q[0].cpu(), top_q[0].cpu()
    return prep


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def caption_sums(prep: Prepared, res: dict[str, dict]) -> dict:
    """Per-caption, per-class sufficient statistics (sums), for ratio-of-sums + bootstrap."""
    tags = prep.tags
    out = {"n": {}, "gap": {}, "flip_q": {}, "cfg": {}}
    sel = {k: torch.tensor([t == k for t in tags], dtype=torch.bool) for k in CLASSES}
    for k, m in sel.items():
        out["n"][k] = int(m.sum())
        out["gap"][k] = float((prep.lp_fp - prep.lp_q)[m].sum())
        # decision flips: FP16 picks the target, the quantized model does not
        out["flip_q"][k] = int((prep.top_fp & ~prep.top_q)[m].sum())
    for key, r in res.items():
        out["cfg"][key] = {k: [float((r["lp"] - prep.lp_q)[m].sum()),
                              int((prep.top_fp & ~r["top"])[m].sum())]
                           for k, m in sel.items()}
    return out


def _arrays(per_caption: list[dict], keys: Sequence[str]):
    import numpy as np
    den = {c: np.array([pc["gap"][c] for pc in per_caption]) for c in CLASSES}
    cnt = {c: np.array([pc["n"][c] for pc in per_caption], dtype=float) for c in CLASSES}
    flip = {c: np.array([pc["flip_q"][c] for pc in per_caption], dtype=float) for c in CLASSES}
    num = {k: {c: np.array([pc["cfg"][k][c][0] for pc in per_caption]) for c in CLASSES}
           for k in keys}
    return den, cnt, flip, num


def _ci(pt: float, boots) -> list[float]:
    import numpy as np
    b = boots[np.isfinite(boots)]
    if b.size == 0:
        return [float(pt), float("nan"), float("nan")]
    lo, hi = np.quantile(b, [0.025, 0.975])
    return [float(pt), float(lo), float(hi)]


def _boot_idx(N: int, n_boot: int, seed: int):
    import numpy as np
    return np.random.default_rng(seed).integers(0, N, size=(n_boot, N))


def _ratio(n, d, idx=None):
    import numpy as np
    with np.errstate(divide="ignore", invalid="ignore"):
        if idx is None:
            return n.sum() / d.sum() if abs(d.sum()) > 1e-9 else float("nan")
        ds = d[idx].sum(1)
        return np.where(np.abs(ds) > 1e-9, n[idx].sum(1) / ds, np.nan)


def restoration(per_caption: list[dict], key: str, cls: str, damage: bool = False) -> float:
    den, _, _, num = _arrays(per_caption, [key])
    r = _ratio(num[key][cls], den[cls])
    return 1.0 - r if damage else r


def summarize(per_caption: list[dict], keys: Sequence[str], n_boot: int = 2000,
              seed: int = 0) -> dict:
    """R (D for weight-damage) per configuration for obj / other tokens and their
    difference: [point, lo, hi], bootstrap over captions (same resamples everywhere)."""
    den, _, _, num = _arrays(per_caption, keys)
    idx = _boot_idx(len(per_caption), n_boot, seed)
    out = {}
    for key in keys:
        sgn = key.startswith("w:damage")
        f = (lambda r: 1.0 - r) if sgn else (lambda r: r)
        row = {}
        pts, bs = {}, {}
        for cls in ("obj", "other", "hall"):
            pts[cls] = f(_ratio(num[key][cls], den[cls]))
            bs[cls] = f(_ratio(num[key][cls], den[cls], idx))
            row[cls] = _ci(pts[cls], bs[cls])
        row["obj_minus_other"] = _ci(pts["obj"] - pts["other"], bs["obj"] - bs["other"])
        out[key] = row
    return out


def gap_summary(per_caption: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """How much log-prob quantization costs per token, by class (the omission
    signature), and how often it flips FP16's top-1 choice."""
    den, cnt, flip, _ = _arrays(per_caption, [])
    idx = _boot_idx(len(per_caption), n_boot, seed)
    out = {}
    for c in CLASSES:
        out[c] = {"n_tokens": int(cnt[c].sum()),
                  "gap_per_token": _ci(_ratio(den[c], cnt[c]), _ratio(den[c], cnt[c], idx)),
                  "flip_rate": _ci(_ratio(flip[c], cnt[c]), _ratio(flip[c], cnt[c], idx))}
    out["obj_minus_other_gap"] = _ci(
        _ratio(den["obj"], cnt["obj"]) - _ratio(den["other"], cnt["other"]),
        _ratio(den["obj"], cnt["obj"], idx) - _ratio(den["other"], cnt["other"], idx))
    out["obj_over_other_flip"] = _ci(
        _ratio(flip["obj"], cnt["obj"]) / _ratio(flip["other"], cnt["other"]),
        _ratio(flip["obj"], cnt["obj"], idx) / _ratio(flip["other"], cnt["other"], idx))
    return out
