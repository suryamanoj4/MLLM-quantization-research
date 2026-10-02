"""Contribution 3 -- A-CAB: Adaptive Cross-Modal Attention Biasing.

Three changes to the proposal, all load-bearing.

1. THE GATE AS WRITTEN IS NOT CAUSAL. Eq. 7 computes E_t from the LM logits at
   step t, and Eq. 8 uses E_t to set beta_t, which changes the attention that
   *produces* those logits. You cannot evaluate it in one pass. Two honest
   resolutions, both implemented:
     gate="prev"     -- use E_{t-1} (free, single pass, exploits the strong
                        autocorrelation of predictive entropy along a caption);
     gate="two_pass" -- forward with delta=0, read E_t, crop the KV cache, re-forward
                        with delta. Exact "current-step" semantics, ~2x decode cost on
                        gated steps only. Use it as the quality ceiling / ablation.

2. MULTIPLICATIVE LOGIT SCALING IS ILL-POSED. Eq. 5 sets S*_vis = beta * S_vis with
   beta >= 1. Attention logits are signed: for a visual key with S_vis < 0, beta > 1
   makes it *more* negative, i.e. the "restorative" multiplier actively suppresses
   exactly the visual tokens the model already ignores. It sharpens the within-visual
   distribution rather than reallocating mass from text to vision, and the sign of
   its effect on total visual mass depends on the logit distribution.

   Replace it with an ADDITIVE bias on the visual logits, S*_vis = S_vis + delta_t.
   Then, writing m_t for the total visual attention mass,

        logit(m*_t) = logit(m_t) + delta_t        (exactly, for any distribution)

   so delta_t is a clean, monotone, interpretable shift of the *log-odds of attending
   to the image*, and the ranking inside the visual block is preserved. The gate
   becomes delta_t = min(delta_max, alpha * ReLU(E_t - tau)).

   It is also far cheaper: an additive bias on a subset of keys is exactly an additive
   attention mask, so it rides the existing mask argument and works unchanged under
   SDPA / FlashAttention. No custom kernel, no eager fallback, and the proposal's
   "<1.5% latency" claim becomes achievable rather than aspirational. The
   multiplicative variant is kept as `mode="mult"` for a faithful ablation.

3. tau IN RAW NATS DOES NOT TRANSFER. Entropy scale depends on vocabulary size and
   model calibration. `calibrate_tau` sets tau to a percentile of the model's own
   decode-time entropy distribution, so "intervene on the top 40% most uncertain
   steps" means the same thing across the precision ladder.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loras import _attn_of, steer_mask


# --------------------------------------------------------------------------- #
# controller
# --------------------------------------------------------------------------- #
@dataclass
class ACABConfig:
    mode: str = "add"              # "add" | "mult" | "off"
    gate: str = "prev"             # "prev" | "two_pass" | "none" (static)
    alpha: float = 1.0
    tau: float = 1.5
    delta_max: float = 3.0
    layers: tuple[int, ...] = ()   # empty = all layers
    heads: dict[int, list[int]] = field(default_factory=dict)   # {} = all heads
    apply_on_prefill: bool = False
    # The first answer token is produced by the prefill, which decode-only A-CAB never
    # touches -- and for POPE that token IS the answer. With first_token=True the
    # prompt's last token is replayed as a decode step so A-CAB can act on it.
    first_token: bool = True


class ACABController:
    """Holds per-step state and injects the bias into every patched attention module."""

    def __init__(self, cfg: ACABConfig, n_layers: int, n_heads: int):
        self.cfg = cfg
        self.n_layers, self.n_heads = n_layers, n_heads
        self.active = False
        self.vis_cols: torch.Tensor | None = None      # bool [B, P]
        self.delta: torch.Tensor | None = None         # float [B]
        self.layer_set = set(cfg.layers) if cfg.layers else set(range(n_layers))
        self._head_masks: dict[int, torch.Tensor] = {}

    def head_mask(self, layer_idx: int, device, dtype) -> torch.Tensor | None:
        if not self.cfg.heads:
            return None
        if layer_idx not in self._head_masks:
            hs = self.cfg.heads.get(layer_idx, [])
            m = torch.zeros(self.n_heads, dtype=dtype, device=device)
            if hs:
                m[torch.tensor(hs, device=device)] = 1.0
            self._head_masks[layer_idx] = m
        return self._head_masks[layer_idx]

    def set_step(self, vis_cols: torch.Tensor | None, delta: torch.Tensor | None):
        self.vis_cols, self.delta = vis_cols, delta

    def wants(self, layer_idx: int, q_len: int) -> bool:
        if not self.active or self.cfg.mode == "off":
            return False
        if layer_idx not in self.layer_set:
            return False
        if q_len > 1 and not self.cfg.apply_on_prefill:
            return False
        if self.vis_cols is None or self.delta is None:
            return False
        return bool((self.delta != 0).any())


# --------------------------------------------------------------------------- #
# additive path: bias injected through the attention mask
# --------------------------------------------------------------------------- #
def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _kv_len(bound: dict, extra: dict, q_len: int, layer_idx: int) -> int | None:
    """Key length for this layer's attention, used only when no mask exists.

    transformers <= 4.5x passes `cache_position` and `past_key_value`;
    5.x passes `past_key_values` and tucks cache_position into **kwargs (or drops it).
    The cache must be queried for THIS layer: by the time layer l > 0 runs, layer 0
    has already appended the new token, so get_seq_length() (which defaults to
    layer 0) over-counts by q_len.
    """
    cp = _first_not_none(bound.get("cache_position"), extra.get("cache_position"))
    if cp is not None and cp.numel() > 0:
        return int(cp[-1].item()) + 1
    pkv = _first_not_none(bound.get("past_key_values"), bound.get("past_key_value"),
                          extra.get("past_key_values"), extra.get("past_key_value"))
    if pkv is not None:
        try:
            return int(pkv.get_seq_length(layer_idx)) + q_len
        except Exception:
            pass
    return None


def _build_bias(ctrl: ACABController, layer_idx: int, B: int, q_len: int,
                kv_len: int, device, dtype) -> torch.Tensor:
    cols = ctrl.vis_cols.to(device)
    if cols.shape[-1] < kv_len:
        cols = F.pad(cols, (0, kv_len - cols.shape[-1]), value=False)
    else:
        cols = cols[:, :kv_len]
    delta = ctrl.delta.to(device=device, dtype=torch.float32).view(B, 1, 1, 1)
    bias = cols[:, None, None, :].to(torch.float32) * delta          # [B,1,1,kv]
    hm = ctrl.head_mask(layer_idx, device, torch.float32)
    if hm is not None:
        bias = bias * hm.view(1, -1, 1, 1)                           # [B,H,1,kv]
    if q_len > 1:
        bias = bias.expand(-1, -1, q_len, -1)
    return bias.to(dtype)


def _normalise_mask(am, dtype):
    """Return a 4D *additive float* mask, or None if there is nothing to merge.

    Handles the three forms transformers has used: additive float 4D (eager, and
    sdpa in 4.x), boolean 4D with True = attend (sdpa in 5.x), and a 2D padding mask.
    Adding a float bias to a boolean mask would silently turn "attend" into +1.0.
    """
    if am is None:
        return None
    if am.dim() == 2:                                        # padding mask [B, kv]
        am = am[:, None, None, :].bool() if am.dtype != torch.bool else am[:, None, None, :]
    if am.dim() != 4:
        return None
    if am.dtype == torch.bool:
        out = torch.zeros(am.shape, dtype=dtype, device=am.device)
        return out.masked_fill(~am, torch.finfo(dtype).min)
    return am.to(dtype) if am.dtype != dtype else am


def patch_attention_additive(layers: nn.ModuleList, ctrl: ACABController) -> Callable[[], None]:
    """Wrap self_attn.forward so the visual keys receive +delta before softmax."""
    originals = []

    for li, layer in enumerate(layers):
        attn = _attn_of(layer)
        orig = attn.forward
        sig = inspect.signature(orig)
        varkw = next((p.name for p in sig.parameters.values()
                      if p.kind is inspect.Parameter.VAR_KEYWORD), None)
        layer_idx = getattr(attn, "layer_idx", li)
        originals.append((attn, orig))

        def make(orig=orig, sig=sig, li=li, varkw=varkw, layer_idx=layer_idx):
            def fwd(*args, **kwargs):
                try:
                    bound = sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    b = bound.arguments
                except TypeError:
                    return orig(*args, **kwargs)

                hs = b.get("hidden_states")
                if hs is None or not ctrl.wants(li, hs.shape[1]):
                    return orig(*args, **kwargs)

                B, q_len = hs.shape[0], hs.shape[1]
                dtype, device = hs.dtype, hs.device
                extra = (b.get(varkw) or {}) if varkw else {}

                # An existing mask is authoritative: the bias must match its width
                # exactly (eager slices it to the key length afterwards anyway).
                base = _normalise_mask(b.get("attention_mask"), dtype)
                kv_len = (int(base.shape[-1]) if base is not None
                          else _kv_len(b, extra, q_len, layer_idx))
                if kv_len is None:
                    return orig(*args, **kwargs)

                bias = _build_bias(ctrl, li, B, q_len, kv_len, device, dtype)
                new_mask = bias if base is None else base + bias
                bound.arguments["attention_mask"] = new_mask
                # BoundArguments.args/.kwargs expand VAR_KEYWORD correctly;
                # orig(**bound.arguments) would nest **kwargs one level deep and
                # silently drop things like output_attentions / flash-attn kwargs.
                return orig(*bound.args, **bound.kwargs)
            return fwd

        attn.forward = make()

    def restore():
        for attn, orig in originals:
            attn.forward = orig
    return restore


# --------------------------------------------------------------------------- #
# multiplicative path (faithful to Eq. 5): custom eager attention function
# --------------------------------------------------------------------------- #
def _acab_mult_attention_factory(ctrl: ACABController):
    from transformers.models.llama.modeling_llama import repeat_kv

    def acab_mult_attention(module, query, key, value, attention_mask,
                            scaling=None, dropout=0.0, **kwargs):
        g = getattr(module, "num_key_value_groups", 1)
        k = repeat_kv(key, g) if g > 1 else key
        v = repeat_kv(value, g) if g > 1 else value
        scaling = scaling if scaling is not None else (query.shape[-1] ** -0.5)
        logits = torch.matmul(query, k.transpose(2, 3)) * scaling      # [B,H,q,kv]

        li = getattr(module, "layer_idx", -1)
        if ctrl.wants(li, query.shape[2]):
            B, H, q_len, kv_len = logits.shape
            cols = ctrl.vis_cols.to(logits.device)
            cols = (F.pad(cols, (0, kv_len - cols.shape[-1]), value=False)
                    if cols.shape[-1] < kv_len else cols[:, :kv_len])
            beta = 1.0 + ctrl.delta.to(logits.device, logits.dtype).view(B, 1, 1, 1)
            hm = ctrl.head_mask(li, logits.device, logits.dtype)
            mult = torch.where(cols[:, None, None, :], beta, torch.ones_like(beta))
            if hm is not None:
                sel = hm.view(1, -1, 1, 1)
                mult = 1.0 + (mult - 1.0) * sel
            logits = logits * mult

        if attention_mask is not None:
            logits = logits + attention_mask[:, :, :, : k.shape[-2]]
        w = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        w = F.dropout(w, p=dropout, training=module.training)
        out = torch.matmul(w, v).transpose(1, 2).contiguous()
        return out, w

    return acab_mult_attention


def install_acab(model, layers: nn.ModuleList, ctrl: ACABController):
    """Install the requested A-CAB mode. Returns an uninstall callable."""
    if ctrl.cfg.mode == "add":
        return patch_attention_additive(layers, ctrl)
    if ctrl.cfg.mode == "mult":
        from transformers.modeling_utils import AttentionInterface
        name = "acab_mult"
        AttentionInterface.register(name, _acab_mult_attention_factory(ctrl))
        # ONLY the language model: CLIP's attention in the vision tower must keep its
        # own implementation (setting it globally crashed 05_latency.py)
        prev = _set_attn_impl(model, name, language_only=True)

        def restore():
            _restore_attn_impl(prev)
        return restore
    return lambda: None


def _set_attn_impl(model, name: str, language_only: bool = False) -> list:
    """Set the attention implementation; returns [(config, previous)] for restoring."""
    if language_only:
        from .model import get_language_model
        cfgs = [get_language_model(model).config]
    else:
        cfgs = [model.config] + [getattr(m, "config", None) for m in model.modules()]
    seen, prev = set(), []
    for cfg in cfgs:
        if cfg is None or id(cfg) in seen or not hasattr(cfg, "_attn_implementation"):
            continue
        seen.add(id(cfg))
        prev.append((cfg, cfg._attn_implementation))
        cfg._attn_implementation = name
    return prev


def _restore_attn_impl(prev: list) -> None:
    for cfg, name in prev:
        cfg._attn_implementation = name


# --------------------------------------------------------------------------- #
# gating
# --------------------------------------------------------------------------- #
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def delta_from_entropy(E: torch.Tensor, cfg: ACABConfig) -> torch.Tensor:
    if cfg.mode == "off":
        return torch.zeros_like(E)
    if cfg.gate == "none":
        return torch.full_like(E, cfg.alpha)
    d = cfg.alpha * torch.relu(E - cfg.tau)
    return d.clamp(max=cfg.delta_max)


# --------------------------------------------------------------------------- #
# generation loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(lm, inputs: dict, ctrl: ACABController | None = None,
             max_new_tokens: int = 64, do_sample: bool = False,
             temperature: float = 1.0, top_p: float = 1.0,
             record: bool = False) -> dict:
    """Greedy / nucleus decoding with per-step entropy gating.

    Returns {"sequences": LongTensor [B, T], "text": [str], "trace": {...}}.
    """
    model, tok = lm.model, lm.processor.tokenizer
    input_ids = inputs["input_ids"]
    B = input_ids.shape[0]
    device = input_ids.device
    eos_id = tok.eos_token_id

    vis_cols = (input_ids == lm.image_token_id)
    cfg = ctrl.cfg if ctrl is not None else ACABConfig(mode="off")
    acting = ctrl is not None and cfg.mode != "off"

    if ctrl is not None:
        ctrl.active = True
        ctrl.set_step(vis_cols, torch.zeros(B, device=device))

    attn_mask = inputs.get("attention_mask")
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)
    trace = {"entropy": [], "delta": [], "tokens": [], "gated": [],
             "first_entropy": None, "first_delta": None}

    if acting and cfg.first_token and input_ids.shape[1] > 1:
        # Prefill all but the last prompt token, then replay that token as a decode
        # step: the first answer token then goes through the path A-CAB acts on.
        # steer_mask keeps LoRAS aligned with the shorter prefill.
        pre = dict(inputs)
        pre["input_ids"] = input_ids[:, :-1]
        pre["attention_mask"] = attn_mask[:, :-1]
        with steer_mask(vis_cols[:, :-1]):
            out = model(**pre, use_cache=True)
        step = dict(input_ids=input_ids[:, -1:], attention_mask=attn_mask,
                    past_key_values=out.past_key_values, use_cache=True)
        # no previous entropy exists for the first answer token, so gate it exactly
        past, logits, E0, d0 = _gated_step(model, step, ctrl, vis_cols, cfg,
                                           two_pass=True, prev_E=None)
        trace["first_entropy"] = E0.float().cpu().tolist()
        trace["first_delta"] = d0.float().cpu().tolist()
    else:
        with steer_mask(vis_cols):
            out = model(**inputs, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    first_logits = logits.detach().float().clone()     # [B, V]; POPE's answer + ECE

    prev_E = entropy_from_logits(logits)
    generated = torch.empty(B, 0, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        nxt = _pick(logits, do_sample, temperature, top_p)
        nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
        generated = torch.cat([generated, nxt[:, None]], dim=1)
        finished = finished | (nxt == eos_id)
        if bool(finished.all()):
            break

        attn_mask = torch.cat([attn_mask, torch.ones(B, 1, dtype=attn_mask.dtype,
                                                     device=device)], dim=1)
        step = dict(input_ids=nxt[:, None], attention_mask=attn_mask,
                    past_key_values=past, use_cache=True)
        past, logits, E, d = _gated_step(model, step, ctrl if acting else None, vis_cols,
                                         cfg, two_pass=(cfg.gate == "two_pass"),
                                         prev_E=prev_E)
        gated = d > 0
        if record:
            trace["entropy"].append(E.float().cpu().tolist())
            trace["delta"].append(d.float().cpu().tolist())
            trace["tokens"].append(nxt.cpu().tolist())
            trace["gated"].append(gated.cpu().tolist())
        prev_E = entropy_from_logits(logits)

    if ctrl is not None:
        ctrl.active = False
        ctrl.set_step(None, None)

    texts = tok.batch_decode(generated, skip_special_tokens=True)
    return {"sequences": generated, "text": [t.strip() for t in texts], "trace": trace,
            "first_logits": first_logits}


def _gated_step(model, step: dict, ctrl, vis_cols, cfg: ACABConfig,
                two_pass: bool, prev_E):
    """One decode forward with the entropy gate. Returns (past, logits, E, delta)."""
    past = step["past_key_values"]
    B = step["input_ids"].shape[0]
    device = step["input_ids"].device
    if ctrl is None:
        o = model(**step)
        E = prev_E if prev_E is not None else entropy_from_logits(o.logits[:, -1, :])
        return o.past_key_values, o.logits[:, -1, :], E, torch.zeros(B, device=device)
    if two_pass:
        cache_len = _cache_len(past)
        ctrl.set_step(vis_cols, torch.zeros(B, device=device))
        o0 = model(**step)
        l0 = o0.logits[:, -1, :]
        E = entropy_from_logits(l0)
        d = delta_from_entropy(E, cfg)
        if bool((d > 0).any()) and _crop(o0.past_key_values, cache_len):
            ctrl.set_step(vis_cols, d)
            o = model(**dict(step, past_key_values=o0.past_key_values))
            return o.past_key_values, o.logits[:, -1, :], E, d
        return o0.past_key_values, l0, E, d
    E = prev_E
    d = delta_from_entropy(E, cfg)
    ctrl.set_step(vis_cols, d)
    o = model(**step)
    return o.past_key_values, o.logits[:, -1, :], E, d


def _cache_len(past) -> int:
    try:
        return int(past.get_seq_length())
    except Exception:
        try:
            return int(past[0][0].shape[-2])
        except Exception:
            return -1


def _crop(past, length: int) -> bool:
    """Roll the cache back to `length` tokens.

    Uses the negative form, crop(-n_tokens_to_drop): positive crop(max_length) is
    deprecated in transformers 5.x and removed in 5.18, while negative crop works
    on every version from 4.38 onward.
    """
    if length < 0 or not hasattr(past, "crop"):
        return False
    drop = _cache_len(past) - length
    if drop > 0:
        past.crop(-drop)
    return True


def _pick(logits, do_sample, temperature, top_p):
    if not do_sample:
        return logits.argmax(-1)
    l = logits.float() / max(temperature, 1e-5)
    if top_p < 1.0:
        srt, idx = torch.sort(l, descending=True, dim=-1)
        cum = torch.softmax(srt, -1).cumsum(-1)
        srt[(cum - torch.softmax(srt, -1)) > top_p] = -float("inf")
        l = torch.full_like(l, -float("inf")).scatter(-1, idx, srt)
    return torch.multinomial(torch.softmax(l, -1), 1).squeeze(-1)


# --------------------------------------------------------------------------- #
# threshold calibration
# --------------------------------------------------------------------------- #
@torch.no_grad()
def calibrate_tau(lm, batches, percentile: float = 60.0,
                  max_new_tokens: int = 48) -> float:
    """tau := the given percentile of this model's own decode-time entropy."""
    vals = []
    for batch in batches:
        r = generate(lm, batch, ctrl=None, max_new_tokens=max_new_tokens, record=True)
        for row in r["trace"]["entropy"]:
            vals.extend(row)
    if not vals:
        return 1.5
    t = torch.tensor(vals)
    return float(torch.quantile(t, percentile / 100.0))


@torch.no_grad()
def measure_visual_mass(lm, inputs: dict, ctrl: ACABController | None = None,
                        max_new_tokens: int = 32,
                        layers: Sequence[int] | None = None) -> dict:
    """M_vis(t) per decoding step, averaged over the selected layers/heads.

    Needs eager attention + output_attentions, so use it on a handful of samples
    for the figures, not inside the main eval loop.
    """
    model = lm.model
    tok = lm.processor.tokenizer
    prev_impl = (_set_attn_impl(model, "eager")
                 if ctrl is None or ctrl.cfg.mode != "mult" else [])
    vis_cols = (inputs["input_ids"] == lm.image_token_id)
    B = vis_cols.shape[0]
    device = vis_cols.device

    if ctrl is not None:
        ctrl.active = True
        ctrl.set_step(vis_cols, torch.zeros(B, device=device))
    with steer_mask(vis_cols):
        out = model(**inputs, use_cache=True, output_attentions=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    attn_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
    prev_E = entropy_from_logits(logits)
    series, toks = [], []
    cfg = ctrl.cfg if ctrl is not None else ACABConfig(mode="off")

    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1)
        toks.append(nxt.cpu().tolist())
        attn_mask = torch.cat([attn_mask, torch.ones(B, 1, dtype=attn_mask.dtype,
                                                     device=device)], 1)
        d = delta_from_entropy(prev_E, cfg) if ctrl is not None else torch.zeros(B, device=device)
        if ctrl is not None:
            ctrl.set_step(vis_cols, d)
        o = model(input_ids=nxt[:, None], attention_mask=attn_mask,
                  past_key_values=past, use_cache=True, output_attentions=True)
        past, logits = o.past_key_values, o.logits[:, -1, :]
        sel = layers if layers is not None else range(len(o.attentions))
        m = []
        for li in sel:
            A = o.attentions[li].float()                       # [B,H,1,kv]
            cols = F.pad(vis_cols, (0, A.shape[-1] - vis_cols.shape[-1]), value=False)
            m.append(A[..., cols[0]].sum(-1).mean(dim=(1, 2)))
        series.append(torch.stack(m).mean(0).cpu().tolist())
        prev_E = entropy_from_logits(logits)
        if bool((nxt == tok.eos_token_id).all()):
            break

    if ctrl is not None:
        ctrl.active = False
    _restore_attn_impl(prev_impl)
    return {"mass": series, "tokens": toks}
