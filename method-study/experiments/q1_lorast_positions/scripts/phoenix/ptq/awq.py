"""AWQ: activation-aware weight quantization (Lin et al., MLSys 2024).

Per decoder layer and per group of Linears that share an input x:

  1. scale search. With a_j = mean_t |x_tj| (average activation magnitude of input
     channel j), try s = a^r for r in {0, 1/20, ..., 19/20} (normalised so
     sqrt(max s * min s) = 1), quantize W diag(s), divide back, and keep the r that
     minimises || block(x; W) - block(x; Q(W diag s) diag(1/s)) ||^2, where `block` is
     the module whose output matters (the whole attention for q/k/v, the whole MLP for
     gate/up, the Linear itself for o_proj and down_proj). r = 0 is plain RTN, so the
     search can only improve on RTN *on the calibration data*.
     The chosen s is folded into the producing op (RMSNorm gain, v_proj rows, up_proj
     rows): exact in full precision.
  2. clip search. For every Linear except q/k (the reference skips them: attention
     scores are sensitive to their range), per output row and weight group, try
     shrinking the clipping range to (1 - i/20) of the absmax, i < 10, and keep the one
     minimising the group's output error on sampled calibration tokens.

Calibration inputs come from the full-precision model (the reference implementation
does the same). The clipped, quantized weight is stored as `phoenix_wq`; the FP
weight stays unclipped so the FP16 reference of LoRAS / probes is the exact model.

Deviation from llm-awq, by design: none in the algorithm; the calibration data is
ours (see calib.py) and the quantizer is our RTN, so AWQ vs RTN differs *only* by
AWQ's search.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..quant import quantize_weight
from .engine import (GROUPS, LayerInput, can_fuse_vo, layer_forward, progress, propagate, sub)
from .transforms import scale_into

N_GRID = 20
MAX_SHRINK = 0.5


def _skip(name: str, cfg) -> bool:
    return any(s in name for s in cfg.skip_names)


@torch.no_grad()
def _capture(layer, inps: list[LayerInput]):
    """One FP pass: attention call kwargs, o_proj / mlp / down_proj inputs, outputs."""
    attn_calls, o_in, mlp_in, down_in, outs = [], [], [], [], []
    sa, mlp = sub(layer, "self_attn"), sub(layer, "mlp")
    hs = [
        sa.register_forward_pre_hook(
            lambda m, a, k: attn_calls.append((a, dict(k))), with_kwargs=True),
        sub(layer, "self_attn.o_proj").register_forward_pre_hook(
            lambda m, a: o_in.append(a[0].reshape(-1, a[0].shape[-1]))),
        mlp.register_forward_pre_hook(lambda m, a: mlp_in.append(a[0])),
        sub(layer, "mlp.down_proj").register_forward_pre_hook(
            lambda m, a: down_in.append(a[0].reshape(-1, a[0].shape[-1]))),
    ]
    try:
        for x in inps:
            outs.append(layer_forward(layer, x))
    finally:
        for h in hs:
            h.remove()
    return attn_calls, torch.cat(o_in), mlp_in, torch.cat(down_in), outs


def _attn_out(sa, call):
    a, k = call
    r = sa(*a, **k)
    return r[0] if isinstance(r, (tuple, list)) else r


@torch.no_grad()
def _search_scale(fcs, x_stat, block_fn, ref, cfg):
    """Return (best scales, best loss, loss at r=0)."""
    orig = [fc.weight.data.clone() for fc in fcs]
    best, best_s, rtn_loss = float("inf"), None, None
    for i in range(N_GRID):
        r = i / N_GRID
        s = x_stat.pow(r).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        for fc, w in zip(fcs, orig):
            fc.weight.data = (quantize_weight((w.float() * s).to(w.dtype), cfg.w_bits,
                                              cfg.group_size, cfg.w_symmetric).float()
                              / s).to(w.dtype)
        loss = block_fn(ref)
        if i == 0:
            rtn_loss = loss
        if loss < best:
            best, best_s = loss, s.clone()
        for fc, w in zip(fcs, orig):
            fc.weight.data = w
    return best_s, best, rtn_loss


def _mse_list(outs, refs):
    num = sum(float((o.float() - r.float()).pow(2).sum()) for o, r in zip(outs, refs))
    den = sum(r.numel() for r in refs)
    return num / den


@torch.no_grad()
def _search_clip(w: torch.Tensor, x: torch.Tensor, cfg, oc_chunk: int = 256) -> torch.Tensor:
    """Per (row, group) clipping absmax minimising that group's output error."""
    oc, ic = w.shape
    gs = cfg.group_size if cfg.group_size and cfg.group_size > 0 else ic
    gs = gs if ic % gs == 0 else ic
    G = ic // gs
    xg = x.float().reshape(-1, G, gs)
    best_all = []
    for o0 in range(0, oc, oc_chunk):
        wc = w[o0:o0 + oc_chunk].float()
        wg = wc.reshape(-1, G, gs)
        ref = torch.einsum("tgs,ogs->otg", xg, wg)
        org_max = wg.abs().amax(-1, keepdim=True)
        best_max = org_max.clone()
        min_err = torch.full(org_max.shape[:2], float("inf"), device=w.device)
        for i in range(int(MAX_SHRINK * N_GRID)):
            mx = org_max * (1 - i / N_GRID)
            cur = torch.maximum(torch.minimum(wg, mx), -mx).reshape(wc.shape)
            q = quantize_weight(cur, cfg.w_bits, gs, cfg.w_symmetric).reshape(-1, G, gs)
            err = (torch.einsum("tgs,ogs->otg", xg, q) - ref).pow(2).mean(1)
            better = err < min_err
            min_err = torch.where(better, err, min_err)
            best_max = torch.where(better[..., None], mx, best_max)
        best_all.append(best_max)
    return torch.cat(best_all, 0)                     # [oc, G, 1]


@torch.no_grad()
def awq(lm, inps: list[LayerInput], cfg, n_clip_tokens: int = 2048, seed: int = 0) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    stats = {"layers": []}
    for li, layer in enumerate(progress(lm.layers, "[awq]")):
        attn_calls, o_in, mlp_in, down_in, outs = _capture(layer, inps)
        sa, mlp = sub(layer, "self_attn"), sub(layer, "mlp")
        row = {}

        # 1. q/k/v  (inspect: the whole attention block)
        names = GROUPS[0][1]
        if not any(_skip(n, cfg) for n in names):
            xa = torch.cat([c[1].get("hidden_states", c[0][0] if c[0] else None)
                            .reshape(-1, o_in.shape[-1]) for c in attn_calls])
            ref = [_attn_out(sa, c) for c in attn_calls]
            fcs = [sub(layer, n) for n in names]
            s, l, l0 = _search_scale(fcs, xa.abs().float().mean(0), lambda r: _mse_list(
                [_attn_out(sa, c) for c in attn_calls], r), ref, cfg)
            scale_into(sub(layer, "input_layernorm"), fcs, s)
            row["qkv"] = (l0, l)
            del xa, ref

        # 2. o_proj  (prev: v_proj)
        if can_fuse_vo(layer) and not _skip("self_attn.o_proj", cfg):
            o = sub(layer, "self_attn.o_proj")
            ref = o(o_in)
            s, l, l0 = _search_scale([o], o_in.abs().float().mean(0),
                                     lambda r: _mse_list([o(o_in)], [r]), ref, cfg)
            scale_into(sub(layer, "self_attn.v_proj"), [o], s)
            row["o"] = (l0, l)

        # 3. gate/up  (inspect: the whole MLP)
        if not any(_skip(n, cfg) for n in GROUPS[2][1]):
            xm = torch.cat([x.reshape(-1, x.shape[-1]) for x in mlp_in])
            ref = mlp(xm)
            fcs = [sub(layer, n) for n in GROUPS[2][1]]
            s, l, l0 = _search_scale(fcs, xm.abs().float().mean(0),
                                     lambda r: _mse_list([mlp(xm)], [r]), ref, cfg)
            scale_into(sub(layer, "post_attention_layernorm"), fcs, s)
            row["mlp"] = (l0, l)
            del xm, ref

        # 4. down_proj  (prev: up_proj)
        if not _skip("mlp.down_proj", cfg):
            d = sub(layer, "mlp.down_proj")
            ref = d(down_in)
            s, l, l0 = _search_scale([d], down_in.abs().float().mean(0),
                                     lambda r: _mse_list([d(down_in)], [r]), ref, cfg)
            scale_into(sub(layer, "mlp.up_proj"), [d], s)
            row["down"] = (l0, l)
        del attn_calls, o_in, mlp_in, down_in

        # clip search on the *scaled* inputs
        feats: dict[str, list] = {}
        names = ["self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                 "mlp.down_proj"]
        hs = [sub(layer, n).register_forward_pre_hook(
            lambda m, a, n=n: feats.setdefault(n, []).append(a[0].reshape(-1, a[0].shape[-1])))
            for n in names]
        try:
            for x in inps:
                layer_forward(layer, x)
        finally:
            for h in hs:
                h.remove()
        for n in names + ["self_attn.q_proj", "self_attn.k_proj"]:
            fc = sub(layer, n)
            if _skip(n, cfg):
                continue
            if n in feats:
                x = torch.cat(feats[n])
                idx = torch.randperm(x.shape[0], generator=g)[:n_clip_tokens].to(x.device)
                mx = _search_clip(fc.weight.data, x[idx], cfg)
                oc, ic = fc.weight.shape
                G = mx.shape[1]
                wg = fc.weight.data.float().reshape(oc, G, -1)
                w = torch.maximum(torch.minimum(wg, mx), -mx).reshape(oc, ic)
            else:                                        # q/k: scaled, not clipped
                w = fc.weight.data.float()
            fc.phoenix_wq = quantize_weight(w.to(fc.weight.dtype), cfg.w_bits,
                                            cfg.group_size, cfg.w_symmetric)
        del feats
        inps[:] = [LayerInput(o, x.kwargs, x.image_mask) for o, x in zip(outs, inps)]
        stats["layers"].append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gains = [v[1] / max(v[0], 1e-20) for r in stats["layers"] for v in r.values()]
    stats["mean_loss_ratio_vs_rtn"] = sum(gains) / max(len(gains), 1)
    print(f"[awq] scale search: block MSE {stats['mean_loss_ratio_vs_rtn']:.3f}x of RTN's "
          "on calibration data (mean over groups)", flush=True)
    return stats
