"""Function-preserving weight transforms: SmoothQuant scaling and QuaRot-style rotation.

Both rewrite weights so the full-precision model computes exactly the same function,
but the tensors the quantizer sees become easier to quantize.

SmoothQuant (Xiao et al., 2023). For a Linear y = x W^T with input channel j, divide
x_j by s_j and multiply column j of W by s_j:  y = (x / s)(W diag(s))^T. Choosing
    s_j = max|x_j|^alpha / max|W_:,j|^(1-alpha)
moves the activation outliers into the weights. The 1/s is folded into the op that
produced x (the RMSNorm weight for q/k/v and gate/up), so it costs nothing at runtime.

Rotation (QuaRot, Ashkboos et al., 2024). For an orthogonal R, y = (x R)(W R)^T. A
random Hadamard R spreads a single huge channel over all channels, so the per-token
max -- which sets the activation scale -- drops by up to sqrt(d). We apply R online to
every quantized Linear's input (a simulation of QuaRot's fused residual rotation plus
its online Hadamards) after folding the RMSNorm gain into the following weights, so
the rotated tensor is exactly the one QuaRot quantizes for q/k/v and gate/up.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import GROUPS, can_fuse_vo, input_hooks, sub


# --------------------------------------------------------------------------- #
# scale folding (shared with AWQ)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def scale_into(prev: nn.Module, fcs: list[nn.Linear], s: torch.Tensor) -> None:
    """Divide prev's output channels by s and multiply the fcs' input columns by s."""
    s = s.float()
    if isinstance(prev, nn.Linear):
        prev.weight.copy_((prev.weight.float() / s[:, None]).to(prev.weight.dtype))
        if prev.bias is not None:
            prev.bias.copy_((prev.bias.float() / s).to(prev.bias.dtype))
    else:                                               # RMSNorm / LayerNorm gain
        prev.weight.copy_((prev.weight.float() / s).to(prev.weight.dtype))
        if getattr(prev, "bias", None) is not None:
            prev.bias.copy_((prev.bias.float() / s).to(prev.bias.dtype))
    for fc in fcs:
        fc.weight.copy_((fc.weight.float() * s[None, :]).to(fc.weight.dtype))


# --------------------------------------------------------------------------- #
# SmoothQuant
# --------------------------------------------------------------------------- #
@torch.no_grad()
def activation_absmax(lm, seqs, names=("self_attn.q_proj", "self_attn.o_proj",
                                       "mlp.gate_proj", "mlp.down_proj")) -> dict:
    stats: dict = {}
    mods = {f"{i}.{n}": sub(layer, n) for i, layer in enumerate(lm.layers) for n in names}

    def fn(key, m, x):
        a = x.abs().amax(0).float()
        stats[key] = a if key not in stats else torch.maximum(stats[key], a)

    with input_hooks(mods, fn):
        for s in seqs:
            lm.model(**s, use_cache=False)
    return stats


@torch.no_grad()
def smoothquant(lm, seqs, alpha: float = 0.85, all_linears: bool = False) -> dict:
    """Faithful to the reference Llama recipe: smooth q/k/v and gate/up (the inputs
    that come out of an RMSNorm). all_linears=True also smooths o_proj (into v_proj)
    and down_proj (into up_proj)."""
    stats = activation_absmax(lm, seqs)
    first = {"input_layernorm": "self_attn.q_proj", "post_attention_layernorm": "mlp.gate_proj",
             "self_attn.v_proj": "self_attn.o_proj", "mlp.up_proj": "mlp.down_proj"}
    info = {"alpha": alpha, "max_scale": 0.0}
    for i, layer in enumerate(lm.layers):
        for prev_name, fc_names in GROUPS:
            if not all_linears and prev_name in ("self_attn.v_proj", "mlp.up_proj"):
                continue
            if prev_name == "self_attn.v_proj" and not can_fuse_vo(layer):
                continue
            fcs = [sub(layer, n) for n in fc_names]
            ax = stats[f"{i}.{first[prev_name]}"].clamp(min=1e-5)
            wx = torch.cat([fc.weight.abs().float() for fc in fcs], 0).amax(0).clamp(min=1e-5)
            s = (ax.pow(alpha) / wx.pow(1 - alpha)).clamp(min=1e-5)
            scale_into(sub(layer, prev_name), fcs, s)
            info["max_scale"] = max(info["max_scale"], float(s.max()))
    return info


# --------------------------------------------------------------------------- #
# rotation
# --------------------------------------------------------------------------- #
def _sylvester(n: int, device) -> torch.Tensor:
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def random_rotation(n: int, seed: int, device) -> torch.Tensor:
    """Randomised Hadamard for n = 2^k; otherwise (randomised Hadamard of the largest
    power-of-two factor) kron (Haar-random orthogonal of the odd remainder), e.g.
    11008 = 256 x 43. Both are orthogonal and mix every channel with every other."""
    g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + n)
    p2 = n & (-n)
    m = n // p2
    signs = (torch.randint(0, 2, (p2,), generator=g) * 2 - 1).float().to(device)
    h = _sylvester(p2, device) * signs[None, :] / math.sqrt(p2)
    if m == 1:
        return h
    a = torch.randn(m, m, generator=g).to(device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None, :]
    return torch.kron(h.contiguous(), q.contiguous())


class RotatedLinear(nn.Linear):
    """nn.Linear whose input is rotated first: y = (x R) W'^T with W' = W R."""

    def forward(self, x):                                    # noqa: D401
        return F.linear(x @ self.phoenix_in_rot, self.weight, self.bias)


@torch.no_grad()
def rotate_linear(fc: nn.Linear, R: torch.Tensor, R_cast: torch.Tensor) -> None:
    """R in float32 for the weight product; R_cast (compute dtype) is shared by every
    Linear of that width, so a 7B model holds one 11008^2 matrix, not 32."""
    fc.weight.copy_((fc.weight.float() @ R).to(fc.weight.dtype))
    fc.register_buffer("phoenix_in_rot", R_cast, persistent=False)
    fc.__class__ = RotatedLinear


@torch.no_grad()
def fold_norm(norm: nn.Module, fcs: list[nn.Linear]) -> None:
    g = norm.weight.float()
    for fc in fcs:
        fc.weight.copy_((fc.weight.float() * g[None, :]).to(fc.weight.dtype))
    norm.weight.fill_(1.0)


@torch.no_grad()
def rotate(lm, seed: int = 0) -> dict:
    layers = lm.layers
    dev = next(layers.parameters()).device
    dtype = next(layers.parameters()).dtype
    rots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def R(n):
        if n not in rots:
            r = random_rotation(n, seed, dev)
            rots[n] = (r, r.to(dtype))
        return rots[n]

    for layer in layers:
        fold_norm(sub(layer, "input_layernorm"),
                  [sub(layer, n) for n in GROUPS[0][1]])
        fold_norm(sub(layer, "post_attention_layernorm"),
                  [sub(layer, n) for n in GROUPS[2][1]])
        for _, names in GROUPS:
            for n in names:
                fc = sub(layer, n)
                rotate_linear(fc, *R(fc.in_features))
    return {"rotations": {n: "hadamard" if n & (n - 1) == 0 else "hadamard (x) orthogonal"
                          for n in rots}}
