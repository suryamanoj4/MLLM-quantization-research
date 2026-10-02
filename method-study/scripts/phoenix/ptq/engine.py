"""Layer-by-layer calibration engine shared by AWQ, GPTQ and SmoothQuant.

GPTQ and AWQ never run the whole 7B model on the calibration set at once. They
capture the hidden states entering decoder layer 0, then walk the stack: process
layer i on its inputs, run layer i to produce the inputs of layer i+1, repeat. Peak
memory is one layer plus the calibration activations (~0.4 GB for 64 x 640 tokens).

Captures are taken with forward hooks and replayed with the exact arguments the model
used, so the engine does not depend on a transformers version's attention signature.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn


class _Stop(Exception):
    pass


@dataclass
class LayerInput:
    hidden: torch.Tensor                  # [1, T, d]
    kwargs: dict                          # everything else the decoder layer received
    image_mask: torch.Tensor | None = None  # [T] bool


def _clean_kwargs(kw: dict) -> dict:
    out = {}
    for k, v in kw.items():
        if k in ("past_key_value", "past_key_values"):
            out[k] = None
        elif k == "use_cache":
            out[k] = False
        elif k == "output_attentions":
            out[k] = False
        else:
            out[k] = v
    return out


@torch.no_grad()
def capture_layer0(lm, seqs: list[dict]) -> list[LayerInput]:
    layer0 = lm.layers[0]
    store: list[LayerInput] = []

    def pre(module, args, kwargs):
        h = args[0] if args else kwargs.pop("hidden_states")
        kw = dict(kwargs)
        kw.pop("hidden_states", None)
        store.append(LayerInput(h.detach(), _clean_kwargs(kw)))
        raise _Stop

    hk = layer0.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        for s in seqs:
            try:
                lm.model(**s, use_cache=False)
            except _Stop:
                pass
            store[-1].image_mask = (s["input_ids"][0] == lm.image_token_id)
    finally:
        hk.remove()
    return store


def layer_forward(layer: nn.Module, inp: LayerInput) -> torch.Tensor:
    r = layer(inp.hidden, **inp.kwargs)
    return r[0] if isinstance(r, (tuple, list)) else r


@torch.no_grad()
def propagate(layer: nn.Module, inps: list[LayerInput]) -> list[LayerInput]:
    return [LayerInput(layer_forward(layer, x), x.kwargs, x.image_mask) for x in inps]


def linear_input(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """What the quantizer will actually see: the (rotated) 2-D input of a Linear."""
    x = x.reshape(-1, x.shape[-1])
    rot = getattr(module, "phoenix_in_rot", None)
    if rot is not None:
        x = x @ rot
    return x


@contextmanager
def input_hooks(modules: dict[str, nn.Module], fn):
    """Call fn(name, module, x2d) with every input a named Linear receives."""
    hs = []
    for name, m in modules.items():
        def pre(mod, args, name=name):
            fn(name, mod, linear_input(mod, args[0]))
        hs.append(m.register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


# --------------------------------------------------------------------------- #
# Llama-family layer anatomy
# --------------------------------------------------------------------------- #
GROUPS = [  # (prev op that can absorb a per-channel scale, linears sharing that input)
    ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    ("self_attn.v_proj", ["self_attn.o_proj"]),
    ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
    ("mlp.up_proj", ["mlp.down_proj"]),
]
ALL_LINEARS = [n for _, ls in GROUPS for n in ls]


def sub(layer: nn.Module, name: str) -> nn.Module:
    return layer.get_submodule(name)


def can_fuse_vo(layer: nn.Module) -> bool:
    """v_proj -> o_proj scaling is exact only without grouped-query attention."""
    v, o = sub(layer, "self_attn.v_proj"), sub(layer, "self_attn.o_proj")
    return v.out_features == o.in_features


def progress(it, desc):
    import sys
    from tqdm import tqdm
    return tqdm(it, desc=desc, unit="layer", file=sys.stderr, dynamic_ncols=True,
                mininterval=0.5 if sys.stderr.isatty() else 30.0)
