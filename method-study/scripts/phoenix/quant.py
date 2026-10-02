"""Simulated ("fake") post-training quantization ladder.

Why simulated rather than AutoAWQ/GPTQ kernels?

1. The proposal's Table 1 requires W8A8 / W4A8 / W4A4 / W2A4. No inference library
   exposes 4-bit *activation* quantization for MLLMs -- AWQ and GPTQ are weight-only
   (W4A16). Simulation is therefore not a shortcut, it is the only way to obtain
   those rows at all. This is standard practice in the PTQ literature
   (OmniQuant, SmoothQuant, Atom and QuaRot all report simulated A-bit results).

2. LoRAS calibration needs FP16 and quantized activations at *exactly* the same
   token positions. With a simulated quantizer both live in one model instance and
   we simply toggle a flag, which removes any risk of prefix misalignment and
   halves VRAM versus loading two checkpoints.

What is simulated:
  * weights      -- RTN, group-wise along the input dimension, asymmetric (default)
                    or symmetric, `w_bits` levels. Dequantized back to the compute
                    dtype and cached, so forward cost is a plain matmul.
  * activations   -- dynamic per-token symmetric MinMax (optionally percentile
                    clipped), `a_bits` levels, applied to the *input* of each
                    quantized Linear.

Plain per-token RTN at A4 is a *pessimistic* lower bound: real W4A4 pipelines add
rotation / outlier handling. Published methods live in `phoenix/ptq/` and plug in
through two hooks this module honours on an nn.Linear before it is wrapped:

  * `phoenix_wq`      a weight already quantized by GPTQ / AWQ (used instead of RTN)
  * `phoenix_in_rot`  an orthogonal matrix applied to the input before activation
                      quantization (QuaRot-style rotation; see ptq/rotate.py)

Select them in a precision spec: `w3a16:method=awq`, `w4a4:method=rot+gptq`,
`w8a8:method=sq`, `nf4` (= w4a16:wfmt=nf4, the bitsandbytes 4-bit format).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

# name -> (weight bits, activation bits)
PRECISION_LADDER: dict[str, tuple[int, int]] = {
    "fp16": (16, 16),
    "w8a16": (8, 16),
    "w8a8": (8, 8),
    "w4a16": (4, 16),
    "w4a8": (4, 8),
    "w4a4": (4, 4),
    "w3a8": (3, 8),
    "w2a4": (2, 4),
    "w2a16": (2, 16),
}


_OPT_KEYS = {"clip": ("a_clip", float), "gs": ("group_size", int),
             "group": ("group_size", int), "targets": ("targets", "tuple"),
             "skip": ("skip_names", "tuple"), "wsym": ("w_symmetric", "bool"),
             "asym": ("a_symmetric", "bool"), "agran": ("a_granularity", str),
             "abits_image": ("a_bits_image", int), "abits_text": ("a_bits_text", int),
             # published PTQ methods (phoenix/ptq)
             "method": ("method", str), "calib": ("calib", str), "ncal": ("n_calib", int),
             "wfmt": ("w_format", str), "alpha": ("sq_alpha", float),
             "actorder": ("gptq_actorder", "bool"), "damp": ("gptq_damp", float),
             "backend": ("backend", str), "seed": ("rot_seed", int),
             "wtok": ("w_tokens", str)}

# noise = control: Gaussian weight noise with the same per-group energy as RTN's error
METHODS = ("rtn", "awq", "gptq", "sq", "rot", "rot+gptq", "sq+gptq", "noise")
CALIB_MODES = ("mm", "text", "wiki")
W_FORMATS = ("int", "nf4")
# spec aliases: name -> (weight bits, act bits, implied options)
ALIASES = {"nf4": (4, 16, {"w_format": "nf4"}),
           "bnb-nf4": (4, 16, {"w_format": "nf4", "backend": "bnb"})}


def parse_precision(spec: str) -> tuple[int, int, dict]:
    """'w4a4:clip=0.999:skip=down_proj' -> (4, 4, {'a_clip': 0.999, 'skip_names': ('down_proj',)})"""
    import re
    head, *parts = str(spec).strip().split(":")
    head = head.lower()
    opts: dict = {}
    if head in ALIASES:
        w, a, implied = ALIASES[head]
        opts.update(implied)
    elif head in PRECISION_LADDER:
        w, a = PRECISION_LADDER[head]
    else:
        m = re.fullmatch(r"w(\d{1,2})a(\d{1,2})", head)
        if not m:
            raise KeyError(f"bad precision '{spec}': use one of {sorted(PRECISION_LADDER)} "
                           "or w<bits>a<bits>[:key=value...]")
        w, a = int(m.group(1)), int(m.group(2))
        if not (1 <= w <= 16 and 1 <= a <= 16):
            raise KeyError(f"bad precision '{spec}': bits must be in 1..16")
    for part in parts:
        if not part:
            continue
        if "=" not in part:
            raise KeyError(f"bad option '{part}' in '{spec}' (expected key=value)")
        k, v = part.split("=", 1)
        if k not in _OPT_KEYS:
            raise KeyError(f"unknown option '{k}' in '{spec}'; known: {sorted(_OPT_KEYS)}")
        field, kind = _OPT_KEYS[k]
        if kind == "tuple":
            opts[field] = tuple(x for x in v.split("+") if x)
        elif kind == "bool":
            opts[field] = v.lower() in ("1", "true", "yes", "on")
        else:
            opts[field] = kind(v)
    return w, a, opts


def precision_slug(spec: str) -> str:
    """Filesystem-safe name for a spec: 'w4a4:clip=0.999' -> 'w4a4_clip0.999'."""
    import re
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(spec).replace("=", "")).strip("_")


@dataclass
class QuantConfig:
    w_bits: int = 4
    a_bits: int = 16
    group_size: int = 128          # weight group size along the input dim; -1 = per-channel
    w_symmetric: bool = False      # asymmetric RTN is the usual choice for W4
    a_symmetric: bool = True       # per-token symmetric is the usual choice for activations
    a_granularity: str = "token"   # "token" | "tensor"
    a_clip: float = 1.0            # percentile clip for activations, 1.0 = plain MinMax
    targets: tuple[str, ...] = ("language",)   # subset of {"language","projector","vision"}
    # Token-selective activation bits (None = use a_bits). Image positions come from the
    # same mask LoRAS uses; decode-step tokens are generated text, so they count as text.
    #   w4a4:abits_text=8   -> image tokens at A4, text at A8   (is the image the problem?)
    #   w4a4:abits_image=8  -> image tokens at A8, text at A4   (is the language side?)
    a_bits_image: int | None = None
    a_bits_text: int | None = None
    skip_names: tuple[str, ...] = ("lm_head",)
    # --- published PTQ (phoenix/ptq); defaults reproduce plain RTN ---
    method: str = "rtn"            # rtn | awq | gptq | sq | rot | rot+gptq | sq+gptq
    calib: str = "mm"              # mm (image+caption) | text (same captions, no image) | wiki
    n_calib: int = 64              # calibration sequences
    w_format: str = "int"          # int | nf4 (bitsandbytes NormalFloat4, absmax blocks)
    sq_alpha: float = 0.85         # SmoothQuant migration strength
    gptq_actorder: bool = False    # GPTQ column reordering (static groups)
    gptq_damp: float = 0.01        # GPTQ Hessian dampening, fraction of mean diag
    backend: str = "sim"           # sim | bnb (real bitsandbytes kernels, nf4 only)
    rot_seed: int = 0              # seed of the random rotation (and of method=noise)
    # Weight-path attribution: quantized weights only at image positions ("image") or
    # only at text positions ("text"); the other positions use the FP16 weights.
    # Generated (decode) tokens are text. Keeps FP weights, doubles the matmuls.
    w_tokens: str | None = None

    def __post_init__(self):
        if self.method not in METHODS:
            raise KeyError(f"unknown method '{self.method}'; known: {METHODS}")
        if self.calib not in CALIB_MODES:
            raise KeyError(f"unknown calib '{self.calib}'; known: {CALIB_MODES}")
        if self.w_format not in W_FORMATS:
            raise KeyError(f"unknown wfmt '{self.w_format}'; known: {W_FORMATS}")
        if self.w_format == "nf4":
            if self.w_bits != 4:
                raise KeyError("wfmt=nf4 is a 4-bit format; use w4a*:wfmt=nf4")
            if "gptq" in self.method or "awq" in self.method:
                raise KeyError("nf4 is a data format quantized by absmax rounding; "
                               "it does not combine with awq/gptq here")
        if self.backend not in ("sim", "bnb"):
            raise KeyError("backend must be sim or bnb")
        if self.backend == "bnb" and (self.w_format != "nf4" or self.a_bits < 16
                                      or self.method != "rtn"):
            raise KeyError("backend=bnb only supports weight-only nf4 (spec 'bnb-nf4')")
        if "gptq" in self.method and self.w_bits >= 16:
            raise KeyError("gptq needs quantized weights (w_bits < 16)")
        if self.method == "noise" and (self.w_bits >= 16 or self.w_format != "int"):
            raise KeyError("method=noise matches int RTN error; use it with w<bits> < 16")
        if self.w_tokens not in (None, "image", "text"):
            raise KeyError("wtok must be image or text")
        if self.w_tokens and self.token_selective:
            raise KeyError("wtok and abits_image/abits_text are separate experiments")

    @property
    def needs_calibration(self) -> bool:
        return self.method != "rtn"

    @classmethod
    def from_ladder(cls, name: str, **kw) -> "QuantConfig":
        """Build from a precision spec; options in the spec override `kw`.

        Accepts the named rungs above and any `w<bits>a<bits>`, optionally followed
        by `:key=value` options (see `parse_precision`), e.g.

            w4a5                          intermediate activation width
            w4a4:clip=0.999               per-token 99.9th-percentile activation clip
            w4a4:skip=down_proj           leave down_proj in FP16 (mixed precision)
            w4a8:targets=language+projector+vision
            w3a16:gs=64                   weight-only, group size 64
        """
        w, a, opts = parse_precision(name)
        merged = dict(kw)
        merged.update(opts)
        if merged.get("w_format") == "nf4" and "group_size" not in opts:
            merged["group_size"] = 64              # bitsandbytes' default blocksize
        if "skip_names" in opts:
            merged["skip_names"] = tuple(kw.get("skip_names", ("lm_head",))) + opts["skip_names"]
        return cls(w_bits=w, a_bits=a, **merged)

    @property
    def is_identity(self) -> bool:
        bits = [self.a_bits] + [b for b in (self.a_bits_image, self.a_bits_text) if b is not None]
        return self.w_bits >= 16 and min(bits) >= 16 and self.method == "rtn"

    def bits_for(self, group: str) -> int:
        b = self.a_bits_image if group == "image" else self.a_bits_text
        return self.a_bits if b is None else b

    @property
    def token_selective(self) -> bool:
        return self.bits_for("image") != self.bits_for("text")


# --------------------------------------------------------------------------- #
# quantizer primitives
# --------------------------------------------------------------------------- #
def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x)


# bitsandbytes NF4 code book (QLoRA, Dettmers et al. 2023): quantiles of N(0,1)
# rescaled to [-1, 1], with an exact zero.
NF4_CODE = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0)


@torch.no_grad()
def quantize_weight_nf4(w: torch.Tensor, block: int = 64) -> torch.Tensor:
    """bitsandbytes-style NF4: blocks of `block` consecutive values of the row-major
    tensor, each scaled by its absmax and rounded to the nearest code."""
    flat = w.float().reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.view(-1, block)
    absmax = blk.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    code = torch.tensor(NF4_CODE, device=w.device)
    mid = (code[1:] + code[:-1]) / 2                     # decision thresholds
    idx = torch.bucketize(blk / absmax, mid)
    deq = code[idx] * absmax
    return deq.reshape(-1)[: w.numel()].view_as(w).to(w.dtype)


@torch.no_grad()
def quantize_weight(w: torch.Tensor, bits: int, group_size: int, symmetric: bool,
                    fmt: str = "int") -> torch.Tensor:
    """RTN quantize-dequantize a [out, in] weight matrix, grouped along `in`."""
    if fmt == "nf4":
        return quantize_weight_nf4(w, group_size if group_size and group_size > 0 else 64)
    if bits >= 16:
        return w.clone()
    out_f, in_f = w.shape
    gs = in_f if group_size in (-1, None) else min(group_size, in_f)
    if in_f % gs != 0:                      # fall back to per-channel if not divisible
        gs = in_f
    wg = w.reshape(out_f, in_f // gs, gs).float()

    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        q = _round_ste(wg / scale).clamp(-qmax - 1, qmax)
        deq = q * scale
    else:
        qmax = 2 ** bits - 1
        mx = wg.amax(dim=-1, keepdim=True)
        mn = wg.amin(dim=-1, keepdim=True)
        scale = ((mx - mn) / qmax).clamp(min=1e-8)
        zero = _round_ste(-mn / scale)
        q = (_round_ste(wg / scale) + zero).clamp(0, qmax)
        deq = (q - zero) * scale

    return deq.reshape(out_f, in_f).to(w.dtype)


@torch.no_grad()
def quantize_activation(x: torch.Tensor, bits: int, symmetric: bool = True,
                        granularity: str = "token", clip: float = 1.0) -> torch.Tensor:
    """Dynamic quantize-dequantize activations. x: [..., in]."""
    if bits >= 16:
        return x
    xf = x.float()

    if granularity == "token":
        if clip < 1.0:
            hi = torch.quantile(xf.abs().flatten(0, -2), clip, dim=-1, keepdim=True)
            hi = hi.reshape(*xf.shape[:-1], 1).clamp(min=1e-8)
        else:
            hi = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        lo = -hi if symmetric else xf.amin(dim=-1, keepdim=True)
    else:
        hi = xf.abs().amax().clamp(min=1e-8).reshape(1)
        lo = -hi if symmetric else xf.amin().reshape(1)

    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = (hi / qmax).clamp(min=1e-8)
        q = _round_ste(xf / scale).clamp(-qmax - 1, qmax)
        deq = q * scale
    else:
        qmax = 2 ** bits - 1
        scale = ((hi - lo) / qmax).clamp(min=1e-8)
        zero = _round_ste(-lo / scale)
        q = (_round_ste(xf / scale) + zero).clamp(0, qmax)
        deq = (q - zero) * scale
    return deq.to(x.dtype)


@torch.no_grad()
def matched_noise(w: torch.Tensor, cfg: "QuantConfig", name: str = "") -> torch.Tensor:
    """W + Gaussian noise whose per-(row, group) variance equals RTN's squared error
    there: the same error energy as quantization, in a random direction."""
    import zlib
    out_f, in_f = w.shape
    gs = cfg.group_size if cfg.group_size and cfg.group_size > 0 and in_f % cfg.group_size == 0 \
        else in_f
    err = (quantize_weight(w, cfg.w_bits, cfg.group_size, cfg.w_symmetric).float() - w.float())
    std = err.reshape(out_f, in_f // gs, gs).pow(2).mean(-1, keepdim=True).sqrt()
    g = torch.Generator(device=w.device).manual_seed(
        cfg.rot_seed * 1_000_003 + zlib.crc32(name.encode()))
    z = torch.randn(out_f, in_f // gs, gs, generator=g, device=w.device)
    return (w.float() + (z * std).reshape(out_f, in_f)).to(w.dtype)


# --------------------------------------------------------------------------- #
# the wrapper module
# --------------------------------------------------------------------------- #
class FakeQuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear that can switch between FP and quantized.

    Holds the original weight and a dequantized quantized copy, so switching is a
    pointer swap. Call `release_fp()` on eval-only runs to free the FP copy.
    """

    def __init__(self, base: nn.Linear, cfg: QuantConfig, name: str = ""):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.register_buffer("weight_fp", base.weight.data, persistent=False)
        pre = getattr(base, "phoenix_wq", None)           # set by GPTQ / AWQ
        if pre is not None:
            wq = pre.to(base.weight.dtype)
        elif cfg.method == "noise":
            wq = matched_noise(base.weight.data, cfg, name)
        else:
            wq = quantize_weight(base.weight.data, cfg.w_bits, cfg.group_size,
                                 cfg.w_symmetric, cfg.w_format)
        self.register_buffer("weight_q", wq, persistent=False)
        self.prequantized = pre is not None
        rot = getattr(base, "phoenix_in_rot", None)       # set by ptq/rotate.py
        if rot is not None:
            self.register_buffer("in_rot", rot, persistent=False)
        else:
            self.in_rot = None
        if base.bias is not None:
            self.register_buffer("bias", base.bias.data, persistent=False)
        else:
            self.bias = None
        self.quant_enabled: bool = True

    def release_fp(self) -> None:
        if self.cfg.w_tokens:                     # the other token type needs FP weights
            return
        if getattr(self, "weight_fp", None) is not None:
            self.weight_fp = None

    @property
    def weight(self) -> torch.Tensor:  # convenience for probes
        return self.weight_q if self.quant_enabled else self.weight_fp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_rot is not None:
            x = x @ self.in_rot                    # weights already hold W @ R
        if not self.quant_enabled:
            w = self.weight_fp
            if w is None:
                raise RuntimeError(
                    f"{self.name}: FP weights were released; reload the model to use fp mode"
                )
            return torch.nn.functional.linear(x, w, self.bias)
        c = self.cfg
        if c.token_selective:
            x = self._quantize_by_token_type(x)
        elif c.a_bits < 16:
            x = quantize_activation(x, c.a_bits, c.a_symmetric, c.a_granularity, c.a_clip)
        if c.w_tokens:
            return self._weights_by_token_type(x)
        return torch.nn.functional.linear(x, self.weight_q, self.bias)

    def _weights_by_token_type(self, x: torch.Tensor) -> torch.Tensor:
        from .loras import SteerContext
        F = torch.nn.functional
        mask = SteerContext.mask
        yq = F.linear(x, self.weight_q, self.bias)
        yf = F.linear(x, self.weight_fp, self.bias)
        if mask is None or x.dim() != 3 or x.shape[:2] != mask.shape:
            # decode steps / text-only inputs: every token is text
            return yq if self.cfg.w_tokens == "text" else yf
        sel = mask if self.cfg.w_tokens == "image" else ~mask
        return torch.where(sel[..., None], yq, yf)

    def _quantize_by_token_type(self, x: torch.Tensor) -> torch.Tensor:
        """Different activation bits at image and text positions (per-token scales, so
        the two groups never influence each other's rounding)."""
        from .loras import SteerContext
        c = self.cfg
        mask = SteerContext.mask
        bi, bt = c.bits_for("image"), c.bits_for("text")
        q = lambda t, b: t if b >= 16 else quantize_activation(
            t, b, c.a_symmetric, "token", c.a_clip)
        if mask is None or x.dim() != 3 or x.shape[:2] != mask.shape:
            return q(x, bt)                  # decode steps: generated text tokens
        out = x.clone()
        if bool(mask.any()):
            out[mask] = q(x[mask], bi)
        if bool((~mask).any()):
            out[~mask] = q(x[~mask], bt)
        return out

    def extra_repr(self) -> str:
        a = (f"a{self.cfg.a_bits}" if not self.cfg.token_selective else
             f"a{self.cfg.bits_for('image')}(image)/a{self.cfg.bits_for('text')}(text)")
        extra = "".join([", nf4" if self.cfg.w_format == "nf4" else "",
                         f", quantized weights at {self.cfg.w_tokens} tokens only"
                         if self.cfg.w_tokens else "",
                         ", prequantized" if self.prequantized else "",
                         ", rotated-input" if self.in_rot is not None else ""])
        return (f"in={self.in_features}, out={self.out_features}, "
                f"w{self.cfg.w_bits}{a}, g={self.cfg.group_size}{extra}")


# --------------------------------------------------------------------------- #
# model surgery
# --------------------------------------------------------------------------- #
def _submodule_roots(model: nn.Module, targets: Sequence[str]) -> list[tuple[str, nn.Module]]:
    """Map logical target names to actual submodules of a LLaVA-style model."""
    roots: list[tuple[str, nn.Module]] = []
    base = getattr(model, "model", model)

    def _first(obj, names):
        for n in names:
            m = getattr(obj, n, None)
            if m is not None:
                return n, m
        return None, None

    if "language" in targets:
        n, m = _first(base, ["language_model", "text_model"])
        if m is None:
            n, m = _first(model, ["language_model"])
        if m is not None:
            roots.append((n, m))
    if "projector" in targets:
        n, m = _first(base, ["multi_modal_projector", "mm_projector", "projector"])
        if m is not None:
            roots.append((n, m))
    if "vision" in targets:
        n, m = _first(base, ["vision_tower", "vision_model"])
        if m is not None:
            roots.append((n, m))
    if not roots:
        roots.append(("model", base))
    return roots


def quantize_model(model: nn.Module, cfg: QuantConfig, verbose: bool = True) -> list[FakeQuantLinear]:
    """Replace nn.Linear with FakeQuantLinear inside the configured target subtrees."""
    if cfg.is_identity:
        if verbose:
            print("[quant] fp16 requested -- no layers wrapped")
        return []

    wrapped: list[FakeQuantLinear] = []
    for root_name, root in _submodule_roots(model, cfg.targets):
        for mod_name, module in list(root.named_modules()):
            for child_name, child in list(module.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                full = f"{root_name}.{mod_name}.{child_name}".replace("..", ".")
                if any(s in full for s in cfg.skip_names):
                    continue
                q = FakeQuantLinear(child, cfg, name=full)
                setattr(module, child_name, q)
                wrapped.append(q)
    if verbose:
        n_params = sum(m.in_features * m.out_features for m in wrapped)
        tag = "" if cfg.method == "rtn" else f" via {cfg.method.upper()}"
        if cfg.method == "noise":
            tag = " as RTN-matched Gaussian NOISE (control)"
        if cfg.w_tokens:
            tag += f", quantized weights at {cfg.w_tokens} tokens only"
        if cfg.w_format == "nf4":
            tag += " (NF4)"
        print(f"[quant] wrapped {len(wrapped)} Linear layers "
              f"({n_params/1e9:.2f}B params){tag} at W{cfg.w_bits}"
              + (f"A{cfg.a_bits}" if not cfg.token_selective else
                 f"A{cfg.bits_for('image')} on image tokens / A{cfg.bits_for('text')} on text")
              + f", group_size={cfg.group_size}, targets={cfg.targets}")
    return wrapped


def set_quant(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantLinear):
            m.quant_enabled = enabled


@contextmanager
def quant_mode(model: nn.Module, enabled: bool):
    """Temporarily force quant on/off. Used to capture paired FP16/quant activations."""
    prev = [(m, m.quant_enabled) for m in model.modules() if isinstance(m, FakeQuantLinear)]
    try:
        set_quant(model, enabled)
        yield
    finally:
        for m, p in prev:
            m.quant_enabled = p


def release_fp_weights(model: nn.Module) -> None:
    """Free the FP16 shadow weights (eval-only runs). Halves weight VRAM."""
    for m in model.modules():
        if isinstance(m, FakeQuantLinear):
            m.release_fp()


def quant_error_report(model: nn.Module, topk: int = 10) -> list[dict]:
    """Per-layer relative weight quantization error -- a sanity check on the ladder."""
    rows = []
    for m in model.modules():
        if isinstance(m, FakeQuantLinear) and m.weight_fp is not None:
            num = (m.weight_q.float() - m.weight_fp.float()).pow(2).sum().item()
            den = m.weight_fp.float().pow(2).sum().item() + 1e-12
            rows.append({"name": m.name, "rel_w_mse": num / den})
    rows.sort(key=lambda r: -r["rel_w_mse"])
    return rows[:topk]


def iter_fq(model: nn.Module) -> Iterable[FakeQuantLinear]:
    for m in model.modules():
        if isinstance(m, FakeQuantLinear):
            yield m
