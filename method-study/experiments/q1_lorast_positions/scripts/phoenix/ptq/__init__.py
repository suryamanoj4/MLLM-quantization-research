"""Published post-training quantization methods, simulated on the same fake-quant
substrate as the rest of Phoenix (see phoenix/quant.py for why simulation).

    method   what it does                                   needs calibration
    rtn      round-to-nearest (the baseline)                no
    awq      activation-aware scaling + clipping search     yes
    gptq     Hessian-based error-compensating rounding      yes
    sq       SmoothQuant activation->weight migration       yes (activation maxima)
    rot      QuaRot-style random Hadamard rotation          no
    rot+gptq rotation, then GPTQ on the rotated weights     yes   (QuaRot's W4A4 recipe)
    sq+gptq  SmoothQuant, then GPTQ                         yes

`nf4` is a weight format, not a method (bitsandbytes' NormalFloat4, the Hugging Face
`load_in_4bit` default); `bnb-nf4` loads the real bitsandbytes kernels instead of
simulating them, to check the simulation.

Only the language model's decoder layers get the method. Projector / vision targets,
when requested, fall back to RTN.
"""
from __future__ import annotations

import gc
import time

import torch

from ..quant import QuantConfig, quantize_model, release_fp_weights


def needs_data(cfg: QuantConfig) -> bool:
    return cfg.method in ("awq", "gptq", "sq", "rot+gptq", "sq+gptq")


@torch.no_grad()
def apply_ptq(lm, cfg: QuantConfig, coco_root=None, seqs=None, seed: int = 0) -> dict:
    """Run the method's calibration and leave the hooks quantize_model() picks up."""
    if cfg.method in ("rtn", "noise"):          # noise is drawn in FakeQuantLinear
        return {}
    from . import calib
    from .engine import capture_layer0
    t0 = time.perf_counter()
    info: dict = {"method": cfg.method, "calib": cfg.calib if needs_data(cfg) else None}
    if needs_data(cfg) and seqs is None:
        if coco_root is None and cfg.calib in ("mm", "text"):
            raise ValueError(f"{cfg.method} with calib={cfg.calib} needs coco_root")
        seqs = calib.build(lm, coco_root, cfg.calib, cfg.n_calib, seed)
    if any(t != "language" for t in cfg.targets):
        print(f"[ptq] note: {cfg.method} is applied to the language model only; "
              f"{[t for t in cfg.targets if t != 'language']} use RTN")

    if cfg.method.startswith("sq"):
        from .transforms import smoothquant
        info["sq"] = smoothquant(lm, seqs, cfg.sq_alpha)
        print(f"[ptq] SmoothQuant alpha={cfg.sq_alpha}: largest migration scale "
              f"{info['sq']['max_scale']:.1f}", flush=True)
    if cfg.method.startswith("rot"):
        from .transforms import rotate
        info["rot"] = rotate(lm, cfg.rot_seed)
        print(f"[ptq] rotated every decoder Linear input: {info['rot']['rotations']}", flush=True)
    if cfg.method == "awq":
        from .awq import awq
        info["awq"] = awq(lm, capture_layer0(lm, seqs), cfg, seed=seed)
    if cfg.method.endswith("gptq"):
        from .gptq import gptq
        info["gptq"] = gptq(lm, capture_layer0(lm, seqs), cfg)
    for k in ("awq", "gptq"):
        if k in info:
            info[k] = {"mean_loss_ratio_vs_rtn": info[k]["mean_loss_ratio_vs_rtn"]}
    info["minutes"] = (time.perf_counter() - t0) / 60
    print(f"[ptq] {cfg.method} done in {info['minutes']:.1f} min", flush=True)
    del seqs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return info


def load_quantized(model_id: str, spec: str, coco_root=None, *, dtype=torch.float16,
                   device: str = "cuda", attn_implementation: str = "sdpa",
                   keep_fp: bool = False, seed: int = 0, verbose: bool = True, **cfg_kw):
    """Load LLaVA, apply the spec's PTQ method and fake-quantize. Returns (lm, cfg, info).

    The one entry point every script uses, so a rung means the same thing everywhere.
    """
    from ..model import load_model
    cfg = QuantConfig.from_ladder(spec, **cfg_kw)
    if cfg.backend == "bnb":
        lm = load_model(model_id, dtype=dtype, device=device,
                        attn_implementation=attn_implementation,
                        bnb_4bit=tuple(cfg.targets))
        return lm, cfg, {"method": "bnb-nf4 (real kernels)"}
    lm = load_model(model_id, dtype=dtype, device=device,
                    attn_implementation=attn_implementation)
    info = apply_ptq(lm, cfg, coco_root=coco_root, seed=seed)
    quantize_model(lm.model, cfg, verbose=verbose)
    if not keep_fp:
        release_fp_weights(lm.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return lm, cfg, info
