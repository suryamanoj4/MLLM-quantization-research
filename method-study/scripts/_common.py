"""Shared CLI plumbing for the Phoenix scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from phoenix.acab import ACABConfig  # noqa: E402
from phoenix.data import check_root  # noqa: E402
from phoenix.loras import attach_loras, load_loras  # noqa: E402
from phoenix.model import DEFAULT_MODEL, load_model  # noqa: E402
from phoenix.quant import QuantConfig, quantize_model, release_fp_weights  # noqa: E402
from phoenix.utils import human_env, resolve_dtype, set_seed  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--coco-root", required=True, help="COCO root (see phoenix/data.py)")
    p.add_argument("--eval-subset", default="val2014")
    p.add_argument("--calib-subset", default="train2014")
    p.add_argument("--precision", default="w4a8",
                   help="a precision spec: fp16 | w4a8 | nf4 | w3a16:method=awq | "
                        "w4a4:method=rot+gptq | ... (see scripts/00_precision_ladder.py)")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--quant-targets", default="language",
                   help="comma list of language,projector,vision")
    p.add_argument("--act-clip", type=float, default=1.0)
    p.add_argument("--act-granularity", default="token", choices=["token", "tensor"])
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--tag", default="")
    return p


def add_loras_args(p):
    p.add_argument("--loras", default="", help="path to a calibrated loras .pt")
    p.add_argument("--loras-scale", type=float, default=1.0)
    return p


def add_acab_args(p):
    p.add_argument("--acab", default="off", choices=["off", "add", "mult"])
    p.add_argument("--gate", default="prev", choices=["prev", "two_pass", "none"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.5)
    p.add_argument("--delta-max", type=float, default=3.0)
    p.add_argument("--acab-layers", default="", help="comma list; empty = all")
    p.add_argument("--acab-skip-first", action="store_true",
                   help="old behaviour: leave the first answer token (POPE's answer) alone")
    p.add_argument("--acab-head-frac", type=float, default=0.0,
                   help=">0 restricts A-CAB to the top fraction of visual heads")
    return p


def build_model(args, keep_fp: bool = False):
    set_seed(args.seed)
    check_root(args.coco_root)
    resolve_calib_subset(args)
    dtype = resolve_dtype(args.dtype)
    print("[env]", human_env())
    qcfg = QuantConfig.from_ladder(
        args.precision,
        group_size=args.group_size,
        targets=tuple(t.strip() for t in args.quant_targets.split(",") if t.strip()),
        a_clip=args.act_clip,
        a_granularity=args.act_granularity,
    )
    if qcfg.backend == "bnb":
        raise SystemExit("bnb-nf4 has no FP16 shadow weights, so LoRAS / drift probes "
                         "cannot use it; use the simulated 'nf4' here")
    lm = load_model(args.model_id, dtype=dtype, device=args.device,
                    attn_implementation=args.attn)
    from phoenix.ptq import apply_ptq
    apply_ptq(lm, qcfg, coco_root=args.coco_root, seed=args.seed)
    quantize_model(lm.model, qcfg)
    if not keep_fp:
        release_fp_weights(lm.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return lm, qcfg


def resolve_calib_subset(args) -> None:
    """A mini-COCO has no train2014; splits.json already guarantees disjointness."""
    from phoenix.data import split_pool
    if getattr(args, "calib_subset", None) and split_pool(args.coco_root, "calib"):
        if args.calib_subset != args.eval_subset:
            args.calib_subset = args.eval_subset


def maybe_attach_loras(lm, args):
    if not getattr(args, "loras", ""):
        return None
    correctors, meta = load_loras(args.loras, lm.device, lm.dtype)
    for c in correctors.values():
        c.scale = args.loras_scale
    attach_loras(lm.layers, correctors)
    print(f"[loras] attached {len(correctors)} correctors "
          f"(layers {sorted({k[0] for k in correctors})}, scale={args.loras_scale})")
    return meta


def acab_cfg_from_args(args, heads=None) -> ACABConfig:
    layers = tuple(int(x) for x in args.acab_layers.split(",") if x.strip())
    return ACABConfig(mode=args.acab, gate=args.gate, alpha=args.alpha,
                      tau=args.tau, delta_max=args.delta_max, layers=layers,
                      heads=heads or {},
                      first_token=not getattr(args, "acab_skip_first", False))


def run_dir(args, name: str) -> Path:
    from phoenix.quant import precision_slug
    tag = args.tag or precision_slug(args.precision)
    d = Path(args.out) / f"{name}__{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d
