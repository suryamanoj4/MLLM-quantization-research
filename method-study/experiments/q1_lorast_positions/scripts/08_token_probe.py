#!/usr/bin/env python
"""Is the A4 collapse caused by the image tokens? Two cheap probes.

1. Outlier census (fp16, no quantization). For every token entering the decoder's
   linear layers, measure what per-token rounding would do to it:
     ratio     max|x| / median|x|  -- how dominated the token is by its largest entry
     erased@4  share of entries that symmetric 4-bit per-token rounding sets to 0
     erased@8  the same at 8 bits
   reported separately for IMAGE and TEXT positions, per layer and per input kind
   (attention input = q/k/v input, attention output, MLP input, down_proj input).
   If image tokens are erased far more at 4 bits, that is the collapse mechanism.

2. Blind fluency (no image at all). Text-only questions and teacher-forced perplexity
   of COCO reference captions, at each precision. If the language model is fluent
   without an image at W4A4, the collapse needs the image to happen.

    python scripts/08_token_probe.py --coco-root data/coco-mini --precisions fp16,w4a4
    # then the causal arms in the ladder:
    python scripts/00_precision_ladder.py --coco-root data/coco-mini --out runs/ladder_tokens \\
        --ladder "fp16,w4a8,w4a4,w4a4:abits_text=8,w4a4:abits_image=8"
"""
import argparse
import gc
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from phoenix.data import batches, build_calibration_set, check_root
from phoenix.model import DEFAULT_MODEL, load_model
from phoenix.probes import blind_captions, blind_fluency
from phoenix.ptq import load_quantized
from phoenix.utils import human_env, resolve_dtype, save_json, set_seed

KINDS = {"self_attn.q_proj": "attn_in", "self_attn.o_proj": "attn_out",
         "mlp.gate_proj": "mlp_in", "mlp.down_proj": "down_in"}

# --------------------------------------------------------------------------- #
# 1. outlier census
# --------------------------------------------------------------------------- #
@torch.no_grad()
def census(lm, samples, batch_size: int) -> dict:
    acc = defaultdict(lambda: {"n": 0, "ratio": 0.0, "e4": 0.0, "e8": 0.0, "maxabs": 0.0})
    state = {"mask": None}
    hooks = []
    for li, layer in enumerate(lm.layers):
        for name, kind in KINDS.items():
            mod = layer.get_submodule(name)

            def pre(m, args, li=li, kind=kind):
                x = args[0]
                mask = state["mask"]
                if x.dim() != 3 or mask is None or x.shape[:2] != mask.shape:
                    return
                a = x.detach().abs().float()
                mx = a.amax(-1)
                med = a.median(-1).values
                e4 = (a < (mx / 7 / 2).unsqueeze(-1)).float().mean(-1)      # sym 4-bit
                e8 = (a < (mx / 127 / 2).unsqueeze(-1)).float().mean(-1)    # sym 8-bit
                ratio = mx / med.clamp(min=1e-6)
                for group, sel in (("image", mask), ("text", ~mask)):
                    if not bool(sel.any()):
                        continue
                    e = acc[(li, kind, group)]
                    e["n"] += int(sel.sum())
                    e["ratio"] += float(ratio[sel].sum())
                    e["e4"] += float(e4[sel].sum())
                    e["e8"] += float(e8[sel].sum())
                    e["maxabs"] = max(e["maxabs"], float(mx[sel].max()))
            hooks.append(mod.register_forward_pre_hook(pre))
    try:
        for b in batches(samples, lm, batch_size=batch_size):
            state["mask"] = b["input_ids"] == lm.image_token_id
            lm.model(**b, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    rows = []
    for (li, kind, group), e in sorted(acc.items()):
        n = max(e["n"], 1)
        rows.append({"layer": li, "kind": kind, "group": group, "tokens": e["n"],
                     "ratio": e["ratio"] / n, "erased4": e["e4"] / n, "erased8": e["e8"] / n,
                     "maxabs": e["maxabs"]})
    return {"rows": rows}


def print_census(res: dict):
    rows = res["rows"]
    print("\nshare of each token's entries that per-token rounding sets to zero "
          "(mean over tokens and layers)")
    print(f"{'input':<12}{'group':<8}{'erased@4':>10}{'erased@8':>10}{'max/median':>12}{'max|x|':>10}")
    for kind in KINDS.values():
        for group in ("image", "text"):
            rs = [r for r in rows if r["kind"] == kind and r["group"] == group]
            if not rs:
                continue
            w = sum(r["tokens"] for r in rs)
            e4 = sum(r["erased4"] * r["tokens"] for r in rs) / w
            e8 = sum(r["erased8"] * r["tokens"] for r in rs) / w
            ra = sum(r["ratio"] * r["tokens"] for r in rs) / w
            mx = max(r["maxabs"] for r in rs)
            print(f"{kind:<12}{group:<8}{e4:>10.3f}{e8:>10.3f}{ra:>12.1f}{mx:>10.1f}")
    img = [r for r in rows if r["group"] == "image"]
    txt = {(r["layer"], r["kind"]): r for r in rows if r["group"] == "text"}
    gaps = sorted(((r["erased4"] - txt[(r["layer"], r["kind"])]["erased4"], r)
                   for r in img if (r["layer"], r["kind"]) in txt), key=lambda t: -t[0])[:8]
    print("\nlargest image-minus-text gaps in erased@4 (where image tokens suffer most)")
    for g, r in gaps:
        print(f"  layer {r['layer']:2d} {r['kind']:<9} image {r['erased4']:.3f}  "
              f"text {txt[(r['layer'], r['kind'])]['erased4']:.3f}  gap {g:+.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--coco-root", required=True)
    p.add_argument("--precisions", default="fp16,w4a8,w4a4",
                   help="specs for the blind-fluency probe, e.g. 'fp16,w3a16:method=awq' "
                        "(the census always runs in fp16)")
    p.add_argument("--n-images", type=int, default=16)
    p.add_argument("--n-captions", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--skip-census", action="store_true")
    p.add_argument("--skip-blind", action="store_true")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/token_probe")
    a = p.parse_args()

    set_seed(a.seed)
    check_root(a.coco_root)
    print("[env]", human_env(), flush=True)
    out = {"config": vars(a)}

    if not a.skip_census:
        lm = load_model(a.model_id, dtype=resolve_dtype(a.dtype), device=a.device)
        samples = build_calibration_set(a.coco_root, "train2014", n_images=a.n_images,
                                        seed=a.seed + 3, fallback_subset="val2014")
        print(f"[census] {len(samples)} calibration images, fp16", flush=True)
        out["census"] = census(lm, samples, a.batch_size)
        print_census(out["census"])
        del lm
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not a.skip_blind:
        caps = blind_captions(a.coco_root, a.n_captions)
        out["blind"] = {}
        for prec in [x.strip() for x in a.precisions.split(",") if x.strip()]:
            lm, _, _ = load_quantized(a.model_id, prec, a.coco_root,
                                      dtype=resolve_dtype(a.dtype), device=a.device)
            r = blind_fluency(lm, caps)
            out["blind"][prec] = r
            f = r["fluency"]
            print(f"\n[blind {prec}] text-only caption PPL {r['blind_ppl']:.2f}  "
                  f"len {f['avg_len']:.1f}  rep4 {f['rep4']:.3f}")
            print(f"  e.g. {r['answers'][0][:160]!r}", flush=True)
            del lm
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    Path(a.out).mkdir(parents=True, exist_ok=True)
    save_json(out, Path(a.out) / "token_probe.json")
    print(f"\nwritten to {Path(a.out) / 'token_probe.json'}")
    if "blind" in out and "fp16" in out["blind"]:
        base = out["blind"]["fp16"]["blind_ppl"]
        print("\nreading: if a precision is fluent here (PPL near fp16's "
              f"{base:.1f}, normal length, low rep4) but broken on images in the ladder, "
              "the collapse needs the image to happen.")


if __name__ == "__main__":
    main()
