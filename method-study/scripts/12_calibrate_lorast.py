#!/usr/bin/env python
"""Calibrate LoRAS-T: text-side residual-stream correctors for a quantized rung.

Gate 2 put the grounding loss at text positions, so this fits one rank-r closed-form
corrector per decoder layer on the residual stream at text positions (prompt +
caption), sequentially, against FP16 hidden states on FP16's own captions for the
calibration images (disjoint from every evaluation image).

    python scripts/12_calibrate_lorast.py --coco-root data/coco-mini \
        --precision w3a16 --out runs/lorast/w3_rtn.pt
    python scripts/12_calibrate_lorast.py --coco-root data/coco-mini \
        --precision w3a16:method=gptq:calib=wiki --out runs/lorast/w3_gptq.pt

Then evaluate as extra ladder rungs next to the Gate-2 rows (same pool, --resume):

    python scripts/00_precision_ladder.py --coco-root data/coco-mini --out runs/gate2 \
        --preset gate2 --ladder-extra "w3a16+lorast=runs/lorast/w3_rtn.pt" ... --resume

The script ends with a held-out check that costs seconds: the KL between FP16 and
quantized next-token distributions on the validation captions, with and without the
correctors. If that does not drop, do not spend GPU time on the ladder.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from phoenix import loras_text as LT
from phoenix.loras import steer_enabled, steer_mask
from phoenix.model import DEFAULT_MODEL
from phoenix.ptq import load_quantized
from phoenix.quant import quant_mode
from phoenix.utils import human_env, resolve_dtype, save_json, set_seed


@torch.no_grad()
def heldout_kl(lm, seqs, idx, steer: bool) -> dict:
    """Mean KL(FP16 || quant) and top-1 agreement over caption-answer positions."""
    kl_sum, agree, n = 0.0, 0, 0
    for j in idx:
        s = seqs[j]
        img = s["input_ids"] == lm.image_token_id
        with steer_mask(img), steer_enabled(False), quant_mode(lm.model, False):
            lp = lm.model(**s, use_cache=False).logits[0].float().log_softmax(-1)
        with steer_mask(img), steer_enabled(steer), quant_mode(lm.model, True):
            lq = lm.model(**s, use_cache=False).logits[0].float().log_softmax(-1)
        a0 = s["_answer_start"]
        lp, lq = lp[a0 - 1:-1], lq[a0 - 1:-1]          # predictions of the answer tokens
        kl_sum += float(F.kl_div(lq, lp, log_target=True, reduction="sum"))
        agree += int((lp.argmax(-1) == lq.argmax(-1)).sum())
        n += lp.shape[0]
    return {"kl": kl_sum / max(n, 1), "top1_agree": agree / max(n, 1), "tokens": n}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--coco-root", required=True)
    p.add_argument("--precision", default="w3a16")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-calib", type=int, default=192)
    p.add_argument("--source", default="gen", choices=["gen", "ref"],
                   help="gen = FP16's own captions (default); ref = COCO references")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--qa-per-image", type=int, default=0,
                   help="also calibrate on this many FP16-answered yes/no object questions "
                        "per calibration image (half present, half absent); 2 is a good value")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--energy", type=float, default=None)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--layers", default="", help="comma list; empty = every layer")
    p.add_argument("--layers-per-pass", type=int, default=1)
    p.add_argument("--val-every", type=int, default=5, help="every k-th sequence is held out")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    set_seed(a.seed)
    print("[env]", human_env(), flush=True)
    t0 = time.perf_counter()
    lm, cfg, info = load_quantized(a.model_id, a.precision, a.coco_root,
                                   dtype=resolve_dtype(a.dtype), device=a.device,
                                   keep_fp=True, seed=a.seed, group_size=a.group_size)
    print(f"[loras-t] {a.precision} ready ({time.perf_counter() - t0:.0f}s): {info}", flush=True)

    seqs, caps = LT.caption_sequences(lm, a.coco_root, a.n_calib, seed=a.seed, source=a.source,
                                      max_new_tokens=a.max_new_tokens)
    # where the caption starts in each sequence (for the held-out KL; +-1 token)
    tok = lm.processor.tokenizer
    for s, cap in zip(seqs, caps):
        n_cap = len(tok(" " + cap.strip(), add_special_tokens=False)["input_ids"])
        s["_answer_start"] = s["input_ids"].shape[1] - n_cap
    model_in = [{k: v for k, v in s.items() if not k.startswith("_")} for s in seqs]

    qa_cal, qa_probe = [], []
    if a.qa_per_image:
        qa_cal, qa_probe, qa_rec = LT.qa_sequences(lm, a.coco_root, a.n_calib,
                                                   per_image=a.qa_per_image, seed=a.seed)
    n_layers = len(lm.layers)
    layer_ids = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(n_layers))
    val_idx = [j for j in range(len(seqs)) if a.val_every and j % a.val_every == 0]

    before = heldout_kl(lm, seqs, val_idx, steer=False)
    print(f"[loras-t] held-out KL(fp16||{a.precision}) = {before['kl']:.4f}  "
          f"top-1 agreement {before['top1_agree']:.3f}  ({before['tokens']} tokens)", flush=True)
    # the QA batches follow the captions in the calibration list; hold out the same ones
    qa_val = [j for j in range(len(qa_cal))
              if a.val_every and (len(model_in) + j) % a.val_every == 0]
    if qa_cal:
        before["qa"] = LT.heldout_pyes(lm, qa_probe, qa_val, steer=False)
        print(f"[loras-t] held-out yes/no: |dP(yes)| = {before['qa']['abs']:.4f}  "
              f"mean shift {before['qa']['shift']:+.4f}  ({before['qa']['n']} questions)", flush=True)
    # interleave so every k-th calibration item (caption or QA batch) is held out
    cal_items = model_in + qa_cal

    correctors, diag = LT.calibrate(lm, cal_items, layer_ids, rank=a.rank, ridge=a.ridge,
                                    layers_per_pass=a.layers_per_pass, energy=a.energy,
                                    val_every=a.val_every)
    after = heldout_kl(lm, seqs, val_idx, steer=True)
    print(f"[loras-t] held-out KL with LoRAS-T = {after['kl']:.4f}  "
          f"top-1 agreement {after['top1_agree']:.3f}", flush=True)
    if qa_cal:
        after["qa"] = LT.heldout_pyes(lm, qa_probe, qa_val, steer=True)
        print(f"[loras-t] held-out yes/no with LoRAS-T: |dP(yes)| = {after['qa']['abs']:.4f}  "
              f"mean shift {after['qa']['shift']:+.4f}", flush=True)
    for sc in (0.5, 1.5):
        LT.set_scale(lm.layers, sc)
        r = heldout_kl(lm, seqs, val_idx, steer=True)
        if qa_cal:
            r["qa"] = LT.heldout_pyes(lm, qa_probe, qa_val, steer=True)
        print(f"[loras-t]   scale {sc}: KL {r['kl']:.4f}  top-1 {r['top1_agree']:.3f}"
              + (f"  |dP(yes)| {r['qa']['abs']:.4f} shift {r['qa']['shift']:+.4f}" if qa_cal else ""),
              flush=True)
        after[f"scale_{sc}"] = r
    LT.set_scale(lm.layers, 1.0)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_params = sum(c.A.numel() + c.B.numel() + c.bias.numel() for c in correctors.values())
    meta = {"precision": a.precision, "rank": a.rank, "ridge": a.ridge, "source": a.source,
            "n_calib": len(seqs), "qa_per_image": a.qa_per_image, "n_qa_batches": len(qa_cal),
            "layers": layer_ids, "params": n_params,
            "kl_before": before, "kl_after": after}
    LT.save(out, correctors, meta)
    save_json({"meta": meta, "layers": {str(k): v for k, v in diag.items()},
               "calib_captions": caps[:20]}, out.with_suffix(".json"))
    print(f"[loras-t] {len(correctors)} correctors, {n_params / 1e6:.1f}M parameters "
          f"({n_params * 2 / 2**20:.0f} MiB fp16) -> {out}  "
          f"total {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
