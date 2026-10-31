#!/usr/bin/env python
"""Layer-depth drift profiling: where does quantization actually damage the
visual representation, and how much of that damage is linearly recoverable.

Outputs drift.json with, per layer, the FP16-vs-quantized cosine similarity and
relative MSE of (a) the residual stream and (b) the K / V projections at visual
token positions, plus the LoRAS layer selection those numbers imply.

Run this BEFORE calibrating. If the drift peaks somewhere other than the top
quartile, the proposal's `l >= 0.75L` injection rule is the wrong place to spend
parameters and you want to say so in the writeup.
"""
from _common import base_parser, build_model, run_dir

import torch

from phoenix.data import batches, build_calibration_set
from phoenix.probes import drift_profile, select_layers
from phoenix.utils import save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-images", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--topk", type=int, default=8)
    args = p.parse_args()

    lm, qcfg = build_model(args, keep_fp=True)     # need both weight sets
    samples = build_calibration_set(args.coco_root, args.calib_subset,
                                    n_images=args.n_images, seed=args.seed)

    def it():
        return batches(samples, lm, batch_size=args.batch_size)

    with torch.no_grad():
        drift = drift_profile(lm, it(), lambda ids: ids == lm.image_token_id,
                              max_batches=10 ** 9)

    sel = {
        "max_drift_k": select_layers(drift, k=args.topk, site="k"),
        "max_drift_v": select_layers(drift, k=args.topk, site="v"),
        "upper_quartile": select_layers(drift, site="k", strategy="upper_quartile",
                                        n_layers=lm.n_layers),
        "all": list(range(lm.n_layers)),
    }

    d = run_dir(args, "drift")
    save_json({"precision": args.precision, "drift": drift, "selection": sel,
               "n_layers": lm.n_layers, "n_images": args.n_images}, d / "drift.json")

    print("\nlayer   hidden_cos  k_relmse  v_relmse")
    for i in range(lm.n_layers):
        h = drift["hidden"][i]["cos"] if i < len(drift["hidden"]) else float("nan")
        k = drift["k"][i]["rel_mse"]
        v = drift["v"][i]["rel_mse"]
        print(f"{i:5d}   {h:10.4f}  {k:8.4f}  {v:8.4f}")
    print("\nsuggested LoRAS layers (max drift):", sel["max_drift_k"])
    print("proposal's a priori rule           :", sel["upper_quartile"])
    print(f"\nwritten to {d/'drift.json'}")


if __name__ == "__main__":
    main()
