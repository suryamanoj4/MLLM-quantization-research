#!/usr/bin/env python
"""Calibrate LoRAS (Contribution 2).

Closed-form reduced-rank ridge regression on cached FP16/quantized K,V activations
at visual token positions, fitted sequentially block by block. Writes a .pt with the
correctors plus a diagnostics json containing, per layer, the rank-vs-recovery curve
and the full-rank *ceiling* -- i.e. how much of the quantization error is linearly
recoverable at all.

    python scripts/02_calibrate_loras.py --coco-root /data/coco --precision w4a8 \
        --layers 8,10,12,14,16,18,20,22 --rank 16
"""
from _common import base_parser, build_model, run_dir

from phoenix.calibrate import calibrate_loras, save
from phoenix.data import batches, build_calibration_set
from phoenix.loras import loras_param_count, summarise
from phoenix.utils import load_json, save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-calib", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--energy", type=float, default=None,
                   help="if set, choose rank per layer to reach this share of the "
                        "achievable recovery (e.g. 0.9) instead of a fixed --rank")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--layers", default="",
                   help="comma list; empty = read drift.json, else upper quartile")
    p.add_argument("--drift-json", default="")
    p.add_argument("--layers-per-pass", type=int, default=4,
                   help="1 = strictly sequential (slowest, most faithful)")
    p.add_argument("--sites", default="k,v")
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--stats-device", default=None)
    args = p.parse_args()

    lm, qcfg = build_model(args, keep_fp=True)

    if args.layers:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
    elif args.drift_json:
        layer_ids = load_json(args.drift_json)["selection"]["max_drift_k"]
        print(f"[loras] layers from drift profile: {layer_ids}")
    else:
        layer_ids = list(range(int(0.75 * lm.n_layers), lm.n_layers))
        print(f"[loras] no drift profile given, falling back to upper quartile: {layer_ids}")

    samples = build_calibration_set(args.coco_root, args.calib_subset,
                                    n_images=args.n_calib, seed=args.seed + 7)
    print(f"[loras] {len(samples)} calibration images from {args.calib_subset}")

    def it():
        return batches(samples, lm, batch_size=args.batch_size)

    correctors, diag = calibrate_loras(
        lm, it, lambda ids: ids == lm.image_token_id,
        layer_ids=layer_ids, rank=args.rank, ridge=args.ridge,
        sites=tuple(s.strip() for s in args.sites.split(",") if s.strip()),
        layers_per_pass=args.layers_per_pass, use_bias=not args.no_bias,
        energy=args.energy, val_frac=args.val_frac, stats_device=args.stats_device,
    )

    d = run_dir(args, "loras")
    meta = {"precision": args.precision, "layers": layer_ids, "rank": args.rank,
            "ridge": args.ridge, "energy": args.energy, "n_calib": args.n_calib,
            "sites": args.sites, "layers_per_pass": args.layers_per_pass,
            "model_id": args.model_id, "group_size": args.group_size}
    save(str(d / "loras.pt"), correctors, diag, meta)
    save_json({"meta": meta,
               "diagnostics": {f"{k[0]}|{k[1]}": v for k, v in diag.items()}},
              d / "loras_diagnostics.json")

    print("\n" + summarise({k: v for k, v in diag.items()}))
    n = loras_param_count(correctors)
    tot = sum(p.numel() for p in lm.model.parameters())
    print(f"\nLoRAS parameters: {n/1e6:.2f}M ({100*n/tot:.3f}% of the model)")
    print(f"written to {d/'loras.pt'}")

    worst = min(v["ceiling"] for v in diag.values())
    best = max(v["ceiling"] for v in diag.values())
    print(f"\nlinear recoverability ceiling across fitted sites: "
          f"{worst:.3f} .. {best:.3f}")
    if best < 0.25:
        print("NOTE: even a full-rank linear map removes <25% of the quantization "
              "error here. That is a result worth reporting, and it says LoRAS "
              "should be argued as a cheap partial corrector, not a reconstruction.")


if __name__ == "__main__":
    main()
