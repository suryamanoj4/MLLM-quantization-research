#!/usr/bin/env python
"""A-CAB hyperparameter sweep (alpha x tau) with the guard metrics attached.

The sweep reports CHAIR *and* object coverage, caption length and 4-gram repetition
in the same row, because pushing alpha up eventually buys a CHAIR improvement purely
by making captions shorter and more repetitive. A row that improves CHAIR_i while
coverage falls is not a win, and this table makes that visible instead of hiding it.
"""
from _common import (acab_cfg_from_args, add_acab_args, add_loras_args,
                     base_parser, build_model, maybe_attach_loras, run_dir)

from phoenix.acab import calibrate_tau
from phoenix.chair import ChairScorer
from phoenix.data import batches, build_caption_set, get_pope
from phoenix.evaluate import acab_session, run_captions, run_pope
from phoenix.utils import save_json


def main():
    p = base_parser(__doc__)
    add_loras_args(p)
    add_acab_args(p)
    p.add_argument("--alphas", default="0.0,0.5,1.0,2.0,4.0")
    p.add_argument("--tau-percentiles", default="40,60,80")
    p.add_argument("--n-chair-images", type=int, default=60)
    p.add_argument("--n-pope-images", type=int, default=50)
    p.add_argument("--pope-split", default="adversarial")
    p.add_argument("--chair-batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--with-pope", action="store_true")
    p.add_argument("--synonyms", default=None)
    args = p.parse_args()

    lm, qcfg = build_model(args, keep_fp=False)
    maybe_attach_loras(lm, args)

    cap = build_caption_set(args.coco_root, args.eval_subset,
                            n_images=args.n_chair_images, seed=args.seed)
    scorer = ChairScorer(args.coco_root, args.eval_subset, args.synonyms)
    pope = (get_pope(args.coco_root, args.eval_subset, args.pope_split,
                     n_images=args.n_pope_images, seed=args.seed)
            if args.with_pope else None)

    tau_calib = build_caption_set(args.coco_root, args.eval_subset, n_images=8,
                                  seed=args.seed + 999)
    taus = {}
    for q in [float(x) for x in args.tau_percentiles.split(",") if x.strip()]:
        taus[q] = calibrate_tau(lm, batches(tau_calib, lm, 2), q)
        print(f"[acab] tau(p{q:.0f}) = {taus[q]:.3f}")

    rows = []
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    for q, tau in taus.items():
        for a in alphas:
            args.alpha, args.tau = a, tau
            args.acab = "off" if a == 0.0 else args.acab if args.acab != "off" else "add"
            cfg = acab_cfg_from_args(args)
            with acab_session(lm, cfg) as ctrl:
                r = run_captions(lm, cap, ctrl, batch_size=args.chair_batch_size,
                                 max_new_tokens=args.max_new_tokens,
                                 record_trace=True, verbose=False)
                sc = scorer.score(r["records"])
                sc.pop("hallucinated", None)
                sc["coverage"] = scorer.coverage(r["records"])
                sc.update(r["fluency"])
                g = [x for tr in r["traces"] for row in tr["gated"] for x in row]
                sc["gated_frac"] = sum(g) / max(len(g), 1)
                if pope is not None:
                    pm = run_pope(lm, pope, ctrl, batch_size=8, verbose=False)
                    sc["pope_f1"] = pm["f1"]
                    sc["pope_yes"] = pm["yes_ratio"]
            row = {"tau_pct": q, "tau": tau, "alpha": a, **sc}
            rows.append(row)
            print(f"p{q:.0f} a={a:<4} CHAIR_s={sc['CHAIR_s']:.3f} CHAIR_i={sc['CHAIR_i']:.3f} "
                  f"cov={sc['coverage']:.3f} len={sc['avg_len']:.1f} rep4={sc['rep4']:.3f} "
                  f"gate={sc['gated_frac']:.2f}"
                  + (f" F1={sc.get('pope_f1', float('nan')):.3f}" if pope is not None else ""),
                  flush=True)

    d = run_dir(args, "acab_sweep")
    save_json({"config": vars(args), "rows": rows}, d / "sweep.json")
    print(f"\nwritten to {d/'sweep.json'}")


if __name__ == "__main__":
    main()
