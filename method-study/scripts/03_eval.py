#!/usr/bin/env python
"""Evaluate one configuration on POPE and/or CHAIR.

    # quantized baseline
    python scripts/03_eval.py --coco-root /data/coco --precision w4a8

    # + LoRAS
    python scripts/03_eval.py --coco-root /data/coco --precision w4a8 \
        --loras runs/loras__w4a8/loras.pt

    # + LoRAS + A-CAB
    python scripts/03_eval.py --coco-root /data/coco --precision w4a8 \
        --loras runs/loras__w4a8/loras.pt --acab add --alpha 1.0 --tau-percentile 60
"""
from _common import (acab_cfg_from_args, add_acab_args, add_loras_args,
                     base_parser, build_model, maybe_attach_loras, run_dir)

import torch

from phoenix.acab import calibrate_tau
from phoenix.chair import ChairScorer
from phoenix.data import batches, build_caption_set, get_pope, load_captions
from phoenix.evaluate import acab_session, run_captions, run_pope
from phoenix.metrics import caption_perplexity
from phoenix.probes import rank_visual_heads, visual_attention_stats
from phoenix.utils import save_json


def main():
    p = base_parser(__doc__)
    add_loras_args(p)
    add_acab_args(p)
    p.add_argument("--tasks", default="pope,chair")
    p.add_argument("--pope-splits", default="random,popular,adversarial")
    p.add_argument("--pope-file", default="", help="official POPE jsonl (preferred)")
    p.add_argument("--n-pope-images", type=int, default=100)
    p.add_argument("--n-chair-images", type=int, default=100)
    p.add_argument("--pope-batch-size", type=int, default=8)
    p.add_argument("--chair-batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--tau-percentile", type=float, default=None,
                   help="set tau from this percentile of the model's own decode "
                        "entropy instead of --tau")
    p.add_argument("--synonyms", default=None)
    p.add_argument("--ppl", action="store_true", help="also report reference-caption PPL")
    args = p.parse_args()

    lm, qcfg = build_model(args, keep_fp=False)
    loras_meta = maybe_attach_loras(lm, args)

    results = {"config": vars(args), "quant": qcfg.__dict__, "loras_meta": loras_meta}
    tasks = {t.strip() for t in args.tasks.split(",")}

    # ---- A-CAB threshold + head selection ---------------------------------- #
    heads = None
    if args.acab != "off":
        calib = build_caption_set(args.coco_root, args.eval_subset,
                                  n_images=8, seed=args.seed + 999)
        if args.tau_percentile is not None:
            args.tau = calibrate_tau(lm, batches(calib, lm, 2), args.tau_percentile)
            print(f"[acab] tau <- p{args.tau_percentile:.0f} of decode entropy = {args.tau:.3f}")
        if args.acab_head_frac > 0:
            from phoenix.data import open_image
            from phoenix.model import prepare_batch
            b = prepare_batch(lm, [open_image(calib[0].image_path)], [calib[0].question])
            from phoenix.acab import _restore_attn_impl, _set_attn_impl
            prev = _set_attn_impl(lm.model, "eager")   # every config, not just the top one
            with torch.no_grad():
                out = lm.model(**b, use_cache=False, output_attentions=True)
            _restore_attn_impl(prev)
            vis = (b["input_ids"] == lm.image_token_id)[0]
            st = visual_attention_stats(out.attentions, vis)
            heads = rank_visual_heads(st["mass"], args.acab_head_frac)
            print(f"[acab] restricted to {sum(len(v) for v in heads.values())} visual heads "
                  f"across {len(heads)} layers")
            results["acab_heads"] = {str(k): v for k, v in heads.items()}
    cfg = acab_cfg_from_args(args, heads)
    results["acab_cfg"] = cfg.__dict__ | {"heads": "<selected>" if heads else {}}

    # ---- POPE --------------------------------------------------------------- #
    if "pope" in tasks:
        results["pope"] = {}
        for split in [s.strip() for s in args.pope_splits.split(",") if s.strip()]:
            samples = get_pope(args.coco_root, args.eval_subset, split,
                               n_images=args.n_pope_images, seed=args.seed,
                               pope_file=args.pope_file or None)
            with acab_session(lm, cfg) as ctrl:
                m = run_pope(lm, samples, ctrl, batch_size=args.pope_batch_size)
            results["pope"][split] = m
            print(f"[pope:{split}] acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
                  f"yes={m['yes_ratio']:.3f} ece={m.get('ece', float('nan')):.4f}")

    # ---- CHAIR -------------------------------------------------------------- #
    if "chair" in tasks:
        samples = build_caption_set(args.coco_root, args.eval_subset,
                                    n_images=args.n_chair_images, seed=args.seed)
        with acab_session(lm, cfg) as ctrl:
            r = run_captions(lm, samples, ctrl, batch_size=args.chair_batch_size,
                             max_new_tokens=args.max_new_tokens, record_trace=True)
        scorer = ChairScorer(args.coco_root, args.eval_subset, args.synonyms)
        sc = scorer.score(r["records"])
        sc.pop("hallucinated", None)
        sc["coverage"] = scorer.coverage(r["records"])
        sc.update(r["fluency"])
        if r["traces"]:
            g = [x for tr in r["traces"] for row in tr["gated"] for x in row]
            sc["gated_frac"] = sum(g) / max(len(g), 1)
        results["chair"] = sc                      # includes per_caption counts
        results["chair_image_ids"] = [int(x["image_id"]) for x in r["records"]]
        results["captions"] = r["records"]         # all of them: needed to audit CHAIR
        print(f"[chair] CHAIR_s={sc['CHAIR_s']:.4f} CHAIR_i={sc['CHAIR_i']:.4f} "
              f"coverage={sc['coverage']:.4f} len={sc['avg_len']:.1f} "
              f"rep4={sc['rep4']:.3f} gated={sc.get('gated_frac', 0):.3f}")

        if args.ppl:
            refs = load_captions(args.coco_root, args.eval_subset)
            results["ppl"] = caption_perplexity(lm, samples, refs)
            print(f"[ppl] reference-caption PPL = {results['ppl']:.3f}")

    d = run_dir(args, "eval")
    save_json(results, d / "results.json")
    print(f"\nwritten to {d/'results.json'}")


if __name__ == "__main__":
    main()
