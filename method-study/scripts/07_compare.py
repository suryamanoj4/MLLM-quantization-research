#!/usr/bin/env python
"""Paired comparison of two 03_eval.py runs (same questions, same images).

    python scripts/07_compare.py runs/eval__w4a5_base runs/eval__w4a5_loras
    python scripts/07_compare.py runs/eval__w4a5_base runs/eval__w4a5_{loras,acab,both}

Prints, for each later run against the first, the change in POPE F1 / accuracy per
split and in CHAIR_i / CHAIR_s, with 95% CIs from an image-clustered bootstrap.
'*' marks a CI that excludes zero. An unpaired "0.887 vs 0.885" comparison cannot
tell you whether a 0.002 change is real; this can.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phoenix.stats import compare_chair, compare_pope, fmt_delta


def load(d):
    f = Path(d) / "results.json" if Path(d).is_dir() else Path(d)
    return f.parent.name, json.loads(f.read_text())


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    base_name, base = load(sys.argv[1])
    n_boot = 2000
    for other in sys.argv[2:]:
        name, res = load(other)
        print(f"\n{name}  vs  {base_name}")
        for split, m in res.get("pope", {}).items():
            b = base.get("pope", {}).get(split)
            if not b or "preds" not in m or "preds" not in b:
                print(f"  POPE/{split:12s} (no per-question data -- rerun with the current 03_eval.py)")
                continue
            if m["image_ids"] != b["image_ids"]:
                print(f"  POPE/{split:12s} question sets differ; not comparable")
                continue
            f1 = compare_pope(m["preds"], b["preds"], m["labels"], m["image_ids"], "f1", n_boot)
            acc = compare_pope(m["preds"], b["preds"], m["labels"], m["image_ids"], "acc", n_boot)
            print(f"  POPE/{split:12s} F1 {m['f1']:.4f}  dF1 {fmt_delta(f1)}   "
                  f"dAcc {fmt_delta(acc)}   yes {m['yes_ratio']:.3f} (was {b['yes_ratio']:.3f})")
        ca, cb = res.get("chair", {}), base.get("chair", {})
        if "per_caption" in ca and "per_caption" in cb and \
                res.get("chair_image_ids") == base.get("chair_image_ids") and res.get("chair_image_ids"):
            ids = res["chair_image_ids"]
            ci = compare_chair(ca["per_caption"], cb["per_caption"], ids, "CHAIR_i", n_boot)
            cs = compare_chair(ca["per_caption"], cb["per_caption"], ids, "CHAIR_s", n_boot)
            print(f"  CHAIR_i {ca['CHAIR_i']:.4f}  d {fmt_delta(ci)}   "
                  f"CHAIR_s {ca['CHAIR_s']:.4f}  d {fmt_delta(cs)}")
            print(f"  coverage {ca.get('coverage', float('nan')):.3f} (was {cb.get('coverage', float('nan')):.3f})"
                  f"   len {ca.get('avg_len', float('nan')):.1f} (was {cb.get('avg_len', float('nan')):.1f})"
                  f"   gated {ca.get('gated_frac', 0):.2f}")
            if ci["delta"] < 0 and ca.get("coverage", 1) < 0.9 * cb.get("coverage", 1):
                print("  NOTE: CHAIR fell but coverage fell >10% too -- partly a 'says less' effect")
        elif ca:
            print("  CHAIR: no per-caption data in one of the runs -- rerun with the current 03_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
