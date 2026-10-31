#!/usr/bin/env python
"""Which objects does a quantized model stop mentioning?

Uses the captions a ladder run already saved (no GPU). For every (image, object
category) that is annotated in COCO *and* mentioned by the reference (fp16) caption,
a rung "drops" it if its own caption no longer mentions it; it "gains" an annotated
object fp16 missed. Rates are broken down by object size -- the largest instance of
that category as a fraction of the image area -- and by category.

If quantization erases visual detail, drops should concentrate on small objects, and
more so than under the noise control (method=noise). A size-flat drop pattern points
to a generic language-side change in what the captions talk about.

    python scripts/10_object_analysis.py --coco-root data/coco-mini --ladder runs/gate1/ladder.json
    python scripts/10_object_analysis.py --coco-root data/coco-mini --ladder runs/gate2/ladder.json
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phoenix.chair import ChairScorer
from phoenix.utils import save_json

BUCKETS = [("<1%", 0.0, 0.01), ("1-5%", 0.01, 0.05), ("5-20%", 0.05, 0.20), (">20%", 0.20, 1.01)]


def load_objects(root: Path, subset: str) -> dict[int, dict[str, float]]:
    """image_id -> {category name: largest instance area / image area}."""
    d = json.loads((root / "annotations" / f"instances_{subset}.json").read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    size = {im["id"]: im.get("width", 640) * im.get("height", 480) for im in d["images"]}
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for a in d["annotations"]:
        iid, name = a["image_id"], cats.get(a["category_id"])
        if name is None:
            continue
        frac = float(a.get("area", 0.0)) / max(size.get(iid, 1), 1)
        out[iid][name] = max(out[iid].get(name, 0.0), frac)
    return out


def bucket(frac: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= frac < hi:
            return name
    return BUCKETS[-1][0]


def pairs(ref_caps, row_caps, objects, scorer):
    """Per image: list of (category, size bucket, ref mentions?, row mentions?)."""
    out = {}
    for r_ref, r_row in zip(ref_caps, row_caps):
        iid = int(r_ref["image_id"])
        assert iid == int(r_row["image_id"]), "caption sets differ between rungs"
        m_ref, m_row = scorer.extract(r_ref["caption"]), scorer.extract(r_row["caption"])
        out[iid] = [(c, bucket(f), c in m_ref, c in m_row) for c, f in objects.get(iid, {}).items()]
    return out


def rates(per_img, ids):
    acc = defaultdict(lambda: [0, 0, 0, 0])      # kept, dropped, missed, gained
    for i in ids:
        for _, b, a, q in per_img[i]:
            for key in (b, "all"):
                e = acc[key]
                if a:
                    e[0] += 1
                    e[1] += (not q)
                else:
                    e[2] += 1
                    e[3] += q
    return acc


def drop_rate(acc, key):
    k, d = acc[key][0], acc[key][1]
    return d / k if k else float("nan")


def boot(per_img, n_boot=1000, seed=0):
    ids = list(per_img)
    rng = random.Random(seed)
    keys = [b for b, _, _ in BUCKETS] + ["all"]
    samples = {k: [] for k in keys + ["small_minus_large"]}
    for _ in range(n_boot):
        s = [rng.choice(ids) for _ in ids]
        acc = rates(per_img, s)
        for k in keys:
            samples[k].append(drop_rate(acc, k))
        samples["small_minus_large"].append(drop_rate(acc, "<1%") - drop_rate(acc, ">20%"))
    ci = {}
    for k, v in samples.items():
        v = sorted(x for x in v if x == x)
        ci[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (float("nan"),) * 2
    return ci


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco-root", required=True)
    p.add_argument("--ladder", required=True, help="a ladder.json with saved captions")
    p.add_argument("--ref", default="fp16")
    p.add_argument("--subset", default="val2014")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.coco_root)
    table = json.loads(Path(a.ladder).read_text())["table"]
    objects = load_objects(root, a.subset)
    scorer = ChairScorer(root, a.subset)
    ref = table[a.ref]["captions"]

    # how often does the reference mention an annotated object, by size?
    base = rates(pairs(ref, ref, objects, scorer), [int(r["image_id"]) for r in ref])
    print(f"{len(ref)} captions. Reference ({a.ref}) mentions annotated objects:")
    for b, _, _ in BUCKETS:
        k, m = base[b][0], base[b][2]
        print(f"   {b:>6}: {k}/{k + m} = {k / max(k + m, 1):.2f}")

    hdr = "".join(f"{b:>15}" for b, _, _ in BUCKETS)
    print(f"\nshare of the reference's mentioned objects that each rung DROPS (95% CI over images)")
    print(f"{'rung':<34}{hdr}{'all':>15}{'small-large':>16}{'gained':>8}")
    result = {"reference": a.ref, "reference_mention_rate": {
        b: base[b][0] / max(base[b][0] + base[b][2], 1) for b, _, _ in BUCKETS}, "rungs": {}}
    cat_drops: dict[str, Counter] = {}
    for name, row in table.items():
        if name == a.ref or "captions" not in row:
            continue
        per_img = pairs(ref, row["captions"], objects, scorer)
        acc = rates(per_img, list(per_img))
        ci = boot(per_img, a.n_boot)
        cells = []
        for b in [x for x, _, _ in BUCKETS] + ["all"]:
            r = drop_rate(acc, b)
            cells.append(f"{r:.2f} [{ci[b][0]:.2f},{ci[b][1]:.2f}]".rjust(15))
        sml = drop_rate(acc, "<1%") - drop_rate(acc, ">20%")
        lo, hi = ci["small_minus_large"]
        star = "*" if not (lo <= 0 <= hi) else " "
        gained = acc["all"][3]
        print(f"{name:<34}{''.join(cells)}{f'{sml:+.2f}{star}':>16}{gained:>8}")
        cnt = Counter(c for v in per_img.values() for c, _, x, y in v if x and not y)
        cat_drops[name] = cnt
        result["rungs"][name] = {
            "drop_rate": {b: drop_rate(acc, b) for b in [x for x, _, _ in BUCKETS] + ["all"]},
            "ci": ci, "kept": {b: acc[b][0] for b in acc}, "dropped": {b: acc[b][1] for b in acc},
            "gained": gained, "missed_by_ref": acc["all"][2], "top_dropped": cnt.most_common(10)}

    pooled = sum(cat_drops.values(), Counter())
    ref_mentions = Counter(c for r in ref for c in scorer.extract(r["caption"])
                           if c in objects.get(int(r["image_id"]), {}))
    print("\nmost-dropped categories, pooled over rungs (drops / reference mentions per rung):")
    n_r = max(len(cat_drops), 1)
    for c, n in pooled.most_common(12):
        print(f"   {c:<16} {n:>4} drops   {n / n_r / max(ref_mentions[c], 1):.2f} per mention")
    print("\nReading: compare each method's size profile with method=noise. Drops concentrated in"
          " small objects beyond the noise control = visual detail lost to quantization.")
    out = Path(a.out) if a.out else Path(a.ladder).with_name("object_analysis.json")
    save_json(result, out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
