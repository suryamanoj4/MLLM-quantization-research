#!/usr/bin/env python
"""Offline tests for the mini-COCO data layer.

Builds a synthetic COCO-shaped dataset in a temp directory (no network, no model)
and checks the things that would otherwise only fail after a 40-minute eval run:
annotation subsetting, the calibration/evaluation partition, official-POPE
preference, and CHAIR/POPE scoring against known ground truth.

    python tests/test_data.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from phoenix.chair import COCO_80, ChairScorer
from phoenix.data import (build_calibration_set, build_caption_set, build_pope,
                          check_root, get_pope, load_instances, split_pool)
from phoenix.fetch import (filter_pope, image_dest, pope_image_ids, read_splits,
                           subset_captions, subset_instances, write_splits)

OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, cond, detail=""):
    print(f"[{OK if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------------------- #
# synthetic COCO
# --------------------------------------------------------------------------- #
N_IMAGES = 400
N_POPE_IMAGES = 120
SUBSET = "val2014"


def build_fake_full_coco(cache: Path, rng: random.Random):
    """A COCO-shaped instances/captions pair covering N_IMAGES images."""
    cats = [{"id": i + 1, "name": n, "supercategory": "thing"}
            for i, n in enumerate(COCO_80)]
    by_name = {c["name"]: c["id"] for c in cats}
    images, anns, caps, gt = [], [], [], {}
    ann_id = cap_id = 1
    for k in range(N_IMAGES):
        iid = 1000 + k * 7
        images.append({"id": iid, "file_name": f"COCO_{SUBSET}_{iid:012d}.jpg",
                       "width": 640, "height": 480})
        objs = rng.sample(COCO_80, rng.randint(1, 5))
        gt[iid] = set(objs)
        for o in objs:
            anns.append({"id": ann_id, "image_id": iid, "category_id": by_name[o],
                         "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0})
            ann_id += 1
        for _ in range(5):
            caps.append({"id": cap_id, "image_id": iid,
                         "caption": "A photo of " + " and ".join(objs) + "."})
            cap_id += 1
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"instances_{SUBSET}.json").write_text(json.dumps(
        {"info": {}, "licenses": [], "images": images, "annotations": anns,
         "categories": cats}))
    (cache / f"captions_{SUBSET}.json").write_text(json.dumps(
        {"info": {}, "licenses": [], "images": images, "annotations": caps}))
    return [im["id"] for im in images], gt


def build_fake_pope(cache: Path, image_ids, gt, rng: random.Random):
    """POPE-format jsonl over the first N_POPE_IMAGES images, 6 questions each."""
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("random", "popular", "adversarial"):
        rows, qid = [], 1
        for iid in image_ids[:N_POPE_IMAGES]:
            present = sorted(gt[iid])
            absent = [c for c in COCO_80 if c not in gt[iid]]
            for o in (present * 3)[:3]:
                rows.append({"question_id": qid, "image": f"COCO_{SUBSET}_{iid:012d}.jpg",
                             "text": f"Is there a {o} in the image?", "label": "yes"})
                qid += 1
            for o in rng.sample(absent, 3):
                rows.append({"question_id": qid, "image": f"COCO_{SUBSET}_{iid:012d}.jpg",
                             "text": f"Is there a {o} in the image?", "label": "no"})
                qid += 1
        p = cache / f"coco_pope_{split}.json"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        out[split] = p
    return out


def write_images(root: Path, ids):
    """Noise images at a realistic size -- a flat 64x48 JPEG compresses to under the
    1 KB threshold that download/verify use to spot truncated downloads."""
    import numpy as np
    (root / SUBSET).mkdir(parents=True, exist_ok=True)
    rs = np.random.RandomState(0)
    for iid in ids:
        arr = rs.randint(0, 256, (96, 128, 3), dtype=np.uint8)
        Image.fromarray(arr).save(image_dest(root, iid, SUBSET), quality=85)


# =========================================================================== #
def main():
    rng = random.Random(0)
    tmp = Path(tempfile.mkdtemp(prefix="phoenix-minicoco-"))
    root, cache = tmp / "coco-mini", tmp / "_cache"
    try:
        print("== building a synthetic COCO ==")
        all_ids, gt = build_fake_full_coco(cache, rng)
        pope_files = build_fake_pope(cache, all_ids, gt, rng)
        pope_ids = pope_image_ids(pope_files["random"])
        check("POPE image ids parse out of the jsonl", len(pope_ids) == N_POPE_IMAGES,
              f"n={len(pope_ids)}")

        n_eval, n_calib = 60, 48
        eval_ids = pope_ids[:n_eval]
        pool = sorted(set(all_ids) - set(pope_ids))
        rng.shuffle(pool)
        calib_ids = sorted(pool[:n_calib])
        keep = set(eval_ids) | set(calib_ids)

        print("\n== annotation subsetting ==")
        objs = subset_instances(cache / f"instances_{SUBSET}.json", keep,
                                root / "annotations" / f"instances_{SUBSET}.json",
                                log=lambda *_: None)
        subset_captions(cache / f"captions_{SUBSET}.json", keep,
                        root / "annotations" / f"captions_{SUBSET}.json",
                        log=lambda *_: None)
        sub = json.loads((root / "annotations" / f"instances_{SUBSET}.json").read_text())
        got = {im["id"] for im in sub["images"]}
        check("subset contains exactly the kept images", got == keep,
              f"{len(got)} vs {len(keep)}")
        check("no stray annotations survive",
              all(a["image_id"] in keep for a in sub["annotations"]))
        check("all 80 categories are preserved", len(sub["categories"]) == 80)
        check("returned object map matches the source ground truth",
              all(objs[i] == gt[i] for i in keep), )
        full_sz = (cache / f"instances_{SUBSET}.json").stat().st_size
        sub_sz = (root / "annotations" / f"instances_{SUBSET}.json").stat().st_size
        check("subset is much smaller than the source", sub_sz < full_sz / 2,
              f"{full_sz/1e3:.0f} KB -> {sub_sz/1e3:.0f} KB")

        # both parser backends must agree
        import phoenix.fetch as F
        a = F._load_coco_json(cache / f"instances_{SUBSET}.json", keep, log=lambda *_: None)
        try:
            import ijson  # noqa: F401
            have_ijson = True
        except ImportError:
            have_ijson = False
        if have_ijson:
            import builtins
            orig = builtins.__import__

            def no_ijson(name, *a_, **k_):
                if name == "ijson":
                    raise ImportError("disabled for the test")
                return orig(name, *a_, **k_)
            builtins.__import__ = no_ijson
            try:
                b = F._load_coco_json(cache / f"instances_{SUBSET}.json", keep,
                                      log=lambda *_: None)
            finally:
                builtins.__import__ = orig
            same = (len(a["images"]) == len(b["images"])
                    and len(a["annotations"]) == len(b["annotations"])
                    and {i["id"] for i in a["images"]} == {i["id"] for i in b["images"]})
            check("ijson streaming and in-memory parsing agree", same,
                  f"{len(a['annotations'])} annotations both ways")
        else:
            check("ijson streaming and in-memory parsing agree", True, "(ijson absent)")

        print("\n== POPE filtering and the split manifest ==")
        for s, f in pope_files.items():
            n = filter_pope(f, set(eval_ids), root / "pope" / f"coco_pope_{s}.json")
            check(f"POPE '{s}' filtered to the eval pool", n == n_eval * 6,
                  f"{n} questions")
        write_images(root, keep)
        write_splits(root, eval_ids, calib_ids, SUBSET)

        try:
            write_splits(root / "bad", eval_ids, eval_ids[:3] + calib_ids, SUBSET)
            check("write_splits refuses an overlapping partition", False)
        except AssertionError:
            check("write_splits refuses an overlapping partition", True)

        sp = read_splits(root)
        check("manifest round-trips", set(sp["eval_ids"]) == set(eval_ids)
              and set(sp["calib_ids"]) == set(calib_ids))

        print("\n== the samplers respect the partition ==")
        check_root(root)
        check("split_pool exposes both pools",
              split_pool(root, "eval") == set(eval_ids)
              and split_pool(root, "calib") == set(calib_ids))

        cap = build_caption_set(root, SUBSET, n_images=40, seed=0)
        check("caption set draws only from the eval pool",
              all(s.image_id in set(eval_ids) for s in cap), f"n={len(cap)}")

        calib = build_calibration_set(root, "train2014", n_images=32, seed=0,
                                      fallback_subset=SUBSET)
        check("calibration set draws only from the calib pool",
              all(s.image_id in set(calib_ids) for s in calib), f"n={len(calib)}")
        check("CALIBRATION AND EVALUATION DO NOT OVERLAP",
              not ({s.image_id for s in calib} & {s.image_id for s in cap}))

        loc = build_pope(root, SUBSET, "adversarial", n_images=20, seed=0)
        check("locally constructed POPE stays in the eval pool",
              all(s.image_id in set(eval_ids) for s in loc), f"n={len(loc)}")
        check("local POPE is balanced",
              sum(1 for s in loc if s.label == "yes") * 2 == len(loc))

        off = get_pope(root, SUBSET, "adversarial", n_images=20, verbose=False)
        check("get_pope prefers the official file",
              len(off) == 120 and len({s.image_id for s in off}) == 20,
              f"{len(off)} questions over {len({s.image_id for s in off})} images")
        check("official POPE samples point at files that exist",
              all(Path(s.image_path).exists() for s in off))
        check("official POPE labels survive parsing",
              {s.label for s in off} == {"yes", "no"})

        print("\n== scoring against known ground truth ==")
        scorer = ChairScorer(root, SUBSET)
        iid = eval_ids[0]
        truth = sorted(gt[iid])
        perfect = "A photo of " + " and ".join(truth) + "."
        r = scorer.score([{"image_id": iid, "caption": perfect}])
        check("a perfectly grounded caption has CHAIR 0",
              r["CHAIR_s"] == 0 and r["CHAIR_i"] == 0,
              f"s={r['CHAIR_s']} i={r['CHAIR_i']}")
        check("coverage is 1.0 for that caption",
              abs(scorer.coverage([{"image_id": iid, "caption": perfect}]) - 1.0) < 1e-9)

        absent = next(c for c in COCO_80 if c not in gt[iid])
        r2 = scorer.score([{"image_id": iid, "caption": perfect[:-1] + f" and a {absent}."}])
        check("one hallucinated object is detected",
              r2["CHAIR_s"] == 1.0 and abs(r2["CHAIR_i"] - 1 / (len(truth) + 1)) < 1e-9,
              f"s={r2['CHAIR_s']} i={r2['CHAIR_i']:.3f}")
        short = scorer.score([{"image_id": iid, "caption": f"A {truth[0]}."}])
        cov = scorer.coverage([{"image_id": iid, "caption": f"A {truth[0]}."}])
        check("saying less games CHAIR but is caught by coverage",
              short["CHAIR_i"] == 0 and cov < 1.0,
              f"CHAIR_i={short['CHAIR_i']} coverage={cov:.2f}  <- why both are reported")

        from phoenix.metrics import pope_metrics
        preds = [s.label for s in off]
        m = pope_metrics(preds, [s.label for s in off])
        check("oracle predictions give F1 = 1", abs(m["f1"] - 1.0) < 1e-9)
        m2 = pope_metrics(["yes"] * len(off), [s.label for s in off])
        check("always-yes gives yes_ratio 1 and recall 1",
              abs(m2["yes_ratio"] - 1.0) < 1e-9 and abs(m2["recall"] - 1.0) < 1e-9,
              f"F1={m2['f1']:.3f} (the Lexical Fallback failure mode)")

        print("\n== verify() ==")
        sys.argv = ["fetch_data.py", "--root", str(root), "--verify"]
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import importlib
        fd = importlib.import_module("fetch_data")
        rc = fd.verify(root, SUBSET)
        check("verify() passes on a well-formed mini-COCO", rc == 0)

        (root / SUBSET / f"COCO_{SUBSET}_{sorted(keep)[0]:012d}.jpg").unlink()
        rc2 = fd.verify(root, SUBSET)
        check("verify() fails when an image is missing", rc2 != 0)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("all data tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
