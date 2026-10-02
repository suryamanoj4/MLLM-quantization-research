#!/usr/bin/env python
"""Build a mini-COCO for Phoenix: ~100 MB instead of ~20 GB.

Nothing here is trained, so the project only needs a few hundred evaluation images,
a couple of hundred disjoint calibration images, and the ground truth for exactly
those. This script fetches individual images rather than the 6.2 GB val2014 zip, and
pulls only the two annotation members it needs out of the 241 MB annotations zip
using HTTP range requests.

    python scripts/fetch_data.py --root data/coco-mini                 # ~110 MB
    python scripts/fetch_data.py --root data/coco-mini --preset tiny   # ~40 MB
    python scripts/fetch_data.py --root data/coco-mini --verify        # check only

Evaluation images come from the official POPE question set (all three splits share
the same 500 images), so POPE numbers stay comparable to the published ones.
Calibration images are drawn from val2014 images that appear in *no* POPE split, and
the partition is pinned in splits.json, which phoenix/data.py reads and enforces.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phoenix.fetch import (FetchError, bytes_on_disk, download_images, fetch_annotation_members,
                           fetch_pope, filter_pope, human, image_dest, pope_image_ids,
                           read_splits, subset_captions, subset_instances, write_splits)

PRESETS = {
    "tiny": dict(n_eval=100, n_calib=96),
    "dev": dict(n_eval=250, n_calib=192),
    "full": dict(n_eval=500, n_calib=256),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/coco-mini")
    p.add_argument("--subset", default="val2014")
    p.add_argument("--preset", choices=sorted(PRESETS), default="dev")
    p.add_argument("--n-eval", type=int, default=None,
                   help="evaluation images (POPE + CHAIR); overrides --preset")
    p.add_argument("--n-calib", type=int, default=None,
                   help="calibration images, disjoint from every POPE image")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--annotations-mode", choices=["auto", "range", "full"], default="auto",
                   help="auto: measure range vs plain-GET speed and take the faster; "
                        "range: fetch only the 2 needed members (~55 MB); "
                        "full: one plain GET of the 253 MB zip, extract, delete")
    p.add_argument("--connections", type=int, default=8,
                   help="parallel range connections for the annotation members")
    p.add_argument("--no-captions", action="store_true",
                   help="skip captions_val2014 (~30 MB); weakens CHAIR ground truth")
    p.add_argument("--annotations-zip", default=None,
                   help="use a local annotations_trainval2014.zip instead of fetching")
    p.add_argument("--keep-cache", action="store_true",
                   help="keep the full annotation jsons after subsetting")
    p.add_argument("--verify", action="store_true", help="verify an existing root and exit")
    p.add_argument("--dry-run", action="store_true", help="print the plan, fetch nothing")
    args = p.parse_args()

    root = Path(args.root)
    n_eval = args.n_eval if args.n_eval is not None else PRESETS[args.preset]["n_eval"]
    n_calib = args.n_calib if args.n_calib is not None else PRESETS[args.preset]["n_calib"]

    if args.verify:
        return verify(root, args.subset)

    cache = root / "_cache"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        anc = next((q for q in [root, *root.parents] if q.exists()), root)
        why = "no write permission on" if isinstance(e, PermissionError) else "cannot create under"
        print(f"[error] cannot create {root}: {why} {anc} ({e.strerror}).\n"
              f"        Use a path you own, e.g.  --root data/coco-mini", file=sys.stderr)
        return 2
    print(f"[plan] root={root}  eval={n_eval} images  calib={n_calib} images  "
          f"subset={args.subset}")
    print(f"[plan] estimated final size ~{human((n_eval + n_calib) * 160e3 + 6e6)}")
    if args.dry_run:
        print("[plan] --dry-run, stopping here")
        return

    # ---- 1. POPE question files (tiny) ------------------------------------- #
    pope_dir = root / "pope"
    pope_files = fetch_pope(cache / "pope")
    pope_all_ids = pope_image_ids(pope_files["random"])
    for s, f in pope_files.items():
        ids = set(pope_image_ids(f))
        if ids != set(pope_all_ids):
            print(f"[warn] POPE split '{s}' references a different image set; "
                  f"using the union for the leakage guard")
            pope_all_ids = sorted(set(pope_all_ids) | ids)
    print(f"[pope] {len(pope_all_ids)} unique images across {len(pope_files)} splits")
    eval_ids = pope_all_ids[:n_eval]

    # ---- 2. annotations, minimally ----------------------------------------- #
    want = ["instances"] + ([] if args.no_captions else ["captions"])
    kw = {"url": args.annotations_zip} if args.annotations_zip else {}
    if args.annotations_zip and not str(args.annotations_zip).startswith("http"):
        full = _from_local_zip(Path(args.annotations_zip), args.subset, want, cache)
    else:
        full = fetch_annotation_members(args.subset, want, cache,
                                        mode=args.annotations_mode,
                                        connections=args.connections, **kw)

    all_ids = _all_image_ids(full["instances"])
    print(f"[ann] {args.subset} has {len(all_ids)} annotated images")

    # ---- 3. calibration pool: disjoint from EVERY POPE image ---------------- #
    pool = sorted(set(all_ids) - set(pope_all_ids))
    rng = random.Random(args.seed + 7)
    rng.shuffle(pool)
    calib_ids = sorted(pool[:n_calib])
    assert not (set(calib_ids) & set(pope_all_ids))
    print(f"[split] eval={len(eval_ids)}  calib={len(calib_ids)} "
          f"(drawn from {len(pool)} images in no POPE split)")

    keep = set(eval_ids) | set(calib_ids)

    # ---- 4. subset the annotations ----------------------------------------- #
    ann_dir = root / "annotations"
    subset_instances(full["instances"], keep, ann_dir / f"instances_{args.subset}.json")
    if not args.no_captions:
        subset_captions(full["captions"], keep, ann_dir / f"captions_{args.subset}.json")

    # ---- 5. images ---------------------------------------------------------- #
    failed = download_images(root, sorted(keep), args.subset, workers=args.workers)
    if failed:
        print(f"[img] {len(failed)} failed; retrying once")
        failed = download_images(root, failed, args.subset, workers=4)
    if failed:
        print(f"[img] still failing: {failed[:10]}{' ...' if len(failed) > 10 else ''}")
        eval_ids = [i for i in eval_ids if i not in set(failed)]
        calib_ids = [i for i in calib_ids if i not in set(failed)]
        print(f"[img] dropped them from the splits -> eval={len(eval_ids)} "
              f"calib={len(calib_ids)}")

    # ---- 6. POPE filtered to the kept evaluation images --------------------- #
    for s, f in pope_files.items():
        n = filter_pope(f, set(eval_ids), pope_dir / f"coco_pope_{s}.json")
        print(f"[pope] {s}: {n} questions over {len(eval_ids)} images")

    # ---- 7. manifest -------------------------------------------------------- #
    sp = write_splits(root, eval_ids, calib_ids, args.subset,
                      meta={"seed": args.seed, "preset": args.preset,
                            "pope_images_excluded_from_calib": len(pope_all_ids),
                            "source": "official POPE + COCO val2014"})
    print(f"[split] wrote {sp}")

    if not args.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
        print("[cache] removed _cache (pass --keep-cache to keep the full annotations)")

    print(f"\n[done] {root} = {human(bytes_on_disk(root))}")
    verify(root, args.subset)
    print(f"\nNext:\n  python scripts/00_precision_ladder.py --coco-root {root} "
          f"--ladder fp16,w4a8,w4a4")


def _all_image_ids(path: Path) -> list[int]:
    import json
    from phoenix.fetch import _ijson_fast, _ProgressReader, progress
    ijson = _ijson_fast()
    if ijson is None:
        print(f"[ann] indexing {path.name} in memory (~10-30 s)")
        with open(path) as f:
            return [int(im["id"]) for im in json.load(f)["images"]]
    with open(path, "rb") as f, progress(total=path.stat().st_size,
                                         desc=f"[ann] index {path.name}") as bar:
        return [int(im["id"]) for im in ijson.items(_ProgressReader(f, bar), "images.item")]


def _from_local_zip(zpath: Path, subset: str, want, cache: Path) -> dict:
    import zipfile
    from phoenix.fetch import WANTED_MEMBERS
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    with zipfile.ZipFile(zpath) as z:
        for kind in want:
            member = WANTED_MEMBERS[kind].format(subset=subset)
            dest = cache / Path(member).name
            if not dest.exists():
                print(f"[ann] extracting {member} from {zpath}")
                with z.open(member) as src, open(dest, "wb") as f:
                    shutil.copyfileobj(src, f, 8 << 20)
            out[kind] = dest
    return out


# --------------------------------------------------------------------------- #
def verify(root: Path, subset: str) -> int:
    import json
    print(f"\n[verify] {root}")
    ok = True

    sp = read_splits(root)
    if sp is None:
        print("  FAIL  splits.json missing")
        return 1
    ev, ca = set(sp["eval_ids"]), set(sp["calib_ids"])
    print(f"  ok    splits.json: {len(ev)} eval, {len(ca)} calib, "
          f"{len(ev & ca)} overlapping")
    if ev & ca:
        ok = False

    inst = root / "annotations" / f"instances_{subset}.json"
    if not inst.exists():
        print(f"  FAIL  {inst} missing")
        return 1
    with open(inst) as f:
        data = json.load(f)
    have = {im["id"] for im in data["images"]}
    missing = (ev | ca) - have
    print(f"  {'ok  ' if not missing else 'FAIL'}  annotations cover all split images "
          f"({len(have)} images, {len(data['annotations'])} objects, "
          f"{len(data['categories'])} categories)")
    ok &= not missing

    from PIL import Image
    bad, checked = [], 0
    for iid in sorted(ev | ca):
        p = image_dest(root, iid, subset)
        if not p.exists() or p.stat().st_size < 1024:
            bad.append(iid)
            continue
        if checked < 12:
            try:
                Image.open(p).convert("RGB").load()
                checked += 1
            except Exception:                         # noqa: BLE001
                bad.append(iid)
    print(f"  {'ok  ' if not bad else 'FAIL'}  {len(ev|ca)-len(bad)}/{len(ev|ca)} images "
          f"present ({checked} decoded as a spot check)")
    ok &= not bad

    pope = sorted((root / "pope").glob("coco_pope_*.json"))
    if pope:
        for f in pope:
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            labs = sum(1 for r in rows if r["label"] == "yes")
            print(f"  ok    {f.name}: {len(rows)} questions, "
                  f"{labs}/{len(rows)} positive")
    else:
        print("  note  no official POPE files; scripts will construct POPE locally")

    print(f"  size  {human(bytes_on_disk(root))}")
    print("[verify]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main() or 0)
    except FetchError as e:
        print(f"\n[error] {e}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("\n[abort] interrupted; rerun to resume from what was downloaded",
              file=sys.stderr)
        raise SystemExit(130) from None
