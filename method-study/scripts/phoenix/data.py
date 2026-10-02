"""COCO / POPE / CHAIR data plumbing.

Directory layout expected (`--coco-root`):
    coco/
      val2014/COCO_val2014_000000000042.jpg ...
      annotations/instances_val2014.json
      annotations/captions_val2014.json          (optional, improves CHAIR ground truth)
      train2014/ ...                             (optional, used for calibration)

POPE questions can either be loaded from the official release
(https://github.com/RUCAIBox/POPE, `--pope-file`) or reconstructed locally with the
same sampling rules. Use the official files for numbers you intend to publish;
local construction exists so the pipeline runs with nothing but COCO on disk, and
so you can build POPEv2-style variants.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image


@dataclass
class Sample:
    image_id: int
    image_path: str
    question: str
    label: str | None = None       # "yes" / "no" for POPE
    obj: str | None = None
    split: str | None = None


# --------------------------------------------------------------------------- #
# split manifest (written by scripts/fetch_data.py)
# --------------------------------------------------------------------------- #
def split_pool(root: str | Path, which: str) -> set[int] | None:
    """Return the allowed image ids for 'eval' or 'calib', or None if unpinned.

    When a mini-COCO built by scripts/fetch_data.py is used, every sampler below is
    restricted to its own pool, so calibration images can never reach evaluation.
    """
    from .fetch import read_splits
    s = read_splits(root)
    if s is None:
        return None
    return set(s["eval_ids"] if which == "eval" else s["calib_ids"])


def _restrict(img_ids, pool: set[int] | None, what: str) -> list[int]:
    if pool is None:
        return list(img_ids)
    keep = [i for i in img_ids if i in pool]
    if not keep:
        raise RuntimeError(
            f"splits.json pins the {what} pool but none of its {len(pool)} images are "
            "in the annotations. Rebuild with scripts/fetch_data.py."
        )
    return keep


def coco_image_path(root: str | Path, image_id: int, subset: str = "val2014") -> Path:
    root = Path(root)
    if subset.endswith("2014"):
        return root / subset / f"COCO_{subset}_{image_id:012d}.jpg"
    return root / subset / f"{image_id:012d}.jpg"


def load_instances(root: str | Path, subset: str = "val2014") -> tuple[dict[int, set[str]], list[int]]:
    """Return {image_id: {category names}} and the ordered image id list."""
    path = Path(root) / "annotations" / f"instances_{subset}.json"
    with open(path) as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    objs: dict[int, set[str]] = defaultdict(set)
    for a in data["annotations"]:
        objs[a["image_id"]].add(cats[a["category_id"]])
    img_ids = [im["id"] for im in data["images"] if im["id"] in objs]
    return objs, sorted(img_ids)


def load_captions(root: str | Path, subset: str = "val2014") -> dict[int, list[str]]:
    path = Path(root) / "annotations" / f"captions_{subset}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    out: dict[int, list[str]] = defaultdict(list)
    for a in data["annotations"]:
        out[a["image_id"]].append(a["caption"])
    return out


# --------------------------------------------------------------------------- #
# POPE
# --------------------------------------------------------------------------- #
POPE_TEMPLATE = "Is there a {obj} in the image?"


def build_pope(root: str | Path, subset: str = "val2014", split: str = "random",
               n_images: int = 100, per_image: int = 6, seed: int = 0) -> list[Sample]:
    root = Path(root)
    objs, img_ids = load_instances(root, subset)
    rng = random.Random(seed)
    img_ids = _restrict(img_ids, split_pool(root, "eval"), "eval")
    rng.shuffle(img_ids)
    img_ids = img_ids[:n_images]

    all_cats = sorted({c for s in objs.values() for c in s})
    freq = Counter(c for iid in objs for c in objs[iid])
    popular = [c for c, _ in freq.most_common()]

    co = defaultdict(Counter)
    for s in objs.values():
        for a in s:
            for b in s:
                if a != b:
                    co[a][b] += 1

    out: list[Sample] = []
    n_pos = per_image // 2
    for iid in img_ids:
        present = sorted(objs[iid])
        if not present:
            continue
        pos = [present[i % len(present)] for i in range(n_pos)]
        absent_pool = [c for c in all_cats if c not in objs[iid]]
        if split == "random":
            neg = rng.sample(absent_pool, min(n_pos, len(absent_pool)))
        elif split == "popular":
            neg = [c for c in popular if c in set(absent_pool)][:n_pos]
        elif split == "adversarial":
            score = Counter()
            for c in present:
                score.update(co[c])
            ranked = [c for c, _ in score.most_common() if c in set(absent_pool)]
            neg = (ranked + [c for c in popular if c in set(absent_pool)])[:n_pos]
        else:
            raise ValueError(f"unknown POPE split '{split}'")
        path = str(coco_image_path(root, iid, subset))
        for o in pos:
            out.append(Sample(iid, path, POPE_TEMPLATE.format(obj=o), "yes", o, split))
        for o in neg:
            out.append(Sample(iid, path, POPE_TEMPLATE.format(obj=o), "no", o, split))
    return out


def load_pope_file(path: str, root: str | Path, subset: str = "val2014") -> list[Sample]:
    """Read an official POPE jsonl (`image`, `text`, `label`)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            img = d["image"]
            m = re.search(r"(\d{6,})", img)
            iid = int(m.group(1)) if m else -1
            p = Path(root) / subset / img if not Path(img).is_absolute() else Path(img)
            if not p.exists():
                p = coco_image_path(Path(root), iid, subset)
            out.append(Sample(iid, str(p), d["text"], d["label"].strip().lower(),
                              None, d.get("category")))
    return out


def official_pope_path(root: str | Path, split: str) -> Path | None:
    """scripts/fetch_data.py drops the official question files in <root>/pope/."""
    p = Path(root) / "pope" / f"coco_pope_{split}.json"
    return p if p.exists() and p.stat().st_size > 0 else None


def get_pope(root: str | Path, subset: str, split: str, n_images: int = 100,
             seed: int = 0, pope_file: str | None = None,
             verbose: bool = True) -> list[Sample]:
    """Official POPE questions when available, locally constructed ones otherwise.

    Precedence: explicit --pope-file > <root>/pope/coco_pope_<split>.json > local
    construction. Official files keep the numbers comparable to the published ones;
    local construction only exists so the pipeline runs with nothing but COCO.
    """
    path = pope_file or official_pope_path(root, split)
    if path:
        samples = load_pope_file(str(path), root, subset)
        for s in samples:
            s.split = split
        keep = split_pool(root, "eval")
        if keep is not None:
            samples = [s for s in samples if s.image_id in keep]
        ids, trimmed = [], []
        for s in samples:
            if s.image_id not in ids:
                if len(ids) >= n_images:
                    continue
                ids.append(s.image_id)
            trimmed.append(s)
        if verbose:
            print(f"[pope:{split}] official questions: {len(trimmed)} over "
                  f"{len(ids)} images ({Path(path).name})")
        return trimmed
    if verbose:
        print(f"[pope:{split}] no official file found, constructing locally "
              f"(numbers will not match published POPE exactly)")
    return build_pope(root, subset, split, n_images=n_images, seed=seed)


# --------------------------------------------------------------------------- #
# CHAIR / caption set
# --------------------------------------------------------------------------- #
def build_caption_set(root: str | Path, subset: str = "val2014", n_images: int = 100,
                      seed: int = 0, prompt: str | None = None) -> list[Sample]:
    from .model import CAPTION_PROMPT
    root = Path(root)
    objs, img_ids = load_instances(root, subset)
    rng = random.Random(seed + 1)
    ids = _restrict(img_ids, split_pool(root, "eval"), "eval")
    rng.shuffle(ids)
    ids = ids[:n_images]
    q = prompt or CAPTION_PROMPT
    return [Sample(i, str(coco_image_path(root, i, subset)), q, None, None, "caption")
            for i in ids]


def build_calibration_set(root: str | Path, subset: str = "train2014", n_images: int = 256,
                          seed: int = 123, prompt: str | None = None,
                          fallback_subset: str = "val2014",
                          exclude_ids: Sequence[int] = ()) -> list[Sample]:
    """Calibration images MUST be disjoint from evaluation images."""
    from .model import CAPTION_PROMPT
    root = Path(root)
    try:
        objs, img_ids = load_instances(root, subset)
        use = subset
    except FileNotFoundError:
        objs, img_ids = load_instances(root, fallback_subset)
        use = fallback_subset
        print(f"[data] {subset} annotations not found; calibrating from a disjoint "
              f"slice of {fallback_subset}")
    ex = set(exclude_ids)
    pool = split_pool(root, "calib")
    if pool is not None:
        # a pinned mini-COCO: calibration may only ever see its own images
        ids = _restrict(img_ids, pool, "calib")
        ids = [i for i in ids if i not in ex]
    else:
        ids = [i for i in img_ids if i not in ex]
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[:n_images]
    q = prompt or CAPTION_PROMPT
    return [Sample(i, str(coco_image_path(root, i, use)), q, None, None, "calib")
            for i in ids]


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #
def open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def batches(samples: Sequence[Sample], lm, batch_size: int = 4,
            max_batches: int | None = None) -> Iterator[dict]:
    from .model import prepare_batch
    n = 0
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        imgs = [open_image(s.image_path) for s in chunk]
        yield prepare_batch(lm, imgs, [s.question for s in chunk])
        n += 1
        if max_batches is not None and n >= max_batches:
            return


def batches_with_samples(samples: Sequence[Sample], lm, batch_size: int = 4):
    from .model import prepare_batch
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        imgs = [open_image(s.image_path) for s in chunk]
        yield chunk, prepare_batch(lm, imgs, [s.question for s in chunk])


def check_root(root: str | Path) -> Path:
    root = Path(root)
    if not (root / "annotations").exists():
        raise FileNotFoundError(
            f"{root}/annotations not found. Expected layout:\n"
            f"  {root}/val2014/*.jpg\n"
            f"  {root}/annotations/instances_val2014.json\n"
            f"  {root}/splits.json            (optional but recommended)\n\n"
            f"Build a ~110 MB mini-COCO with:\n"
            f"  python scripts/fetch_data.py --root {root}\n"
            f"(a full COCO checkout also works unchanged)"
        )
    s = split_pool(root, "eval")
    if s is not None:
        print(f"[data] splits.json active: {len(s)} eval / "
              f"{len(split_pool(root, 'calib'))} calib images, enforced separately")
    return root
