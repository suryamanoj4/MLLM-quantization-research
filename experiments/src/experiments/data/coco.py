from __future__ import annotations

import concurrent.futures
import json
import pathlib
import random
import time
import zipfile

import requests

COCO_IMAGE_URL = "http://images.cocodataset.org/{split}/{filename}"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
NEEDED_ANNOTATIONS = ("instances_val2014.json", "captions_val2014.json", "captions_train2014.json")


def _download(url: str, dest: pathlib.Path, chunk: int = 1 << 20, retries: int = 3) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for c in r.iter_content(chunk):
                        f.write(c)
            return
        except (requests.RequestException, OSError) as e:
            if attempt == retries - 1:
                raise
            print(f"[download] retry {attempt + 1}/{retries} for {url}: {e}")
            time.sleep(2 ** attempt)


def _download_image(filename: str, split: str, out_dir: pathlib.Path) -> None:
    dest = out_dir / filename
    if dest.exists():
        return
    _download(COCO_IMAGE_URL.format(split=split, filename=filename), dest)


def download_images(filenames: list[str], split: str, out_dir: pathlib.Path, workers: int = 8) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda fn: _download_image(fn, split, out_dir), filenames))
    return out_dir


def download_annotations(ann_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    ann_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ann_dir / "annotations_trainval2014.zip"
    _download(ANNOTATIONS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in NEEDED_ANNOTATIONS:
            zf.extract(name, ann_dir)
    return {n: ann_dir / n for n in NEEDED_ANNOTATIONS}


def load_instances(path: pathlib.Path) -> dict[int, set[str]]:
    with open(path) as f:
        ann = json.load(f)
    cat = {c["id"]: c["name"] for c in ann["categories"]}
    result: dict[int, set[str]] = {img["id"]: set() for img in ann["images"]}
    for a in ann["annotations"]:
        result[a["image_id"]].add(cat[a["category_id"]])
    return result


def load_captions(path: pathlib.Path) -> dict[int, list[str]]:
    with open(path) as f:
        ann = json.load(f)
    result: dict[int, list[str]] = {}
    for c in ann["annotations"]:
        result.setdefault(c["image_id"], []).append(c["caption"])
    return result


def image_id_to_filename(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


def filename_to_image_id(filename: str) -> int:
    return int(filename.split("_")[-1].split(".")[0])


def sample_val2014_filenames(instances: dict[int, set[str]], n: int, seed: int) -> list[str]:
    ids = sorted(instances.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    return [image_id_to_filename(i) for i in ids[:n]]