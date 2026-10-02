from __future__ import annotations

import json
import pathlib
import random

POPE_SPLITS = ("random", "popular", "adversarial")


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pope_questions(data_dir: pathlib.Path) -> dict[str, list[dict]]:
    pope_dir = data_dir / "pope"
    return {s: load_jsonl(pope_dir / f"coco_pope_{s}.json") for s in POPE_SPLITS}


def resample_pope(questions: list[dict], n_images: int | None, seed: int) -> list[dict]:
    by_image: dict[str, list[dict]] = {}
    for q in questions:
        by_image.setdefault(q["image"], []).append(q)
    images = sorted(by_image)
    if n_images is not None and n_images < len(images):
        rng = random.Random(seed)
        rng.shuffle(images)
        images = images[:n_images]
    out = []
    for img in images:
        out.extend(by_image[img])
    return out


def resample_chair_images(images: list[dict], n_images: int | None, seed: int) -> list[dict]:
    if n_images is None or n_images >= len(images):
        return images
    rng = random.Random(seed)
    return rng.sample(images, n_images)


def base_seed_for(image_name: str, seed: int) -> int:
    return seed + sum(ord(c) for c in image_name)


def load_chair_image_list(data_dir: pathlib.Path) -> list[dict]:
    path = data_dir / "chair" / "chair_images.json"
    if path.exists():
        return json.loads(path.read_text())
    return []