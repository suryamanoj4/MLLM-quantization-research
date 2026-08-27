from __future__ import annotations

import json
import pathlib

from tqdm import tqdm

from ..config import Config
from ..data.datasets import base_seed_for, load_jsonl
from ..data.prompts import CHAIR_PROMPT, chat_prompt
from .chair_utils import extract_mentions_with_spans
from .decoding import AttentionTracker, decode


def run_chair(
    model,
    processor,
    tokenizer,
    image_list: list[dict],
    cfg: Config,
    image_dir: pathlib.Path,
    instances: dict[int, set[str]],
    captions: dict[int, list[str]],
    out_path: pathlib.Path | None = None,
) -> list[dict]:
    records: list[dict] = []
    done: set[str] = set()
    if out_path is not None and out_path.exists():
        records = load_jsonl(out_path)
        done = {r["image"] for r in records}
        print(f"[chair] resuming: {len(done)} images already done")

    fh = open(out_path, "a") if out_path is not None else None
    try:
        for item in tqdm(image_list, desc="chair", unit="img"):
            filename = item["image"]
            if filename in done:
                continue
            image_id = item["image_id"]
            from PIL import Image

            image = Image.open(image_dir / filename).convert("RGB")
            prompt = chat_prompt(CHAIR_PROMPT, with_image=True)
            seed = base_seed_for(filename, cfg.seed)
            tracker = AttentionTracker(store_full_rows=cfg.store_full_rows) if cfg.capture_attention else None
            text_out, token_ids = decode(
                model, processor, tokenizer, prompt, image, cfg, cfg.max_new_tokens_chair, seed, tracker=tracker
            )

            gt = set(instances.get(image_id, set()))
            for cap in captions.get(image_id, []):
                gt |= _mention_classes(cap)

            mentions = []
            offsets = tokenizer(text_out, return_offsets_mapping=True).offset_mapping
            for cls, start, end in extract_mentions_with_spans(text_out):
                tok_range = _tokens_for_span(offsets, start, end)
                attn = None
                if tracker is not None and tok_range:
                    mass = [tracker.step_mass[i] for i in tok_range if i < len(tracker.step_mass)]
                    attn = sum(mass) / len(mass) if mass else None
                mentions.append(
                    {"class": cls, "grounded": cls in gt, "attention": attn, "tokens": tok_range}
                )

            record = {
                "image": filename,
                "image_id": image_id,
                "caption": text_out,
                "n_mentions": len(mentions),
                "n_hallucinated": sum(1 for m in mentions if not m["grounded"]),
                "attention": tracker.summary() if tracker is not None else None,
                "mentions": mentions,
            }
            records.append(record)
            if fh is not None:
                fh.write(json.dumps(record, default=float) + "\n")
                fh.flush()
    finally:
        if fh is not None:
            fh.close()
    return records


def _mention_classes(caption: str) -> set[str]:
    return {m[0] for m in extract_mentions_with_spans(caption)}


def _tokens_for_span(offsets, start: int, end: int) -> list[int]:
    idx = []
    for i, (s, e) in enumerate(offsets):
        if s is None or e is None:
            continue
        if e <= start:
            continue
        if s >= end:
            break
        idx.append(i)
    return idx