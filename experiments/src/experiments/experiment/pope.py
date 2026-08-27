from __future__ import annotations

import json
import pathlib
import re
from typing import Any

from tqdm import tqdm

from ..config import Config
from ..data.datasets import base_seed_for, load_jsonl
from ..data.prompts import chat_prompt
from .decoding import AttentionTracker, decode

_YES = re.compile(r"^\s*(yes|yeah|yep)", re.IGNORECASE)
_NO = re.compile(r"^\s*(no|nope)", re.IGNORECASE)


def parse_answer(text: str) -> str:
    if _YES.search(text):
        return "yes"
    if _NO.search(text):
        return "no"
    low = text.lower()
    if "yes" in low:
        return "yes"
    return "no"


def run_pope(
    model,
    processor,
    tokenizer,
    questions: list[dict],
    cfg: Config,
    image_dir: pathlib.Path,
    max_new_tokens: int | None = None,
    out_path: pathlib.Path | None = None,
) -> list[dict]:
    mnt = max_new_tokens or cfg.max_new_tokens_pope
    records: list[dict] = []
    done: set[str] = set()
    if out_path is not None and out_path.exists():
        records = load_jsonl(out_path)
        done = {r["image"] for r in records}
        print(f"[pope] resuming: {len(done)} images already done")

    fh = open(out_path, "a") if out_path is not None else None
    try:
        for q in tqdm(questions, desc="pope", unit="q"):
            if q["image"] in done:
                continue
            from PIL import Image

            image = Image.open(image_dir / q["image"]).convert("RGB")
            prompt = chat_prompt(q["text"], with_image=True)
            seed = base_seed_for(q["image"], cfg.seed)
            tracker = AttentionTracker(store_full_rows=cfg.store_full_rows) if cfg.capture_attention else None
            text_out, _ = decode(
                model, processor, tokenizer, prompt, image, cfg, mnt, seed, tracker=tracker
            )
            answer = parse_answer(text_out)
            record = {
                "image": q["image"],
                "question": q["text"],
                "label": q["label"],
                "answer": answer,
                "correct": answer == q["label"],
                "raw": text_out,
                "attention": tracker.summary() if tracker is not None else None,
            }
            records.append(record)
            if fh is not None:
                fh.write(json.dumps(record, default=float) + "\n")
                fh.flush()
    finally:
        if fh is not None:
            fh.close()
    return records