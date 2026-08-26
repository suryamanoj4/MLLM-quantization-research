from __future__ import annotations

import re
from typing import Any

from ..config import Config
from ..data.datasets import base_seed_for
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
    image_dir,
    max_new_tokens: int | None = None,
) -> list[dict]:
    records: list[dict] = []
    mnt = max_new_tokens or cfg.max_new_tokens_pope
    for q in questions:
        from PIL import Image

        image = Image.open(image_dir / q["image"]).convert("RGB")
        prompt = chat_prompt(q["text"], with_image=True)
        seed = base_seed_for(q["image"], cfg.seed)
        tracker = AttentionTracker(store_full_rows=cfg.store_full_rows) if cfg.capture_attention else None
        text_out, _ = decode(
            model, processor, tokenizer, prompt, image, cfg, mnt, seed, tracker=tracker
        )
        answer = parse_answer(text_out)
        label = q["label"]
        records.append(
            {
                "image": q["image"],
                "question": q["text"],
                "label": label,
                "answer": answer,
                "correct": answer == label,
                "raw": text_out,
                "attention": tracker.summary() if tracker is not None else None,
            }
        )
    return records