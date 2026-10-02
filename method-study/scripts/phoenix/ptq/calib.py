"""Calibration sequences for the published PTQ methods.

Three sources, chosen with `calib=` in a precision spec:

  mm    image + caption, in LLaVA's chat format: what the deployed model actually
        sees (576 image tokens + ~60 text tokens per sequence). The MLLM-aware choice.
  text  the *same* COCO captions with the image removed, packed into sequences of the
        same token count as the mm ones. Isolates "did calibration see image tokens?"
        from "did calibration see this domain / this many tokens?".
  wiki  WikiText-2 train, random 512-token windows (AWQ's protocol is 128 x 512 tokens
        of generic text): what LLM toolchains calibrate on by default.

All sequences come from the calibration pool (splits.json), never from evaluation.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

from ..data import build_calibration_set, load_captions, open_image
from ..model import CAPTION_PROMPT, SYSTEM, build_prompt


def _tok(lm):
    return getattr(lm.processor, "tokenizer", lm.processor)


def _captions_for(root, samples) -> dict[int, list[str]]:
    caps: dict[int, list[str]] = {}
    for subset in ("train2014", "val2014"):
        try:
            caps.update(load_captions(root, subset))
        except FileNotFoundError:
            continue
    return {s.image_id: caps.get(s.image_id, []) for s in samples}


def mm_sequences(lm, root, n: int, seed: int = 0) -> list[dict]:
    samples = build_calibration_set(root, "train2014", n_images=n, seed=seed + 11,
                                    fallback_subset="val2014")
    caps = _captions_for(root, samples)
    rng = random.Random(seed)
    out = []
    for s in samples:
        c = caps.get(s.image_id) or ["A photo."]
        text = build_prompt(CAPTION_PROMPT) + " " + " ".join(rng.sample(c, min(2, len(c))))
        enc = lm.processor(images=[open_image(s.image_path)], text=[text], return_tensors="pt")
        out.append({k: (v.to(lm.device, lm.dtype) if k == "pixel_values" else v.to(lm.device))
                    for k, v in enc.items()})
    return out


def text_sequences(lm, root, n: int, seed: int = 0, target_len: int | None = None) -> list[dict]:
    """COCO captions without images, packed to `target_len` tokens per sequence."""
    samples = build_calibration_set(root, "train2014", n_images=10_000, seed=seed + 11,
                                    fallback_subset="val2014")
    pool = [c for cs in _captions_for(root, samples).values() for c in cs]
    if not pool:
        raise RuntimeError("no calibration captions found (annotations/captions_*.json)")
    rng = random.Random(seed)
    rng.shuffle(pool)
    tok = _tok(lm)
    target_len = target_len or 640
    head = f"{SYSTEM} USER: {CAPTION_PROMPT} ASSISTANT:"
    out, k = [], 0
    for _ in range(n):
        parts = []
        ids = tok(head, return_tensors="pt")["input_ids"]
        while ids.shape[1] < target_len:
            parts.append(pool[k % len(pool)])
            k += 1
            if len(parts) % 8 == 0 or k % len(pool) == 0:
                ids = tok(head + " " + " ".join(parts), return_tensors="pt")["input_ids"]
            if len(parts) > 400:
                break
        ids = tok(head + " " + " ".join(parts), return_tensors="pt")["input_ids"][:, :target_len]
        out.append({"input_ids": ids.to(lm.device),
                    "attention_mask": torch.ones_like(ids).to(lm.device)})
    return out


def wikitext(split: str = "train") -> str:
    """WikiText-2 raw, from the Hugging Face hub (parquet; needs pyarrow)."""
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as e:                                   # pragma: no cover
        raise RuntimeError("calib=wiki / validation needs: pip install pyarrow") from e
    f = hf_hub_download("Salesforce/wikitext", repo_type="dataset",
                        filename=f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet")
    return "\n\n".join(pq.read_table(f).column("text").to_pylist())


def wiki_sequences(lm, n: int, seed: int = 0, seqlen: int = 512) -> list[dict]:
    tok = _tok(lm)
    ids = tok(wikitext("train"), return_tensors="pt")["input_ids"][0]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        i = rng.randrange(0, ids.numel() - seqlen - 1)
        x = ids[i:i + seqlen][None]
        out.append({"input_ids": x.to(lm.device), "attention_mask": torch.ones_like(x).to(lm.device)})
    return out


def build(lm, root, mode: str, n: int, seed: int = 0) -> list[dict]:
    """Calibration inputs, one sequence per dict (batch size 1)."""
    if mode == "mm":
        seqs = mm_sequences(lm, root, n, seed)
    elif mode == "text":
        probe = mm_sequences(lm, root, min(n, 8), seed)       # match the mm token budget
        L = int(sum(s["input_ids"].shape[1] for s in probe) / len(probe))
        seqs = text_sequences(lm, root, n, seed, target_len=L)
    elif mode == "wiki":
        seqs = wiki_sequences(lm, n, seed)
    else:
        raise KeyError(mode)
    ntok = sum(int(s["input_ids"].shape[1]) for s in seqs)
    nimg = sum(int((s["input_ids"] == lm.image_token_id).sum()) for s in seqs)
    print(f"[ptq] calibration '{mode}': {len(seqs)} sequences, {ntok} tokens "
          f"({100 * nimg / max(ntok, 1):.0f}% image tokens)", flush=True)
    return seqs
