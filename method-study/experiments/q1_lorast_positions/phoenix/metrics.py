"""POPE scoring, fluency guards and calibration error."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import torch


def parse_yes_no(text: str) -> str | None:
    t = text.strip().lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    words = t.split()
    for w in words[:6]:
        if w in ("yes", "yeah", "yep"):
            return "yes"
        if w in ("no", "nope", "not"):
            return "no"
    if "yes" in words:
        return "yes"
    if "no" in words:
        return "no"
    return None


def pope_metrics(preds: Sequence[str | None], labels: Sequence[str]) -> dict:
    """POPE convention: 'yes' is the positive class."""
    tp = fp = tn = fn = unk = 0
    for p, l in zip(preds, labels):
        if p is None:
            unk += 1
            p = "no"                      # POPE convention: unparseable counts as negative
        if p == "yes" and l == "yes":
            tp += 1
        elif p == "yes" and l == "no":
            fp += 1
        elif p == "no" and l == "no":
            tn += 1
        else:
            fn += 1
    n = max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"accuracy": (tp + tn) / n, "precision": prec, "recall": rec, "f1": f1,
            "yes_ratio": (tp + fp) / n, "unparsed": unk / n,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n}


def expected_calibration_error(probs: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 10) -> float:
    if not probs:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for p, c in zip(probs, correct):
        b = min(n_bins - 1, int(p * n_bins))
        bins[b].append((p, c))
    ece, n = 0.0, len(probs)
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(1 for _, c in b if c) / len(b)
        ece += (len(b) / n) * abs(conf - acc)
    return ece


_LLAVA_YES = ("▁Yes", "Yes", "▁yes", "yes")
_LLAVA_NO = ("▁No", "No", "▁no", "no")
_QWEN_YES = ("Yes", "yes", "▁Yes", "▁yes", "ĠYes", "Ġyes")
_QWEN_NO = ("No", "no", "▁No", "▁no", "ĠNo", "Ġno")


def _answer_token_ids(tokenizer, words: Sequence[str]) -> list[int]:
    """Single-token ids for yes/no in either SentencePiece (LLaVA) or BPE (Qwen)."""
    from .model import model_family
    if model_family() != "qwen2_5_vl":
        ids = [tokenizer.convert_tokens_to_ids(t) for t in words]
        return [i for i in ids if i is not None and i >= 0]

    # Qwen uses `Yes` / `No` and `ĠYes` / `ĠNo`, not SentencePiece `▁Yes`.
    # convert_tokens_to_ids returns unk for absent tokens, so encode both bare and
    # space-prefixed forms and drop unk/eos/pad.
    ids: set[int] = set()
    for w in words:
        for variant in (w, " " + w):
            try:
                enc = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                enc = []
            if len(enc) == 1:
                ids.add(int(enc[0]))
        v = tokenizer.convert_tokens_to_ids(w)
        if isinstance(v, int) and v >= 0:
            ids.add(int(v))
    bad = {getattr(tokenizer, "unk_token_id", None),
           getattr(tokenizer, "eos_token_id", None),
           getattr(tokenizer, "pad_token_id", None),
           getattr(tokenizer, "bos_token_id", None)}
    return sorted(i for i in ids if i not in bad)


def yes_no_probability(logits: torch.Tensor, tokenizer) -> torch.Tensor:
    """P(yes) / (P(yes) + P(no)) from the first generated position -- for ECE."""
    from .model import model_family
    qwen = model_family() == "qwen2_5_vl"
    ids_yes = _answer_token_ids(tokenizer, _QWEN_YES if qwen else _LLAVA_YES)
    ids_no = _answer_token_ids(tokenizer, _QWEN_NO if qwen else _LLAVA_NO)
    if not ids_yes or not ids_no:
        raise RuntimeError("could not resolve yes/no token ids for this tokenizer")
    p = torch.softmax(logits.float(), -1)
    py = p[:, ids_yes].sum(-1)
    pn = p[:, ids_no].sum(-1)
    return py / (py + pn).clamp(min=1e-9)


# --------------------------------------------------------------------------- #
# fluency / degeneration guards
# --------------------------------------------------------------------------- #
def repetition_rate(text: str, n: int = 4) -> float:
    w = text.split()
    if len(w) < n + 1:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = Counter(grams)
    return 1.0 - len(c) / max(len(grams), 1)


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    grams, tot = set(), 0
    for t in texts:
        w = t.split()
        for i in range(len(w) - n + 1):
            grams.add(tuple(w[i:i + n]))
            tot += 1
    return len(grams) / max(tot, 1)


def fluency_report(texts: Sequence[str]) -> dict:
    return {
        "avg_len": sum(len(t.split()) for t in texts) / max(len(texts), 1),
        "rep4": sum(repetition_rate(t, 4) for t in texts) / max(len(texts), 1),
        "distinct2": distinct_n(texts, 2),
        "empty_frac": sum(1 for t in texts if not t.strip()) / max(len(texts), 1),
    }


@torch.no_grad()
def caption_perplexity(lm, samples, reference_captions: dict, batch_size: int = 2,
                       max_items: int = 100) -> float:
    """Teacher-forced PPL of human reference captions. Catches fluency damage that
    CHAIR cannot see -- an intervention that shortens captions lowers CHAIR for the
    wrong reason."""
    from .data import open_image
    from .model import build_prompt, CAPTION_PROMPT
    tok = lm.processor.tokenizer
    total_nll, total_tok = 0.0, 0
    items = [s for s in samples if reference_captions.get(int(s.image_id))][:max_items]
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        imgs = [open_image(s.image_path) for s in chunk]
        refs = [reference_captions[int(s.image_id)][0].strip() for s in chunk]
        prompts = [build_prompt(CAPTION_PROMPT) + " " + r for r in refs]
        enc = lm.processor(images=imgs, text=prompts, return_tensors="pt", padding=True)
        enc = {k: (v.to(lm.device, lm.dtype) if k == "pixel_values" else v.to(lm.device))
               for k, v in enc.items()}
        from .loras import steer_mask
        # without the mask LoRAS silently does nothing here (identical PPL in every cell)
        with steer_mask(enc["input_ids"] == lm.image_token_id):
            out = lm.model(**enc, use_cache=False)
        logits = out.logits[:, :-1]
        target = enc["input_ids"][:, 1:]
        from .model import model_family
        # Qwen's BPE tokeniser can merge the leading space differently; LLaVA's
        # original count used the bare caption, so keep that path unchanged.
        n_ref = [len(tok((" " + r) if model_family() == "qwen2_5_vl" else r,
                         add_special_tokens=False)["input_ids"]) for r in refs]
        lp = torch.log_softmax(logits.float(), -1)
        for b, k in enumerate(n_ref):
            if k <= 0:
                continue
            tgt = target[b, -k:]
            nll = -lp[b, -k:].gather(-1, tgt[:, None]).squeeze(-1)
            total_nll += float(nll.sum())
            total_tok += k
    return math.exp(total_nll / max(total_tok, 1))
