#!/usr/bin/env python
"""CPU checks for the Qwen adapter that do not need model weights.

    python qwen/test_qwen_adapter.py

Covers the two places where Qwen differs from LLaVA and a silent failure would poison
the numbers: the chat-template prompt (must contain an image placeholder that the
processor expands) and the yes/no token ids (SentencePiece vs BPE).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch

from phoenix import model as M
from phoenix.metrics import _answer_token_ids, yes_no_probability


class FakeTokenizer:
    unk_token_id = 0
    eos_token_id = 2
    pad_token_id = 2
    bos_token_id = None
    _ids = {"Yes": 5, "No": 6, " Yes": 7, " No": 8}

    def encode(self, s, add_special_tokens=False):
        # Unknown strings mimic a multi-token encoding, so they are not mistaken for
        # a single answer token.
        return [self._ids[s]] if s in self._ids else [9, 9]

    def convert_tokens_to_ids(self, t):
        return {"Yes": 5, "No": 6}.get(t, 0)


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = messages[0]["content"]
        if isinstance(content, str):
            return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        kinds = [c["type"] for c in content]
        text = next((c.get("text", "") for c in content if c["type"] == "text"), "")
        image = "<|vision_start|><|image_pad|><|vision_end|>" if "image" in kinds else ""
        return (f"<|im_start|>user\n{image}{text}<|im_end|>\n"
                "<|im_start|>assistant\n")


def main():
    proc = FakeProcessor()
    M._FAMILY, M._PROCESSOR = "qwen2_5_vl", proc

    p = M.build_prompt("Is there a dog in the image?")
    assert "<|vision_start|><|image_pad|><|vision_end|>" in p, p
    assert p.rstrip().endswith("assistant"), repr(p)
    assert "Is there a dog" in p

    t = M.build_text_prompt("Describe a busy street.")
    assert "<|image_pad|>" not in t, t
    assert t.rstrip().endswith("assistant"), repr(t)

    yes = _answer_token_ids(proc.tokenizer, ("Yes", "yes", "▁Yes", "▁yes", "ĠYes", "Ġyes"))
    no = _answer_token_ids(proc.tokenizer, ("No", "no", "▁No", "▁no", "ĠNo", "Ġno"))
    assert set(yes) == {5, 7}, yes
    assert set(no) == {6, 8}, no
    assert not (set(yes) & set(no))

    logits = torch.full((1, 12), -20.0)
    logits[0, 5] = 10.0          # Yes
    logits[0, 6] = 0.0           # No
    py = float(yes_no_probability(logits, proc.tokenizer))
    assert py > 0.99, py

    # LLaVA fallback still works when the adapter is not in Qwen mode.
    M._FAMILY, M._PROCESSOR = "llava", None
    llava = M.build_prompt("hello")
    assert "<image>" in llava and "USER:" in llava
    assert "<|image_pad|>" not in M.build_text_prompt("hello")

    print("qwen adapter: all checks passed")


if __name__ == "__main__":
    main()
