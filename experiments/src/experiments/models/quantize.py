from __future__ import annotations

import json
import pathlib
import random

import torch


def _tokenize_texts(tokenizer, texts: list[str], seq_len: int) -> list[dict]:
    examples = []
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=seq_len).input_ids
        if len(ids) < 32:
            continue
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return examples


def build_calibration(
    tokenizer,
    source: str,
    captions_train_path: pathlib.Path,
    n_samples: int,
    seq_len: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    if source == "coco_train_captions":
        with open(captions_train_path) as f:
            ann = json.load(f)
        texts = [c["caption"] for c in ann["annotations"]]
        rng.shuffle(texts)
        return _tokenize_texts(tokenizer, texts[: max(n_samples * 2, 256)], seq_len)
    raise ValueError(
        f"Unknown calibration source {source!r}; use 'coco_train_captions' (needs captions_train2014.json)"
    )


def quantize_auto_gptq(
    model_id: str,
    out_dir: pathlib.Path,
    bits: int,
    group_size: int,
    damp_percent: float,
    desc_act: bool,
    calibration: list[dict],
    device: str,
) -> pathlib.Path:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    quant_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=desc_act,
        sym=True,
    )
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id, quantize_config=quant_config, torch_dtype=torch.float16, device_map=device
    )
    model.quantize(calibration)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out_dir), safetensors=True, use_safetensors=True)
    return out_dir


def quantize(
    model_id: str,
    out_dir: pathlib.Path,
    bits: int,
    backend: str,
    calibration: list[dict],
    device: str,
    group_size: int = 128,
    damp_percent: float = 0.1,
    desc_act: bool = True,
) -> pathlib.Path:
    if device.startswith("cuda"):
        pass
    else:
        print(f"[quantize] WARNING: quantizing on CPU is very slow; prefer CUDA.")
    if backend == "auto_gptq":
        return quantize_auto_gptq(
            model_id, out_dir, bits, group_size, damp_percent, desc_act, calibration, device
        )
    raise ValueError(f"Unknown gptq_backend {backend!r} (supported: auto_gptq)")