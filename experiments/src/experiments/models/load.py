from __future__ import annotations

import pathlib

import torch
from transformers import AutoProcessor

from ..config import Config
from .download import load_torch, resolve_device, resolve_dtype


def load_variant(cfg: Config, variant: str):
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(device)

    if variant == "fp16":
        model = load_torch(cfg.model_id, device, dtype)
        processor = AutoProcessor.from_pretrained(cfg.model_id)
        return model, processor, device, None

    bits = int(variant.strip("w"))
    if bits not in (4, 8):
        raise ValueError(f"Unsupported variant {variant!r}")
    quant_dir = cfg.checkpoints_dir / f"gptq-w{bits}"
    if not (quant_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Quantized checkpoint missing at {quant_dir}; run the quantize step first"
        )
    from transformers import GPTQConfig

    qcfg = GPTQConfig(
        bits=bits,
        group_size=cfg.gptq_group_size,
        desc_act=cfg.gptq_desc_act,
        modules_to_not_convert=["vision_tower", "multi_modal_projector"],
        disable_exllama=True,
        use_cuda_fp16=device.startswith("cuda"),
    )
    model = load_torch(str(quant_dir), device, torch.float16, quant_config=qcfg)
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    return model, processor, device, quant_dir