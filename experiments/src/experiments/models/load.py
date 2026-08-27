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

    if variant == "w8":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = load_torch(cfg.model_id, device, torch.float16, quant_config=bnb)
        processor = AutoProcessor.from_pretrained(cfg.model_id)
        return model, processor, device, None

    bits = int(variant.strip("w"))
    if bits != 4:
        raise ValueError(f"Unsupported variant {variant!r}")
    quant_dir = cfg.checkpoints_dir / "gptq-llm-w4"
    if not (quant_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Quantized checkpoint missing at {quant_dir}; run the quantize step first"
        )
    from transformers import GPTQConfig, LlamaForCausalLM

    base = load_torch(cfg.model_id, device, dtype)
    base.language_model.to("cpu")
    torch.cuda.empty_cache()

    qcfg = GPTQConfig(
        bits=4,
        group_size=cfg.gptq_group_size,
        desc_act=cfg.gptq_desc_act,
        disable_exllama=True,
        use_cuda_fp16=device.startswith("cuda"),
    )
    qllm = LlamaForCausalLM.from_pretrained(
        str(quant_dir),
        quantization_config=qcfg,
        torch_dtype=torch.float16,
        device_map=device,
    )
    base.language_model = qllm
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    return base, processor, device, quant_dir