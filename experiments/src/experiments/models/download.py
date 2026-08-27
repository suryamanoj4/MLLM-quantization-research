from __future__ import annotations

import pathlib

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor


def download_checkpoint(model_id: str, local_dir: pathlib.Path) -> pathlib.Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(local_dir))
    return local_dir


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def load_torch(model_id: str, device: str, dtype: torch.dtype, quant_config=None) -> torch.nn.Module:
    from transformers import LlavaForConditionalGeneration

    kwargs = dict(torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["torch_dtype"] = torch.float16
    return LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)