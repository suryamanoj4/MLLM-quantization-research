"""Small shared helpers: seeding, IO, timing, logging."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


@contextmanager
def timer(name: str, out: dict | None = None, sync_cuda: bool = True):
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if out is not None:
        out[name] = dt
    else:
        print(f"[timer] {name}: {dt:.3f}s")


def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def human_env() -> dict:
    import transformers

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return info


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32}[name]


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}
