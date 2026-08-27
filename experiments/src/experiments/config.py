from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml


@dataclasses.dataclass
class Config:
    model_id: str = "liuhaotian/llava-v1.5-7b"
    variants: list[str] = dataclasses.field(default_factory=lambda: ["fp16", "w8", "w4"])
    sample_images: int | None = None
    seed: int = 42
    device: str = "auto"

    data_dir: pathlib.Path = pathlib.Path("data")
    checkpoints_dir: pathlib.Path = pathlib.Path("checkpoints")
    output_dir: pathlib.Path = pathlib.Path("results")

    calibration_samples: int = 128
    calibration_source: str = "coco_train_captions"
    gptq_group_size: int = 128
    gptq_backend: str = "auto"
    gptq_damp_percent: float = 0.1
    gptq_desc_act: bool = True
    gptq_seq_len: int = 2048
    keep_base_checkpoint: bool = True

    max_new_tokens_pope: int = 8
    max_new_tokens_chair: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int | None = None
    do_sample: bool = True

    capture_attention: bool = True
    store_full_rows: bool = False
    run_probe: bool = True
    probe_chair_subset: int = 100
    bootstrap_iters: int = 2000
    download_workers: int = 8

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def apply_overrides(self, **kw: Any) -> "Config":
        for k, v in kw.items():
            if v is None or k not in self.__dataclass_fields__:
                continue
            setattr(self, k, v)
        return self

    def resolve_paths(self, root: pathlib.Path) -> "Config":
        for f in ("data_dir", "checkpoints_dir", "output_dir"):
            p = getattr(self, f)
            setattr(self, f, p if p.is_absolute() else root / p)
        return self