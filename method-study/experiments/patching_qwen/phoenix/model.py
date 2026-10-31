"""Model loading, prompt construction and visual-token localisation.

Two model families are supported:
  llava  LLaVA-1.5 (LlavaForConditionalGeneration), FP16, v1 chat template, 576 image tokens
  qwen   Qwen2.5-VL (Qwen2_5_VLForConditionalGeneration), BF16, ChatML template,
         dynamic-resolution image tokens (<|image_pad|>, id from config), M-RoPE

The family is detected from the checkpoint config, so `--model-id` can be a hub id or a
local directory (e.g. a teammate's snapshot: .../models--Qwen--Qwen2.5-VL-7B-Instruct/
snapshots/<hash>). The detected family is also recorded module-wide so that
`build_prompt(q)` produces the right template for the model that was loaded last.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"

# LLaVA-1.5 chat format (v1 template).
SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.")


QWEN_SYSTEM = "You are a helpful assistant."

# family of the most recently loaded model; build_prompt() follows it
_CURRENT = {"family": "llava"}


def detect_family(model_id_or_config) -> str:
    """'qwen' for Qwen2-VL / Qwen2.5-VL checkpoints, else 'llava'."""
    cfg = model_id_or_config
    if isinstance(cfg, str):
        if "qwen" in cfg.lower():
            return "qwen"
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(cfg)
        except Exception:                                    # pragma: no cover
            return "llava"
    mt = str(getattr(cfg, "model_type", "")).lower()
    return "qwen" if mt.startswith("qwen2") else "llava"


def model_family() -> str:
    """Normalize the internal family name for the shared metrics/probe helpers."""
    return "qwen2_5_vl" if _CURRENT["family"] == "qwen" else "llava"


def build_prompt(question: str, with_system: bool = True, family: str | None = None) -> str:
    fam = family or _CURRENT["family"]
    if fam == "qwen":
        # Qwen2.5-VL's own chat template for one image + one user turn
        sys_ = f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n" if with_system else ""
        return (f"{sys_}<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}<|im_end|>\n<|im_start|>assistant\n")
    s = f"{SYSTEM} " if with_system else ""
    return f"{s}USER: <image>\n{question} ASSISTANT:"


def build_text_prompt(question: str, with_system: bool = False) -> str:
    """Text-only prompt (no image token), used by shared probes/calibration code."""
    fam = _CURRENT["family"]
    if fam == "qwen":
        sys_ = f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n" if with_system else ""
        return f"{sys_}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    s = f"{SYSTEM} " if with_system else ""
    return f"{s}USER: {question} ASSISTANT:"


CAPTION_PROMPT = "Please describe this image in detail."
POPE_SUFFIX = " Please answer this question with one word."


@dataclass
class LoadedModel:
    model: nn.Module
    processor: object
    image_token_id: int
    device: torch.device
    dtype: torch.dtype
    family: str = "llava"

    @property
    def eos_ids(self) -> list[int]:
        """Every id that ends a turn (Qwen: <|im_end|> and <|endoftext|>)."""
        ids = []
        gc = getattr(self.model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None) if gc is not None else None
        ids += list(e) if isinstance(e, (list, tuple)) else ([e] if isinstance(e, int) else [])
        t = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        if isinstance(t, int):
            ids.append(t)
        return sorted(set(ids))

    @property
    def lm(self) -> nn.Module:
        """The decoder-only language model (LlamaModel / Qwen2_5_VLTextModel)."""
        return get_language_model(self.model)

    @property
    def layers(self) -> nn.ModuleList:
        return self.lm.layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def get_language_model(model: nn.Module) -> nn.Module:
    """Return the LlamaModel (the thing that owns `.layers`), across transformers versions."""
    cands = []
    base = getattr(model, "model", None)
    if base is not None:
        cands += [getattr(base, "language_model", None), base]
    cands += [getattr(model, "language_model", None), model]
    for c in cands:
        if c is None:
            continue
        inner = getattr(c, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
        if hasattr(c, "layers"):
            return c
    raise RuntimeError("could not locate the decoder stack (.layers) on this model")


def get_image_token_id(model: nn.Module, processor) -> int:
    cfg = model.config
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            return v
    tok = getattr(processor, "tokenizer", processor)
    token = "<|image_pad|>" if _CURRENT["family"] == "qwen" else "<image>"
    v = tok.convert_tokens_to_ids(token)
    if v is None or v < 0:
        raise RuntimeError("cannot determine the image token id")
    return v


def load_model(model_id: str = DEFAULT_MODEL,
               dtype: torch.dtype = torch.float16,
               device: str = "cuda",
               attn_implementation: str = "sdpa",
               low_cpu_mem: bool = True,
               bnb_4bit: tuple | None = None) -> LoadedModel:
    """bnb_4bit=("language",) loads real bitsandbytes NF4 kernels for those parts
    (the Hugging Face `load_in_4bit` path) instead of FP16 weights.

    Qwen2.5-VL is always run in BF16 (FP16 overflows in its MLPs); the image
    resolution budget can be capped with PHOENIX_QWEN_MAX_PIXELS (default: the
    processor's own, which keeps a 640x480 COCO image at ~390 image tokens).
    """
    import os
    from .env import require_cuda
    require_cuda(device)             # fail in ~1 s, not after loading 14 GB of weights

    from transformers import AutoProcessor
    family = detect_family(model_id)
    if family == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration as _Cls
        if dtype == torch.float16:
            print("[model] Qwen2.5-VL: using bfloat16 instead of float16")
            dtype = torch.bfloat16
        pkw = {}
        if os.environ.get("PHOENIX_QWEN_MAX_PIXELS"):
            pkw["max_pixels"] = int(os.environ["PHOENIX_QWEN_MAX_PIXELS"])
        processor = AutoProcessor.from_pretrained(model_id, **pkw)
    else:
        try:
            from transformers import LlavaForConditionalGeneration as _Cls
        except ImportError:                                   # pragma: no cover
            from transformers import AutoModelForVision2Seq as _Cls
        processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(attn_implementation=attn_implementation, low_cpu_mem_usage=low_cpu_mem)
    if bnb_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise RuntimeError("bnb-nf4 needs bitsandbytes: pip install bitsandbytes") from e
        keep = ["lm_head"]
        if "vision" not in bnb_4bit:
            keep += ["vision_tower", "visual"]
        if "projector" not in bnb_4bit:
            keep += ["multi_modal_projector", "merger"]
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=False, llm_int8_skip_modules=keep)
        kw["device_map"] = {"": device}
    # transformers renamed torch_dtype -> dtype in 5.x
    try:
        model = _Cls.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError:
        model = _Cls.from_pretrained(model_id, torch_dtype=dtype, **kw)
    model.eval().requires_grad_(False)
    if not bnb_4bit:
        model.to(device)

    # deterministic padding for batched prefill
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    tok.padding_side = "left"

    _CURRENT["family"] = family
    return LoadedModel(model=model, processor=processor,
                       image_token_id=get_image_token_id(model, processor),
                       device=torch.device(device), dtype=dtype, family=family)


# --------------------------------------------------------------------------- #
# visual token localisation
# --------------------------------------------------------------------------- #
def visual_mask(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Boolean [B, T] mask, True at visual-token positions.

    With transformers >= 4.47 the LLaVA processor expands `<image>` into
    `num_image_tokens` (576 for CLIP-L/14-336 with patch 14) copies, so the mask is
    exact and needs no offset arithmetic. The Qwen2.5-VL processor likewise expands
    <|image_pad|> to one token per merged 28x28 patch.
    """
    m = input_ids == image_token_id
    if not bool(m.any()):
        raise RuntimeError(
            "no image tokens found in input_ids. Your processor is not expanding "
            "<image>; upgrade transformers or set processor.patch_size / "
            "processor.vision_feature_select_strategy from the model config."
        )
    return m


def visual_spans(mask_row: torch.Tensor) -> list[tuple[int, int]]:
    """Contiguous [start, end) spans of True in a 1-D bool tensor."""
    idx = torch.nonzero(mask_row, as_tuple=False).flatten().tolist()
    if not idx:
        return []
    spans, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != p + 1:
            spans.append((s, p + 1))
            s = i
        p = i
    spans.append((s, p + 1))
    return spans


def prepare_batch(lm: LoadedModel, images: Sequence, questions: Sequence[str],
                  max_length: int | None = None) -> dict:
    """Tokenise a batch of (image, question) pairs into model inputs."""
    prompts = [build_prompt(q, family=getattr(lm, "family", None)) for q in questions]
    inputs = lm.processor(images=list(images), text=prompts,
                          return_tensors="pt", padding=True,
                          **({"truncation": True, "max_length": max_length} if max_length else {}))
    return {k: (v.to(lm.device, lm.dtype) if k == "pixel_values" else v.to(lm.device))
            for k, v in inputs.items()}
