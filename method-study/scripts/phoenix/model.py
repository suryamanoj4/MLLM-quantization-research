"""Model loading, prompt construction and visual-token localisation for LLaVA-1.5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"

# LLaVA-1.5 chat format (v1 template).
SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.")


def build_prompt(question: str, with_system: bool = True) -> str:
    s = f"{SYSTEM} " if with_system else ""
    return f"{s}USER: <image>\n{question} ASSISTANT:"


CAPTION_PROMPT = "Please describe this image in detail."
POPE_SUFFIX = " Please answer this question with one word."


@dataclass
class LoadedModel:
    model: nn.Module
    processor: object
    image_token_id: int
    device: torch.device
    dtype: torch.dtype

    @property
    def lm(self) -> nn.Module:
        """The decoder-only language model (LlamaModel)."""
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
    v = tok.convert_tokens_to_ids("<image>")
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
    (the Hugging Face `load_in_4bit` path) instead of FP16 weights."""
    from .env import require_cuda
    require_cuda(device)             # fail in ~1 s, not after loading 14 GB of weights

    from transformers import AutoProcessor
    try:
        from transformers import LlavaForConditionalGeneration as _Cls
    except ImportError:                                       # pragma: no cover
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
            keep.append("vision_tower")
        if "projector" not in bnb_4bit:
            keep.append("multi_modal_projector")
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

    return LoadedModel(model=model, processor=processor,
                       image_token_id=get_image_token_id(model, processor),
                       device=torch.device(device), dtype=dtype)


# --------------------------------------------------------------------------- #
# visual token localisation
# --------------------------------------------------------------------------- #
def visual_mask(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Boolean [B, T] mask, True at visual-token positions.

    With transformers >= 4.47 the LLaVA processor expands `<image>` into
    `num_image_tokens` (576 for CLIP-L/14-336 with patch 14) copies, so the mask is
    exact and needs no offset arithmetic.
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
    prompts = [build_prompt(q) for q in questions]
    inputs = lm.processor(images=list(images), text=prompts,
                          return_tensors="pt", padding=True,
                          **({"truncation": True, "max_length": max_length} if max_length else {}))
    return {k: (v.to(lm.device, lm.dtype) if k == "pixel_values" else v.to(lm.device))
            for k, v in inputs.items()}
