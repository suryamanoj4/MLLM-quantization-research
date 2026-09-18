from __future__ import annotations

import torch
from transformers import AutoProcessor

from ..config import Config
from .download import load_torch, resolve_device, resolve_dtype

# Submodule name fragments (matched as substrings against dotted parameter
# names by both TorchAoConfig and QuantoConfig) that must stay unquantized.
# The study's isolated variable is the LLM decoder (H1-H4 are about the
# decoder falling back to language priors); quantizing the CLIP tower or the
# projector would confound that with a vision-side failure. GPTQ already
# carves the LLM out on its own (extract_llm), so this mirrors that for the
# two new layer-swap quantizers.
NOT_QUANTIZE_MODULES = ["vision_tower", "multi_modal_projector"]


def _dmap(device: str) -> str:
    return "auto" if device.startswith("cuda") else device


def _warn_if_vision_tower_quantized(model) -> None:
    """Vision-tower carve-out is a config request, not a guarantee -- confirm it held.

    If this fires, the run produced a *different* finding (vision-side collapse,
    not decoder collapse) and must be logged separately rather than folded into
    the H1-H4 numbers. Detection is best-effort (quantized weights show up as a
    non-plain-Tensor type, whether that's quanto's QBytesTensor or torchao's
    AffineQuantizedTensor) -- treat this as a tripwire, not a proof; confirm by
    eye in a real run if it fires.
    """
    hit = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not any(key in name for key in NOT_QUANTIZE_MODULES):
            continue
        weight = getattr(mod, "weight", None)
        if weight is not None and type(weight) is not torch.nn.Parameter and type(weight) is not torch.Tensor:
            hit.append(name)
    if hit:
        print(
            "[load] WARNING: vision-tower carve-out did not hold, these modules "
            f"look quantized despite the exclusion list: {hit}"
        )


def _load_fp16(cfg: Config, device: str, dtype: torch.dtype):
    return load_torch(cfg.model_id, device, dtype, device_map=_dmap(device))


def _load_w8a8(cfg: Config, device: str):
    from transformers import TorchAoConfig

    qcfg = TorchAoConfig(
        "int8_dynamic_activation_int8_weight",
        modules_to_not_convert=NOT_QUANTIZE_MODULES,
    )
    model = load_torch(cfg.model_id, device, torch.float16, quant_config=qcfg, device_map=_dmap(device))
    _warn_if_vision_tower_quantized(model)
    return model


def _load_w4a16(cfg: Config, device: str, dtype: torch.dtype):
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
        use_exllama=False,
        use_cuda_fp16=device.startswith("cuda"),
    )
    qllm = LlamaForCausalLM.from_pretrained(
        str(quant_dir),
        quantization_config=qcfg,
        torch_dtype=torch.float16,
        device_map=device,
    )
    base.language_model = qllm
    return base, quant_dir


def _load_quanto(cfg: Config, device: str, activations: str | None):
    from transformers import QuantoConfig

    qcfg = QuantoConfig(
        weights="int4",
        activations=activations,
        modules_to_not_convert=NOT_QUANTIZE_MODULES,
    )
    model = load_torch(cfg.model_id, device, torch.float16, quant_config=qcfg, device_map=_dmap(device))
    _warn_if_vision_tower_quantized(model)
    return model


def _fake_quantize_int4(x: torch.Tensor) -> torch.Tensor:
    """Simulate int4 activation quantization: round-trip through 4-bit symmetric
    per-token levels, then dequantize back to fp16.

    optimum-quanto (and transformers' QuantoConfig wrapper) hard-reject real
    int4 *activation* quantization -- ActivationQBytesTensor only supports
    8-bit qtypes. This numerically simulates the W4A4 collapse regime the same
    way the fake-quant fallback already documented for AWQ/MQuant on
    non-CUDA-kernel hardware does (see Research Ideation.md), rather than
    claiming a real packed-int4 GEMM kernel that doesn't exist for this stack.
    """
    qmax = 7  # signed 4-bit range: [-8, 7]
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


def _attach_int4_activation_fakequant(module: torch.nn.Module) -> None:
    def _pre_hook(mod, args):
        if not args:
            return args
        x, *rest = args
        return (_fake_quantize_int4(x), *rest)

    for _, sub in module.named_modules():
        if isinstance(sub, torch.nn.Linear):
            sub.register_forward_pre_hook(_pre_hook)


def load_variant(cfg: Config, variant: str):
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(device)
    processor = AutoProcessor.from_pretrained(cfg.model_id)

    if variant == "fp16":
        model = _load_fp16(cfg, device, dtype)
        return model, processor, device, None

    if variant == "w8a8":
        model = _load_w8a8(cfg, device)
        return model, processor, device, None

    if variant == "w4a16":
        model, quant_dir = _load_w4a16(cfg, device, dtype)
        return model, processor, device, quant_dir

    if variant == "w4a8":
        model = _load_quanto(cfg, device, activations="int8")
        return model, processor, device, None

    if variant == "w4a4":
        model = _load_quanto(cfg, device, activations=None)
        _attach_int4_activation_fakequant(model.language_model)
        return model, processor, device, None

    raise ValueError(
        f"Unsupported variant {variant!r}; expected one of fp16, w8a8, w4a16, w4a8, w4a4"
    )
