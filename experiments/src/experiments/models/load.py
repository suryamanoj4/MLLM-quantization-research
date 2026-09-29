"""Model loading for the five-rung precision ladder.

Rungs and how each is realised on this stack (Kaggle T4, sm75, torch 2.5,
transformers 4.46):

    fp16   plain load
    w8a8   int8 per-channel RTN weights + int8 per-token activation fake-quant
    w4a16  GPTQ int4 weights, dequantized once to fp16
    w4a8   the same GPTQ int4 weights + int8 per-token activation fake-quant
    w4a4   the same GPTQ int4 weights + int4 per-token activation fake-quant

Every quantized rung is simulated: weights are rounded to their integer grid
and stored dequantized in fp16, activations are rounded per token on the way
into each decoder Linear. The numbers are those of the integer scheme; only
the kernel that multiplies them is fp16.

Why W8A8 is not run through TorchAO: torchao's safe_int_mm sends any int8
matmul with 16 or fewer rows to a CPU int32 fallback, copying the activation
and the whole int8 weight to the host. Autoregressive decoding is one row per
step, so every decode step of every decoder layer ran on the CPU (GPU at ~4%
utilisation, minutes per POPE question). Only prefill, with ~590 rows, stayed
on cuBLAS -- which is why a single-forward preflight passed.

Why the three W4 rungs are simulated rather than run on int4 kernels: no int4
kernel runs on a T4 with this stack. auto_gptq 0.7.1's CUDA extension does not
build against torch 2.5, so its QuantLinear falls back to a pure-PyTorch
unpack + dequant + matmul on every forward (the first W4A16 run did not finish
100 images in four hours); optimum-quanto's int4 kernels (marlin) need sm80 and
fail to compile; and transformers' QuantoConfig rejects activation
quantization outright. A W4A16 kernel dequantizes int4 to fp16 and runs an
fp16 GEMM anyway, so dequantizing the GPTQ weights once up front yields the
same weights the kernel would multiply by, at fp16 speed.

Sharing one set of GPTQ weights across w4a16 / w4a8 / w4a4 also makes the
activation axis exact: those three rungs differ only in activation bits, so
W4A16 -> W4A8 -> W4A4 isolates activation rounding at genuinely fixed weights.
W8A8 -> W4A8 holds activations at int8 and changes only the weights (int8 RTN
to int4 GPTQ -- the method differs as well as the bit width).

Scope is identical on every rung: the 224 Linear layers of the 32 decoder
layers. The CLIP vision tower, the multimodal projector, the embeddings and
lm_head stay fp16, so the isolated variable is the LLM decoder (H1-H4 are
about the decoder falling back to language priors).
"""

from __future__ import annotations

import gc

import torch
from transformers import AutoProcessor

from ..config import Config
from .download import load_torch, resolve_device, resolve_dtype

# GPTQ quantizes these seven linears in every Llama decoder layer.
_GPTQ_LINEARS_PER_LAYER = 7


def _dmap(device: str) -> str:
    return "auto" if device.startswith("cuda") else device


def _single_gpu_dmap(device: str) -> str:
    """Pin to one GPU instead of accelerate's multi-GPU "auto" split.

    Used for the temporary GPTQ model, whose packed QuantLinear buffers
    (qweight / qzeros / scales / g_idx) must stay together on one device.
    """
    return "cuda:0" if device.startswith("cuda") else device


def _load_fp16(cfg: Config, device: str, dtype: torch.dtype):
    return load_torch(cfg.model_id, device, dtype, device_map=_dmap(device))


def _copy_gptq_weights_dequantized(qllm, target_lm) -> tuple[int, float]:
    """Write each GPTQ QuantLinear's dequantized weight into the matching fp16 Linear.

    The weight is read out by driving the QuantLinear with an identity matrix:
    for y = x W^T, x = I returns W^T exactly (each output element is a single
    product with 1.0). That goes through auto_gptq's own dequant path, g_idx /
    desc_act included, so the packing format is never re-implemented here.

    Every copy is then checked against the QuantLinear itself through the
    module it was written into, which also catches a wrong name mapping.
    Returns (modules copied, worst relative error).
    """
    n, worst = 0, 0.0
    with torch.no_grad():
        for name, qmod in qllm.named_modules():
            if not hasattr(qmod, "qweight"):
                continue
            target = target_lm.get_submodule(name)
            in_f = getattr(qmod, "infeatures", None) or target.in_features
            dev = qmod.qweight.device

            eye = torch.eye(in_f, dtype=torch.float16, device=dev)
            wt = qmod(eye)
            bias = getattr(qmod, "bias", None)
            if bias is not None:
                wt = wt - bias
            w = wt.t().contiguous()
            if w.shape != target.weight.shape:
                raise RuntimeError(f"{name}: dequantized {tuple(w.shape)} != fp16 {tuple(target.weight.shape)}")
            target.weight.copy_(w.to(device=target.weight.device, dtype=target.weight.dtype))

            # local generator: leave the global RNG (per-image decoding seeds) untouched
            g = torch.Generator(device=dev).manual_seed(n)
            x = torch.randn(4, in_f, dtype=torch.float16, device=dev, generator=g)
            ref = qmod(x).float()
            if bias is not None:
                ref = ref - bias.float()
            got = (x.to(target.weight.device) @ target.weight.t()).float().to(dev)
            err = ((ref - got).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
            worst = max(worst, err)

            n += 1
            del eye, wt, w, x, ref, got
    return n, worst


def _load_w4_gptq(cfg: Config, device: str, dtype: torch.dtype):
    """fp16 LLaVA whose decoder linears carry the GPTQ int4 weights, dequantized."""
    quant_dir = cfg.checkpoints_dir / "gptq-llm-w4"
    if not (quant_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Quantized checkpoint missing at {quant_dir}; run the quantize step first"
        )
    from transformers import GPTQConfig, LlamaForCausalLM

    base = _load_fp16(cfg, device, dtype)
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
        device_map=_single_gpu_dmap(device),
    )
    n, worst = _copy_gptq_weights_dequantized(qllm, base.language_model)
    # del alone does not free it: the GPTQ modules sit in reference cycles, so
    # without a collection its ~4 GB stays on the GPU for the rest of the run
    del qllm
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    expected = _GPTQ_LINEARS_PER_LAYER * base.language_model.config.num_hidden_layers
    print(
        f"[load] GPTQ int4 weights dequantized into {n}/{expected} decoder linears; "
        f"worst rel. err vs QuantLinear = {worst:.2e}"
    )
    if n != expected:
        raise RuntimeError(f"expected {expected} GPTQ linears, copied {n}")
    if worst > 1e-2:
        raise RuntimeError(f"dequantized weights disagree with the QuantLinear outputs (rel err {worst:.2e})")
    return base, quant_dir


def _round_weights_int8_per_channel(model) -> int:
    """Round every decoder-layer Linear weight to int8 in place, kept as fp16.

    Per-output-channel symmetric round-to-nearest -- the weight scheme of
    TorchAO's int8_dynamic_activation_int8_weight and of LLM.int8/SmoothQuant
    style W8A8. No calibration.
    """
    n = 0
    with torch.no_grad():
        for sub in model.language_model.model.layers.modules():
            if isinstance(sub, torch.nn.Linear):
                w = sub.weight.float()
                scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
                q = torch.clamp(torch.round(w / scale), -128, 127)
                sub.weight.copy_((q * scale).to(sub.weight.dtype))
                n += 1
                del w, scale, q
    return n


def _fake_quantize_activations(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token symmetric dynamic activation quantization, round-tripped to x.dtype.

    Per-token absmax scale, symmetric, round-to-nearest -- the activation
    scheme of TorchAO's int8_dynamic_activation -- at `bits` bits.
    Computed in float32: in fp16 the 1e-8 floor underflows to zero and an
    all-zero row would divide 0/0 into NaN.
    """
    qmax = 2 ** (bits - 1) - 1  # 127 for int8, 7 for int4
    xf = x.float()
    scale = xf.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(xf / scale), -qmax - 1, qmax)
    return (q * scale).to(x.dtype)


def _attach_activation_fakequant(model, bits: int) -> int:
    """Quantize the input of every decoder-layer Linear. lm_head stays fp16, as
    it does under GPTQ, so all rungs quantize the same scope."""

    def _pre_hook(mod, args):
        if not args:
            return args
        x, *rest = args
        return (_fake_quantize_activations(x, bits), *rest)

    n = 0
    for sub in model.language_model.model.layers.modules():
        if isinstance(sub, torch.nn.Linear):
            sub.register_forward_pre_hook(_pre_hook)
            n += 1
    return n


_W4_ACTIVATION_BITS = {"w4a16": None, "w4a8": 8, "w4a4": 4}


def load_variant(cfg: Config, variant: str):
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(device)
    processor = AutoProcessor.from_pretrained(cfg.model_id)

    if variant == "fp16":
        model = _load_fp16(cfg, device, dtype)
        return model, processor, device, None

    if variant == "w8a8":
        model = _load_fp16(cfg, device, dtype)
        expected = _GPTQ_LINEARS_PER_LAYER * model.language_model.config.num_hidden_layers
        nw = _round_weights_int8_per_channel(model)
        na = _attach_activation_fakequant(model, 8)
        print(f"[load] w8a8: int8 per-channel weights on {nw}/{expected}, "
              f"int8 per-token activation fake-quant on {na}/{expected} decoder linears")
        if nw != expected or na != expected:
            raise RuntimeError(f"expected {expected} decoder linears, got weights={nw} activations={na}")
        return model, processor, device, None

    if variant in _W4_ACTIVATION_BITS:
        model, quant_dir = _load_w4_gptq(cfg, device, dtype)
        bits = _W4_ACTIVATION_BITS[variant]
        if bits is not None:
            n = _attach_activation_fakequant(model, bits)
            expected = _GPTQ_LINEARS_PER_LAYER * model.language_model.config.num_hidden_layers
            print(f"[load] {variant}: int{bits} per-token activation fake-quant on {n}/{expected} decoder linears")
            if n != expected:
                raise RuntimeError(f"expected {expected} hooked linears, got {n}")
        return model, processor, device, quant_dir

    raise ValueError(
        f"Unsupported variant {variant!r}; expected one of fp16, w8a8, w4a16, w4a8, w4a4"
    )
