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


def extract_llm(base_dir: pathlib.Path, tmp_dir: pathlib.Path, device: str) -> pathlib.Path:
    """Extract the LLaVA language model (Vicuna/Llama) into a standalone llama checkpoint."""
    from transformers import LlamaConfig, LlamaForCausalLM, LlavaConfig

    cfg = LlavaConfig.from_pretrained(str(base_dir))
    llm_cfg = LlamaConfig(**cfg.text_config.to_dict())
    llm = LlamaForCausalLM(llm_cfg)
    llm.to("meta")
    prefix = "language_model."
    shards = sorted(base_dir.glob("*.safetensors")) or sorted(base_dir.glob("pytorch_model-*.bin"))
    if not shards:
        raise FileNotFoundError(f"no weight shards found in {base_dir}")
    for shard in shards:
        if shard.suffix == ".safetensors":
            from safetensors.torch import load_file

            sd = load_file(str(shard))
        else:
            sd = torch.load(shard, map_location="cpu", weights_only=True)
        filtered = {
            k[len(prefix):]: v
            for k, v in sd.items()
            if k.startswith(prefix)
        }
        llm.load_state_dict(filtered, strict=False, assign=True)
        del sd, filtered
        torch.cuda.empty_cache()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    llm.save_pretrained(str(tmp_dir), safe_serialization=True)
    del llm
    torch.cuda.empty_cache()
    return tmp_dir


def quantize_llm(
    llm_dir: pathlib.Path,
    out_dir: pathlib.Path,
    bits: int,
    calibration: list[dict],
    device: str,
    group_size: int = 128,
    damp_percent: float = 0.1,
    desc_act: bool = True,
) -> pathlib.Path:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    if not device.startswith("cuda"):
        print("[quantize] WARNING: quantizing on CPU is very slow; prefer CUDA.")

    quant_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=desc_act,
        sym=True,
    )
    model = AutoGPTQForCausalLM.from_pretrained(
        str(llm_dir), quantize_config=quant_config, torch_dtype=torch.float16, device_map=device
    )
    # transformers >= 4.43 hoisted the rotary embedding from the decoder layers up
    # to LlamaModel (`model.rotary_emb`). auto_gptq 0.7.1 predates that and never
    # relocates it, so inv_freq stays on the CPU while the calibration forward runs
    # on the GPU. Move it there and leave it -- auto_gptq does not touch modules
    # outside its own lists.
    #
    # It must NOT be added to `outside_layer_modules`: the module holds no
    # parameters, only the inv_freq buffer, and auto_gptq calls
    # get_device() -> next(module.parameters()) on every entry, which raises
    # StopIteration on a parameterless module.
    inner = getattr(getattr(model, "model", None), "model", None)
    if inner is not None and hasattr(inner, "rotary_emb"):
        inner.rotary_emb.to(device)

    model.quantize(calibration)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out_dir), use_safetensors=True)

    # auto_gptq writes gptq_model-<bits>bit-<group>g.safetensors, a name
    # transformers' from_pretrained does not recognise -- it looks only for
    # model.safetensors / pytorch_model.bin and otherwise raises
    # "no file named model.safetensors found in directory". Rename so the
    # checkpoint loads as-is.
    target = out_dir / "model.safetensors"
    if not target.exists():
        cands = sorted(out_dir.glob("*.safetensors"))
        if cands:
            cands[0].rename(target)

    return out_dir