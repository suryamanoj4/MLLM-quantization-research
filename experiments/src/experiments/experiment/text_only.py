from __future__ import annotations

import pathlib

import torch

from ..config import Config
from ..data.datasets import base_seed_for
from ..data.prompts import chat_prompt


def _yes_no_ids(tokenizer):
    yes = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no = tokenizer("no", add_special_tokens=False).input_ids[0]
    return yes, no


def probe_prior_pope(model, processor, tokenizer, questions: list[dict], cfg: Config) -> list[dict]:
    yes_id, no_id = _yes_no_ids(tokenizer)
    records = []
    for q in questions:
        prompt = chat_prompt(q["text"], with_image=False)
        seed = base_seed_for(q["image"], cfg.seed)
        torch.manual_seed(seed)
        inputs = processor(text=prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[0, -1].float()
        p = torch.softmax(logits, dim=-1)
        p_yes = float(p[yes_id])
        p_no = float(p[no_id])
        records.append(
            {
                "image": q["image"],
                "question": q["text"],
                "p_yes": p_yes,
                "p_no": p_no,
                "prior": "yes" if p_yes >= p_no else "no",
            }
        )
    return records


def _kl(p_img: torch.Tensor, p_txt: torch.Tensor) -> float:
    eps = 1e-12
    return float((p_img * (torch.log(p_img + eps) - torch.log(p_txt + eps))).sum())


def run_contrastive_dkl(
    model,
    processor,
    tokenizer,
    items: list[dict],
    cfg: Config,
    max_new_tokens: int,
    image_dir: pathlib.Path | None = None,
) -> list[float]:
    per_item_kl: list[float] = []
    for item in items:
        text = item["text"]
        image = None
        if item.get("image_name") and image_dir is not None:
            from PIL import Image

            image = Image.open(image_dir / item["image_name"]).convert("RGB")
        seed = base_seed_for(item["image_name"], cfg.seed)
        torch.manual_seed(seed)
        kls = _lockstep_kl(model, processor, tokenizer, text, image, cfg, max_new_tokens)
        per_item_kl.append(sum(kls) / len(kls) if kls else 0.0)
    return per_item_kl


def _lockstep_kl(model, processor, tokenizer, text: str, image, cfg: Config, max_new_tokens: int) -> list[float]:
    def build(with_image: bool):
        prompt = chat_prompt(text, with_image=with_image)
        if with_image:
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        else:
            inputs = processor(text=prompt, return_tensors="pt")
        return inputs.to(next(model.parameters()).device)

    img_in = build(True)
    txt_in = build(False)
    img_ids, txt_ids = img_in["input_ids"], txt_in["input_ids"]
    img_mask = torch.ones_like(img_ids)
    txt_mask = torch.ones_like(txt_ids)
    img_cache = txt_cache = None
    kls: list[float] = []
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            if img_cache is None:
                oi = model(input_ids=img_ids, attention_mask=img_mask, pixel_values=img_in.get("pixel_values"), use_cache=True)
            else:
                oi = model(input_ids=img_ids, attention_mask=img_mask, past_key_values=img_cache, use_cache=True)
            if txt_cache is None:
                ot = model(input_ids=txt_ids, attention_mask=txt_mask, use_cache=True)
            else:
                ot = model(input_ids=txt_ids, attention_mask=txt_mask, past_key_values=txt_cache, use_cache=True)
        img_cache, txt_cache = oi.past_key_values, ot.past_key_values
        p_img = torch.softmax(oi.logits[0, -1].float(), dim=-1)
        p_txt = torch.softmax(ot.logits[0, -1].float(), dim=-1)
        kls.append(_kl(p_img, p_txt))
        nxt = torch.argmax(p_img)
        img_ids = nxt.unsqueeze(0).unsqueeze(0)
        txt_ids = torch.argmax(p_txt).unsqueeze(0).unsqueeze(0)
        img_mask = torch.cat([img_mask, torch.ones(1, 1, device=img_mask.device)], dim=-1)
        txt_mask = torch.cat([txt_mask, torch.ones(1, 1, device=txt_mask.device)], dim=-1)
        if int(nxt) == eos:
            break
    return kls