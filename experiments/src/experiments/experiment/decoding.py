from __future__ import annotations

import torch

from ..config import Config


class AttentionTracker:
    def __init__(self, store_full_rows: bool = False):
        self.store_full_rows = store_full_rows
        self.span: tuple[int, int] | None = None
        self.step_mass: list[float] = []
        self.step_entropy: list[float] = []
        self.layer_mass: list[list[float]] = []
        self.layer_argmax: list[list[list[int]]] = []
        self.rows: list[list[torch.Tensor]] = []
        self.n_heads: int = 0

    def reset(self, span_start: int, span_end: int) -> None:
        self.span = (span_start, span_end)
        self.step_mass = []
        self.step_entropy = []
        self.layer_mass = []
        self.layer_argmax = []
        self.rows = []

    def update(self, attentions) -> None:
        s, e = self.span
        masses: list[float] = []
        entropies: list[float] = []
        argmaxes: list[list[int]] = []
        rows: list[torch.Tensor] = []
        for att in attentions:
            row = att[0, :, -1, :].float()
            self.n_heads = row.shape[0]
            vis = row[:, s:e]
            mass = vis.sum(dim=1)
            masses.append(mass.mean().item())
            p = vis / (vis.sum(dim=1, keepdim=True) + 1e-9)
            ent = -(p * torch.log2(p + 1e-12)).sum(dim=1)
            entropies.append(ent.mean().item())
            argmaxes.append(vis.argmax(dim=1).cpu().tolist())
            if self.store_full_rows:
                rows.append(row.cpu())
        self.layer_mass.append(masses)
        self.step_mass.append(sum(masses) / len(masses))
        self.step_entropy.append(sum(entropies) / len(entropies))
        self.layer_argmax.append(argmaxes)
        if self.store_full_rows:
            self.rows.append(rows)

    def drift_rate(self) -> float:
        if len(self.layer_argmax) < 2:
            return 0.0
        flips = 0
        total = 0
        for prev, cur in zip(self.layer_argmax[:-1], self.layer_argmax[1:]):
            for pl, cl in zip(prev, cur):
                total += len(pl)
                flips += sum(1 for a, b in zip(pl, cl) if a != b)
        return flips / max(total, 1)

    def decile_profile(self, n_bins: int = 10) -> list[float]:
        if not self.step_mass:
            return []
        n = len(self.step_mass)
        edges = [int(i * n / n_bins) for i in range(n_bins + 1)]
        profile = []
        for i in range(n_bins):
            chunk = self.step_mass[edges[i] : edges[i + 1]]
            profile.append(sum(chunk) / len(chunk) if chunk else 0.0)
        return profile

    def summary(self) -> dict:
        return {
            "mean_mass": sum(self.step_mass) / len(self.step_mass) if self.step_mass else 0.0,
            "mean_entropy": sum(self.step_entropy) / len(self.step_entropy) if self.step_entropy else 0.0,
            "drift_rate": self.drift_rate(),
            "n_steps": len(self.step_mass),
            "decile_profile": self.decile_profile(),
        }


def _nucleus(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_p, _ = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    keep = cum <= top_p
    keep[..., 0] = True
    cutoff = sorted_p[keep][-1]
    probs = probs.clone()
    probs[probs < cutoff] = 0.0
    return probs / probs.sum()


def sample_token(logits: torch.Tensor, cfg: Config, step: int) -> torch.Tensor:
    if not cfg.do_sample:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / cfg.temperature, dim=-1)
    if cfg.top_k is not None:
        k = min(cfg.top_k, probs.shape[-1])
        topk = torch.topk(probs, k).values[..., -1]
        probs = probs.clone()
        probs[probs < topk] = 0.0
        probs = probs / probs.sum()
    if cfg.top_p < 1.0:
        probs = _nucleus(probs, cfg.top_p)
    return torch.multinomial(probs, num_samples=1)


def decode(
    model,
    processor,
    tokenizer,
    text: str,
    image,
    cfg: Config,
    max_new_tokens: int,
    seed: int,
    tracker: AttentionTracker | None = None,
) -> tuple[str, list[int]]:
    torch.manual_seed(seed)
    if image is not None:
        inputs = processor(text=text, images=image, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    elif pixel_values is None and image is not None:
        pixel_values = None

    if tracker is not None:
        image_token_index = int(getattr(model.config, "image_token_index", 32000))
        span_idx = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0]
        if len(span_idx) > 0:
            tracker.reset(int(span_idx[0]), int(span_idx[-1]) + 1)
        else:
            tracker.reset(0, 0)

    cache = None
    ids = input_ids
    sampled: list[int] = []
    eos = tokenizer.eos_token_id
    for step in range(max_new_tokens):
        if cache is not None:
            out = model(
                input_ids=ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                output_attentions=tracker is not None,
            )
        else:
            out = model(
                input_ids=ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                use_cache=True,
                output_attentions=tracker is not None,
            )
        cache = out.past_key_values
        if tracker is not None and out.attentions is not None and tracker.span is not None:
            tracker.update(out.attentions)
        logits = out.logits[0, -1].float()
        next_id = sample_token(logits, cfg, step)
        sampled.append(int(next_id))
        ids = next_id.unsqueeze(0)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=device)], dim=-1)
        if eos is not None and int(next_id) == eos:
            break

    text_out = tokenizer.decode(sampled, skip_special_tokens=True).strip()
    return text_out, sampled