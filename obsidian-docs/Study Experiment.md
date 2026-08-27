---
title: Main Study — 7B GPTQ Self-Quantized
date: 2026-08-26
tags:
  - experiment/design
  - project/main-study
status: planned
---

# Main Study — 7B GPTQ Self-Quantized

> [!abstract] Purpose
> The **main study** of this research: run **POPE (all 3 splits, full 9,000 questions) + CHAIR (full 500 captions)** on one model across three precision versions (FP16 / W8 / W4) with per-step attention capture and a text-only prior probe. Verifies **all four hypotheses (H1–H4)** plus S2 and S3 — the full mechanism story, not just "quantization hurts accuracy."

## 1. Why this setup

| Option | Attention hooks | Pre-quantized | Verifies |
|---|---|---|---|
| 7B GPTQ **self-quantized** (chosen) | ✅ | ❌ (+30 min) | H1–H4 + S2 + S3 |
| 13B GPTQ branches (TheBloke) | ✅ | ✅ | H1–H4 + S2 + S3 |
| 7B GGUF ladder (llama.cpp) | ❌ | ✅ | H1 + partial S2 only |

The 7B self-quantized route keeps the documented model family ([[Research Ideation#Key Verified Facts]]) and gains attention access — H3/H4 are what make the lexical-fallback claim meaningful. TheBloke's 7B GPTQ repo is deleted (verified); 13B is the only pre-quantized GPTQ survivor.

> [!note] Scope decision
> This single-model, single-quantizer (GPTQ), weight-only (W4A16/W8A16) study is the complete experiment of record. It does **not** compare models or quantization methods, and does **not** quantize activations (W4A4 regime) — weight-collapse alone triggering lexical fallback is the claim it establishes. Decoding-time countermeasures are future phases ([[Research Ideation#Research Plan]]).

## 2. Checkpoints & Quantization

1. **FP16 baseline:** `liuhaotian/llava-v1.5-7b` (official, ~14 GB)
2. **GPTQ W4 (g128):** self-quantize with GPTQModel on T4, ~15 min
3. **GPTQ W8:** same pipeline, ~15 min

> [!danger] Calibration discipline
> Calibration data: ~128 samples from **MSCOCO train2014**, fixed seed, **disjoint from all evaluation splits** (POPE/CHAIR use val2014). Same calibration set for W4 and W8. Log the exact configs in `configs/` ([[Research Ideation#Research Plan]]).

## 3. Datasets & Decoding — Full Sets

| Benchmark        | Protocol                                                                                                                              | Decoding                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **POPE — full**  | **All 3 splits** (random / popular / adversarial) × 500 images × 6 questions = **9,000 questions**; question files from RUCAIBox/POPE | Nucleus (top-p=0.9, temp=1), per-image seed |
| **CHAIR — full** | 500 MSCOCO val2014 images, one detailed caption each; GT = instance seg ∪ reference captions                                          | Greedy, max 256 tokens                      |

Fixed seed per image, identical prompts across all three precision versions — the pairing that matters ([[Research Ideation#Key Verified Facts]]). POPE's three splits give the full prior-strength axis: random (weak traps) → popular (frequent-object traps) → adversarial (co-occurring-object traps, strongest).

> [!note] Resampling is a code-config option, not a protocol change
> The harness supports a `sample_images` config (default = full set: 500 POPE images, 500 CHAIR images). If time constraints require it, the same protocol runs on any subsample (e.g., 100 images/split + CHAIR-100 ≈ 3–3.5 h) — sampling is by **image** (keeping each 6-question contrastive block intact), fixed seed, image IDs logged. This note is updated only if a resampled run becomes the record.

## 4. Measurements per Variant

- **Attention capture:** per-step rows over the 576-token visual prefix ([[Research Ideation#Metrics at a Glance]]), all layers/heads → $\bar{a}_v(t)$, entropy $H_v$, argmax drift, step-window profile.
- **Text-only probe (S2/S2a/S2b):** same prompts, image masked → per-question $P_{txt}(\text{yes})$, per-step logits for ΔKL ([[Research Ideation#Metrics at a Glance]]).
- **Task metrics:** POPE Acc/F1/yes-ratio **per split**; CHAIR_s / CHAIR_i; mention-level grounding labels for H3.

## 5. Hypothesis Coverage

| Hypothesis                     | Signal                                                  | Figure ([[Research Ideation]]) |
| ------------------------------ | ------------------------------------------------------- | ------------------------------ |
| H1 — monotone rise             | CHAIR_s/i and POPE-F1 ordered FP16 < W8 < W4, per split | F1                             |
| H2 — attention degradation     | $\bar{a}_v$ ↓, $H_v$ ↑ with precision loss              | F2, F5                         |
| H3 — token coupling            | r_pb < 0 on mention-level grounding                     | F3, F4                         |
| H4 — temporal fallback         | attention decay earlier/steeper for W4                  | F2                             |
| S2/S2a/S2b — prior attribution | yes-rate tracks $P_{txt}(\text{yes})$; ΔKL < 0          | F4-style curve                 |
| S3 — layer profile             | per-layer visual attention mass                         | panel in F2                    |

## 6. Time Budget (1× T4 free tier, full sets)

| Step | Time |
|---|---|
| Download FP16 checkpoint | ~10 min |
| Quantize GPTQ W4 + W8 | ~30 min |
| POPE (9,000 Qs) × 3 variants | ~3.3 h |
| CHAIR (500 × 256 tok) × 3 variants | ~4.7 h |
| Text-only probe (FP16 + W4) | ~4 h |
| Analysis + figures | ~1 h |
| **Total (full sets)** | **~13–14 h** |
| **Total (resampled, e.g. 100 img + CHAIR-100)** | **~3–3.5 h** |

Fits **Kaggle free** (30 h/wk) in one week — or ~6–8 h wall-clock running the two T4s in parallel (POPE on one, CHAIR on the other). On **Colab free** (~12 h sessions) full sets need two sessions; the resampled config fits one. CPU fallback: not recommended (GPTQ decode ~5× slower); Qwen2.5-VL-3B remains the CPU-only option if needed ([[Research Ideation#Key Verified Facts]]).

## 7. Expected Results (placeholders)

| Signal | FP16 | W8 | W4 | Interpretation |
|---|---|---|---|---|
| POPE-F1 random / popular / adversarial | ~87.3 / 86.1 / 84.2 (published) | ↓ 1–3 | ↓ 3–8 | Quantization inflates hallucination |
| Yes-ratio on "no" questions | baseline | ↑ | ↑↑ | False yes = prior wins |
| Yes-rate slope vs $P_{txt}(\text{yes})$ | shallow | mid | steepest | Behavior moves toward text-only LM |
| $\bar{a}_v$ visual attention mass | high | mid | low | Visual grounding weakens |
| Attention decay onset (decile) | late | mid | early | Fallback timeline |

Filled numbers land in [[Results]] and feed the proposal's "initial results" section.

## Related Notes

- [[Research Ideation]] — hypotheses, metrics, verified facts, research plan
- [[Results]] — where results are logged
- [[README]] — vault index