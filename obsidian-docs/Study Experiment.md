---
title: Main Study — 7B Precision Ladder (FP16 → W4A4)
date: 2026-08-26
tags:
  - experiment/design
  - project/main-study
status: planned
---

# Main Study — 7B Precision Ladder (FP16 → W4A4)

> [!abstract] Purpose
> The **main study** of this research: run **POPE (all 3 splits, full 9,000 questions) + CHAIR (full 500 captions)** on one model across a five-rung precision ladder (FP16 / W8A8 / W4A16 / W4A8 / W4A4) with per-step attention capture and a text-only prior probe. Verifies **all four hypotheses (H1–H4)** plus S2 and S3 — the full mechanism story, not just "quantization hurts accuracy."

## 1. Why this setup

| Option | Attention hooks | Pre-quantized | Verifies |
|---|---|---|---|
| 7B self-quantized **precision ladder** (chosen) | ✅ | ❌ (+30 min) | H1–H4 + S2 + S3 |
| 13B GPTQ branches (TheBloke) | ✅ | ✅ | H1–H4 + S2 + S3 |
| 7B GGUF ladder (llama.cpp) | ❌ | ✅ | H1 + partial S2 only |

The 7B self-quantized route keeps the documented model family ([[Research Ideation#Key Verified Facts]]) and gains attention access — H3/H4 are what make the lexical-fallback claim meaningful. TheBloke's 7B GPTQ repo is deleted (verified); 13B is the only pre-quantized GPTQ survivor.

> [!note] Scope decision
> This single-model, five-rung precision grid is the complete experiment of record. Rungs are chosen from the literature's information-rich set: **W8A8** (near-lossless control), **W4A16** (the weight-only deployment standard), **W4A8** (the activation-quant frontier), **W4A4** (the aggressive collapse regime). The ladder spans two clean axes — weight bits (16 → 8 → 4) and activation bits (16 → 8 → 4) — so each rung adds exactly one source of collapse. It does **not** compare quantization *methods* at the same bit-width (method is fixed per rung: GPTQ for the W4 weight scheme, quanto/torchao for activation quant), and does **not** scale the model or dataset. Decoding-time countermeasures are future phases ([[Research Ideation#Research Plan]]).

## 2. Precision Ladder & Quantization

| Rung | Config | Method | Cost |
|---|---|---|---|
| 1. **FP16** | `llava-hf/llava-1.5-7b-hf` (official, ~14 GB) | plain load | — |
| 2. **W8A8** | weights int8 + activations int8 (dynamic) | TorchAO | ~5 min |
| 3. **W4A16** | GPTQ W4 (g128), weight-only | self-quantize with auto_gptq | ~15 min |
| 4. **W4A8** | weights int4 + activations int8 | Quanto | ~5 min |
| 5. **W4A4** | weights int4 + **simulated** int4 activations | Quanto + fake-quant hooks | ~5 min |

> [!warning] W4A4 activations are simulated
> `optimum-quanto` caps activation quantization at int8 (hard-rejected `activations="int4"`), and torchao ≤0.8 has no int4-activation path either. The W4A4 rung therefore uses quanto int4 **weights** (real) plus a **fake-quant simulation** of int4 activations: per-token symmetric round-trip (absmax scale, clamp [-8,7], dequantize to fp16) applied as forward pre-hooks on the LM's `Linear` layers. The rounding is value-identical to real int4 activation quantization; only the GEMM accumulation differs (fp16 vs int32) — irrelevant for a behavior/mechanism study. It is **not** a packed int4 kernel. See `experiments/src/experiments/models/load.py:_fake_quantize_int4`.

> [!danger] Calibration discipline
> GPTQ W4A16 uses ~128 samples from **MSCOCO train2014**, fixed seed, **disjoint from all evaluation splits** (POPE/CHAIR use val2014). Quanto/TorchAO rungs are RTN-style and need no calibration. Log the exact configs in `configs/` ([[Research Ideation#Research Plan]]).

## 3. Datasets & Decoding — Full Sets

| Benchmark        | Protocol                                                                                                                              | Decoding                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **POPE — full**  | **All 3 splits** (random / popular / adversarial) × 500 images × 6 questions = **9,000 questions**; question files from RUCAIBox/POPE | Nucleus (top-p=0.9, temp=1), per-image seed |
| **CHAIR — full** | 500 MSCOCO val2014 images, one detailed caption each; GT = instance seg ∪ reference captions                                          | Greedy, max 256 tokens                      |

Fixed seed per image, identical prompts across all five rungs — the pairing that matters ([[Research Ideation#Key Verified Facts]]). POPE's three splits give the full prior-strength axis: random (weak traps) → popular (frequent-object traps) → adversarial (co-occurring-object traps, strongest).

> [!note] Resampling is a code-config option, not a protocol change
> The harness supports a `sample_images` config (default = full set: 500 POPE images, 500 CHAIR images). If time constraints require it, the same protocol runs on any subsample (e.g., 100 images/split + CHAIR-100) — sampling is by **image** (keeping each 6-question contrastive block intact), fixed seed, image IDs logged. This note is updated only if a resampled run becomes the record.

## 4. Measurements per Rung

- **Attention capture:** per-step rows over the 576-token visual prefix ([[Research Ideation#Metrics at a Glance]]), all layers/heads → $\bar{a}_v(t)$, entropy $H_v$, argmax drift, step-window profile.
- **Text-only probe (S2/S2a/S2b):** same prompts, image masked → per-question $P_{txt}(\text{yes})$, per-step logits for ΔKL ([[Research Ideation#Metrics at a Glance]]).
- **Task metrics:** POPE Acc/F1/yes-ratio **per split**; CHAIR_s / CHAIR_i; mention-level grounding labels for H3.

## 5. Hypothesis Coverage

| Hypothesis                     | Signal                                                                 | Figure ([[Research Ideation]]) |
| ------------------------------ | ---------------------------------------------------------------------- | ------------------------------ |
| H1 — monotone rise             | CHAIR_s/i and POPE-F1 ordered FP16 < W8A8 < W4A16 < W4A8 < W4A4, per split | F1                          |
| H2 — attention degradation     | $\bar{a}_v$ ↓, $H_v$ ↑ with precision loss, largest drop at W4A4        | F2, F5                         |
| H3 — token coupling            | r_pb < 0 on mention-level grounding                                     | F3, F4                         |
| H4 — temporal fallback         | attention decay earlier/steeper at low precision                        | F2                             |
| S2/S2a/S2b — prior attribution | yes-rate tracks $P_{txt}(\text{yes})$; ΔKL < 0                          | F4-style curve                 |
| S3 — layer profile             | per-layer visual attention mass                                         | panel in F2                    |

> [!note] Attribution between the axes
> The W4A16→W4A8 step isolates **activation collapse** at fixed W4; W8A8→W4A16 isolates **weight collapse** at fixed activations (modulo method, see [[#1. Why this setup]]). If hallucination tracks the weight axis (W8A8 ≈ FP16, W4A16 ↑) the trigger is weight noise; if it tracks the activation axis (W4A8 ≈ W4A16, W4A4 ↑↑) the trigger is activation rounding of visual tokens (LUQ's high-entropy claim).

## 6. Time Budget (1× T4 free tier, full sets)

| Step | Time |
|---|---|
| Download FP16 checkpoint | ~10 min |
| Quantize GPTQ W4A16 (+ quanto/torchao rungs) | ~35 min |
| POPE (9,000 Qs) × 5 rungs | ~5.5 h |
| CHAIR (500 × 256 tok) × 5 rungs | ~7.8 h |
| Text-only probe (FP16 + W4A16 + W4A4) | ~4 h |
| Analysis + figures | ~1 h |
| **Total (full sets)** | **~18–19 h** |
| **Total (resampled, e.g. 100 img + CHAIR-100)** | **~4–5 h** |

Fits **Kaggle free** (30 h/wk) in one week — or ~9–10 h wall-clock running the two T4s in parallel (POPE on one, CHAIR on the other). On **Colab free** (~12 h sessions) full sets need two sessions; the resampled config fits one. CPU fallback: not recommended (GPTQ decode ~5× slower); Qwen2.5-VL-3B remains the CPU-only option if needed ([[Research Ideation#Key Verified Facts]]).

## 7. Expected Results (placeholders)

| Signal | FP16 | W8A8 | W4A16 | W4A8 | W4A4 | Interpretation |
|---|---|---|---|---|---|---|
| POPE-F1 random / popular / adversarial | ~87.3 / 86.1 / 84.2 (published) | ≈ | ↓ 1–3 | ↓ 1–4 | ↓ 3–8 | Quantization inflates hallucination |
| Yes-ratio on "no" questions | baseline | ≈ | ↑ | ↑ | ↑↑ | False yes = prior wins |
| Yes-rate slope vs $P_{txt}(\text{yes})$ | shallow | shallow | mid | mid | steepest | Behavior moves toward text-only LM |
| $\bar{a}_v$ visual attention mass | high | high | mid | mid | low | Visual grounding weakens |
| Attention decay onset (decile) | late | late | mid | mid | early | Fallback timeline |

Filled numbers land in [[Results]] and feed the proposal's "initial results" section.

## Related Notes

- [[Research Ideation]] — hypotheses, metrics, verified facts, research plan
- [[Results]] — where results are logged
- [[README]] — vault index