---
title: Problem & Hypotheses — Lexical Fallback in Quantized MLLMs
date: 2026-08-25
tags:
  - research/methodology
  - project/core
aliases:
  - Lexical Fallback
status: in-progress
---

# Problem & Hypotheses

## 1. Problem Statement

Deploying MLLMs on edge hardware requires aggressive Post-Training Quantization (PTQ). Extreme precision reduction (W4A8/W4A4) disproportionately corrupts **cross-modal representations** — the visual token manifold projected into the language decoder. Unlike text tokens, multimodal token activations carry **significantly higher entropy** ([[06-Supporting-Evidence#LUQ]]), so low-bit rounding destroys visual conditioning before it degrades language fluency.

The hypothesized consequence is **Lexical Fallback**:

> [!warning] Lexical Fallback (the phenomenon under study)
> When visual token representations are degraded by quantization noise, the autoregressive generation head relies excessively on statistical **language priors**, sampling tokens that are linguistically probable but **visually ungrounded** — i.e., severe object hallucinations. The model "falls back" to behaving like a text-only LM.

### Why we measure this at generation time

The failure is a *decoding-time symptom* of corrupted representations: it manifests in what the decoder attends to and samples. Our evidence study therefore measures two things on the *same generations*: task-level hallucination, and step-by-step cross-modal attention. If hallucination coincides with measurable attention degradation, the lexical-fallback account is supported.

## 2. Gaps in Existing Work

| Gap | Evidence |
|---|---|
| Quantization and decoding are studied separately | QSLaw, MQuant optimize reconstruction error, not generation behavior ([[06-Supporting-Evidence#MQuant]]) |
| Generation heads are agnostic to noise | Quantized MLLMs use full-precision sampling (greedy, nucleus) unchanged |
| Hallucination mitigations assume full precision | VCD, OPERA, DoLa benchmarked on uncompressed models ([[06-Supporting-Evidence#Attention-Hallucination Evidence]]) |
| **No hallucination-focused ablation across precision levels** | LUQ reports POPE as one column among 9 benchmarks; no attention-mechanism analysis ([[06-Supporting-Evidence#Gap Analysis]]) |

## 3. Research Questions

- **RQ1 (Existence):** Does hallucination increase monotonically as precision decreases, *within the same dataset and prompt set*?
- **RQ2 (Mechanism):** Does quantization measurably degrade cross-modal attention (mass, entropy, stability), and does this degradation coincide with hallucination onset?
- **RQ3 (Causality):** At token level, are hallucinated tokens statistically less visually grounded than correct tokens — is lost grounding a sufficient explanation for the observed hallucinations?
- **RQ4 (Temporal):** Does visual grounding collapse over decoding steps (late-step fallback), rather than uniformly?

## 4. Hypotheses

> [!example] H1 — Precision monotonicity
> Hallucination rate (CHAIR_s, CHAIR_i, POPE-F1) increases monotonically with precision reduction: FP16 < W8A8 < W4A8 < W4A4, consistently across models and quantizers.

> [!example] H2 — Attention degradation
> Quantization reduces visual attention mass $\bar{a}_v$ and increases attention entropy over visual tokens, with the largest drop at W4A4.

> [!example] H3 — Token-level grounding coupling
> At the token level, hallucinated object mentions receive significantly less visual attention than grounded ones (negative point-biserial correlation, $p < 0.05$).

> [!example] H4 — Temporal fallback
> Visual attention mass decays across decoding steps, and the decay is steeper and earlier for quantized models than FP16 — the "fallback timeline."

## 5. Position & Novelty

- **First** same-dataset, same-prompt ablation of hallucination metrics across FP16/W8A8/W4A8/W4A4 on LLaVA-1.5 (7B/13B) and Qwen2-VL-7B.
- **First** mechanistic (attention-based) account of quantization-induced hallucination.
- Must position carefully against LUQ (already reports POPE on sub-4-bit LLaVA-1.5) and arXiv:2602.13289 (reliability study on quantized Qwen2-VL-7B) — differentiators: hallucination as object of study, full precision grid, attention mechanism, two model families ([[06-Supporting-Evidence#Gap Analysis]]).

## 6. Phenomenon Chain (what we are testing)

```mermaid
graph LR
    A[PTQ weight-space fixes<br/>QSLaw · MQuant · QuaRot] --> B[Quantized MLLM]
    B --> C[Corrupted cross-modal representations]
    C --> D[Degraded visual attention<br/>lower mass · higher entropy]
    D --> E[Lexical fallback<br/>prior-driven sampling]
    E --> F[Hallucinations<br/>CHAIR · POPE]
    C -.-> F
```

Each arrow is testable: B→C via embedding-error profiles (S1), C→D via attention metrics (H2), D→E via temporal collapse (H4), E→F via linguistic-prior probe (S2). See [[02-Evidence-Experiment-Design#Support Experiments]].

## Related Notes

- [[02-Evidence-Experiment-Design]] — how these hypotheses become an experiment
- [[03-Metrics-Definitions]] — formal definitions of every metric
- [[06-Supporting-Evidence]] — verified facts backing every claim above
- [[README]] — vault index