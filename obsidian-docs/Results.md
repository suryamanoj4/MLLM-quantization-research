---
title: Results Log — Main Study
date: 2026-08-26
tags:
  - research/results
  - project/log
status: in-progress
---

# Results Log

> [!info] Purpose
> Append-only log for every cell of the main study ([[Study Experiment]]). Each completed cell gets an entry below; hypothesis verdicts are tracked in the tracker at the bottom. Entries are added as experiments complete — this note is the growth point of the vault.

## 1. Cell Checklist

- [x] LLaVA-1.5-7B / **FP16** / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29); full 500 pending
- [x] LLaVA-1.5-7B / **W8A8** (simulated) / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29); full 500 pending
- [x] LLaVA-1.5-7B / **W4A16** (GPTQ, dequantized) / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29); full 500 pending
- [x] LLaVA-1.5-7B / **W4A8** (GPTQ + simulated A8) / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29); full 500 pending
- [x] LLaVA-1.5-7B / **W4A4** (GPTQ + simulated A4) / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29): **collapsed**
- [ ] Text-only probe (S2/S2a/S2b) — FP16 + W4A16 + W4A4, POPE + CHAIR
- [ ] S3 — layer-depth attention profile (derived from captured attention)

> [!note] Resampling — run of 2026-09-29
> POPE 100 images/split (600 questions/split, 1,800 total, 6-question blocks intact), CHAIR 100 images, seed 42, identical images/prompts/seeds on every rung. At this size the 95% CI on POPE-F1 is roughly ±3 pts and on CHAIR_s roughly ±10 pts. Artifacts: `obsidian-docs/results/100img-5rung-2026-09-29/`.

> [!warning] Protocol changes in this run (see [[Study Experiment]] §2–3)
> 1. **Every quantized rung is simulated quantization**: weights rounded to their integer grid and stored dequantized in fp16; activations rounded per token on entry to each decoder Linear. Reason: no int4 kernel runs on a T4 (sm75) with this stack (auto_gptq's extension does not build against torch 2.5; quanto's marlin needs sm80; transformers rejects quanto activation quant), and TorchAO's int8 matmul falls back to CPU int32 for ≤16 rows — every decode step (~110 s per POPE question).
> 2. **Methods per rung**: W8A8 = int8 per-channel RTN weights + int8 per-token activations. W4A16 = GPTQ int4 g128 weights (128 COCO train2014 captions, seed 42), dequantized through the QuantLinear itself (worst rel. err 1.1e-3). W4A8 / W4A4 = the **same** GPTQ weights + int8 / int4 per-token activations — so W4A16→W4A8→W4A4 differs only in activation bits.
> 3. **Scope** on every rung: the 224 decoder Linears. Vision tower, projector, embeddings and lm_head stay fp16.
> 4. **CHAIR is greedy** as the protocol specifies (it previously inherited nucleus sampling). CHAIR numbers from before 2026-09-29 are not comparable.

## 2. Entry Template

> [!todo] Entry — `LLaVA-1.5-7B / <precision> / <method> / full|resampled`
> - **Date / status:** YYYY-MM-DD / complete | partial | collapsed
> - **CHAIR:** CHAIR_s = _ , CHAIR_i = _ (n = 500 images)
> - **POPE F1:** random _ · popular _ · adversarial _ (yes-ratio _ per split)
> - **Attention:** $\bar{A}_v$ = _ , entropy $\bar{H}_v$ = _ , drift rate = _ (per [[Research Ideation#Metrics at a Glance]])
> - **F2 fallback timeline:** decay onset decile = _ , slope vs FP16 = _
> - **S2/S2b:** prior-slope = _ , ΔKL = _
> - **Hypotheses touched:** H1 / H2 / H3 / H4
> - **Figures produced:** F_ — artifact path: `results/<precision>/`

## 3. Entries

> [!success] Entry — `LLaVA-1.5-7B / FP16 / plain / resampled-100`
> - **Date / status:** 2026-09-29 / complete
> - **CHAIR:** CHAIR_s = 0.390, CHAIR_i = 0.136 (n = 100 images, greedy; 679 mentions, 92 hallucinated)
> - **POPE F1:** random 0.8990 · popular 0.8505 · adversarial 0.8070 (yes-ratio 0.523 · 0.582 · 0.640)
> - **Attention:** $\bar{A}_v$ = 0.0997 (CHAIR), 0.102–0.105 (POPE splits); $\bar{H}_v$ = 5.49 bits (max 9.17)
> - **F2 fallback timeline:** decays from decile 1: 0.153 → 0.068 (last/first 0.44)
> - **H3:** r_pb(attention, grounded) = +0.216 [0.154, 0.277]
> - **S2/S2b:** not run
> - **Reproducibility:** POPE F1 and yes-ratio identical to 4 decimals with the 2026-09-16 FP16 run (per-image seeds)

> [!success] Entry — `LLaVA-1.5-7B / W8A8 / simulated int8 RTN + int8 act / resampled-100`
> - **Date / status:** 2026-09-29 / complete
> - **CHAIR:** CHAIR_s = 0.460, CHAIR_i = 0.153 (659 mentions, 101 hallucinated)
> - **POPE F1:** random 0.9055 · popular 0.8531 · adversarial 0.8089 (yes-ratio 0.505 · 0.567 · 0.625)
> - **Attention:** $\bar{A}_v$ = 0.1037 (CHAIR); POPE paired vs FP16 **+2.0%** [+0.0018, +0.0025]; $\bar{H}_v$ = 5.15
> - **F2 fallback timeline:** 0.159 → 0.072 (last/first 0.46)
> - **H3:** r_pb = +0.267 [0.204, 0.327]

> [!success] Entry — `LLaVA-1.5-7B / W4A16 / GPTQ g128, dequantized / resampled-100`
> - **Date / status:** 2026-09-29 / complete
> - **CHAIR:** CHAIR_s = 0.320, CHAIR_i = 0.115 (654 mentions, 75 hallucinated)
> - **POPE F1:** random 0.8878 · popular 0.8432 · adversarial 0.8006 (yes-ratio 0.540 · 0.595 · 0.653)
> - **Attention:** $\bar{A}_v$ = 0.0815 (CHAIR); POPE paired vs FP16 **−14.0%** [−0.0153, −0.0135]; $\bar{H}_v$ = 6.29
> - **F2 fallback timeline:** 0.133 → 0.054 (last/first 0.41); whole curve ~13–20% below FP16
> - **H3:** r_pb = +0.188 [0.120, 0.254]

> [!success] Entry — `LLaVA-1.5-7B / W4A8 / GPTQ g128 + simulated int8 act / resampled-100`
> - **Date / status:** 2026-09-29 / complete
> - **CHAIR:** CHAIR_s = 0.390, CHAIR_i = 0.145 (635 mentions, 92 hallucinated)
> - **POPE F1:** random 0.8982 · popular 0.8515 · adversarial 0.8006 (yes-ratio 0.532 · 0.588 · 0.653)
> - **Attention:** $\bar{A}_v$ = 0.0823 (CHAIR); POPE paired vs FP16 **−12.9%** [−0.0143, −0.0124]; $\bar{H}_v$ = 6.00
> - **F2 fallback timeline:** 0.133 → 0.055 (last/first 0.42)
> - **H3:** r_pb = +0.268 [0.204, 0.331]

> [!failure] Entry — `LLaVA-1.5-7B / W4A4 / GPTQ g128 + simulated int4 act / resampled-100`
> - **Date / status:** 2026-09-29 / **collapsed**
> - **POPE F1:** 0.069 · 0.069 · 0.063 (yes-ratio 0.028 · 0.035 · 0.033) — near-degenerate output
> - **CHAIR:** 3 object mentions across 100 captions (all hallucinated); NaN attention in late deciles
> - **Attention:** POPE paired vs FP16 +66% — not interpretable in a broken model
> - **Reading:** naive per-token RTN int4 activations with no outlier handling break the LLaMA decoder. This rung marks the **collapse boundary**, not lexical fallback; a meaningful W4A4 point needs outlier handling (rotation / smoothing). Excluded from H1–H4.

## 4. Hypothesis Verdict Tracker

| Hypothesis | Status | Evidence (entry refs) |
|---|---|---|
| H1 — Precision monotonicity | ⚠️ unresolved at n=100 | POPE-F1 shifts ≤1.1 pts vs ~±3 pt CI; CHAIR non-monotone (W8A8 worst, W4A16 best) within ~±10 pt CI; W4A16 yes-ratio +1.2–1.7 pts (predicted direction, small). Needs full 500. |
| H2 — Attention degradation | ✅ supported on the **weight** axis | W4A16 −14.0%, W4A8 −12.9% visual-attention mass vs FP16 (paired, image-bootstrap CIs exclude 0); entropy 5.49 → 6.29 bits. W8A8 +2.0%. |
| — axis attribution | weight rounding, not int8 activation rounding | W4A16 → W4A8 leaves attention unchanged (0.0815 → 0.0823). |
| H3 — Token-level grounding coupling | ✅ present at every functioning rung | r_pb +0.19 to +0.27, all CIs > 0 (grounded=1 coding: hallucinated mentions receive less visual attention). Persists under quantization; not strengthened. |
| H4 — Temporal fallback | ◐ decay is a baseline property | Visual attention decays through generation at every precision (FP16 0.153 → 0.068). W4 lowers the whole curve; relative decay marginally steeper (0.41 vs 0.44). |
| S2/S2a/S2b — Prior attribution | ⏳ pending | Probe not run. |
| S3 — Layer localization | ⏳ pending | — |

> [!note] Decoding outweighs W8/W4 quantization for CHAIR
> FP16 CHAIR_s is 0.39 under greedy (this run) vs 0.51 under nucleus sampling (2026-09-16) — a 12-point move from decoding alone, larger than any W8/W4 rung. Consistent with a decoding-time intervention (GAD) being the lever.

## Related Notes

- [[Study Experiment]] — the cells this log tracks
- [[Research Ideation]] — hypotheses, metrics, figures
- [[README]] — vault index
