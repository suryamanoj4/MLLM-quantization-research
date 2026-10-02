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

> [!seealso] Reading the results
> For the combined picture (verdicts + inference + cheat-sheet), read [[Results Summary]] — this note is the raw log it summarizes.

## 1. Cell Checklist

- [x] LLaVA-1.5-7B / **FP16** / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29); full 500 pending
- [x] LLaVA-1.5-7B / **W8A8** (simulated) / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29); full 500 pending
- [x] LLaVA-1.5-7B / **W4A16** (GPTQ, dequantized) / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29); full 500 pending
- [x] LLaVA-1.5-7B / **W4A8** (GPTQ + simulated A8) / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29); full 500 pending
- [x] LLaVA-1.5-7B / **W4A4** (GPTQ weights + **simulated** int4 activations) / POPE (3 splits) + CHAIR — resampled 100 img (2026-09-29): **collapsed**
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
> - **Attention:** $\bar{A}_v$ = 0.1037 (CHAIR); POPE paired vs FP16 **+2.0%** [+1.7%, +2.4%]; $\bar{H}_v$ = 5.15
> - **F2 fallback timeline:** 0.159 → 0.072 (last/first 0.46)
> - **H3:** r_pb = +0.267 [0.204, 0.327]

> [!success] Entry — `LLaVA-1.5-7B / W4A16 / GPTQ g128, dequantized / resampled-100`
> - **Date / status:** 2026-09-29 / complete
> - **CHAIR:** CHAIR_s = 0.320, CHAIR_i = 0.115 (654 mentions, 75 hallucinated)
> - **POPE F1:** random 0.8878 · popular 0.8432 · adversarial 0.8006 (yes-ratio 0.540 · 0.595 · 0.653)
> - **Attention:** $\bar{A}_v$ = 0.0815 (CHAIR); POPE paired vs FP16 **−14.0%** [−14.7%, −13.2%]; $\bar{H}_v$ = 6.29
> - **F2 fallback timeline:** 0.133 → 0.054 (last/first 0.41); whole curve ~13–20% below FP16
> - **H3:** r_pb = +0.188 [0.120, 0.254]

> [!success] Entry — `LLaVA-1.5-7B / W4A8 / GPTQ g128 + simulated int8 act / resampled-100`
> - **Date / status:** 2026-09-29 / complete
> - **CHAIR:** CHAIR_s = 0.390, CHAIR_i = 0.145 (635 mentions, 92 hallucinated)
> - **POPE F1:** random 0.8982 · popular 0.8515 · adversarial 0.8006 (yes-ratio 0.532 · 0.588 · 0.653)
> - **Attention:** $\bar{A}_v$ = 0.0823 (CHAIR); POPE paired vs FP16 **−12.9%** [−13.7%, −12.2%]; $\bar{H}_v$ = 6.00
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
| H2 — Attention degradation | ✅ supported on the **weight** axis | W4A16 −14.0%, W4A8 −12.9% visual-attention mass vs FP16 (paired, image-bootstrap CIs exclude 0); CHAIR −18.3% / −17.5%; entropy 5.49 → 6.29 bits. W8A8 +2.0%. Same size on all three POPE splits (§5.3). |
| — axis attribution | weight rounding, not int8 activation rounding | W4A16 → W4A8 leaves attention unchanged (0.0815 → 0.0823). |
| H3 — Token-level grounding coupling | ✅ present at every functioning rung | r_pb +0.19 to +0.27, all CIs > 0 (grounded=1 coding: hallucinated mentions receive less visual attention). Persists under quantization; not strengthened. |
| H4 — Temporal fallback | ◐ decay is a baseline property | Visual attention decays through generation at every precision (FP16 0.153 → 0.068). W4 lowers the whole curve; relative decay marginally steeper (0.41 vs 0.44). |
| S2/S2a/S2b — Prior attribution | ⏳ pending | Probe not run. |
| S3 — Layer localization | ⏳ pending | — |

> [!note] Decoding outweighs W8/W4 quantization for CHAIR
> FP16 CHAIR_s is 0.39 under greedy (this run) vs 0.51 under nucleus sampling (2026-09-16) — a 12-point move from decoding alone, larger than any W8/W4 rung. Consistent with a decoding-time intervention (GAD) being the lever.

## 5. Breakdown Tables — 100-image run (2026-09-29)

> [!info] Source
> Computed from the per-question outputs (`pope_<split>.jsonl`, `chair_captions.jsonl`) of the 2026-09-29 run. The numbers are saved in `results/100img-5rung-2026-09-29/breakdown.json`. The raw jsonl files are not in the repo; they are in the run's results zip. Error bars: at 600 questions per split, POPE accuracy has a 95% CI of roughly ±3 pts; CHAIR (100 captions) roughly ±10 pts.

### 5.1 POPE accuracy per rung and split (600 questions each)

| Rung | Weights / Acts | Split | Accuracy | F1 | Acc on "yes" Qs | Acc on "no" Qs | Yes-ratio |
|---|---|---|---|---|---|---|---|
| FP16 | 16 / 16 | random | **89.7%** | 0.899 | 0.920 | 0.873 | 0.523 |
| | | popular | **83.8%** | 0.851 | 0.920 | 0.757 | 0.582 |
| | | adversarial | **78.0%** | 0.807 | 0.920 | 0.640 | 0.640 |
| W8A8 | 8 / 8 | random | **90.5%** | 0.906 | 0.910 | 0.900 | 0.505 |
| | | popular | **84.3%** | 0.853 | 0.910 | 0.777 | 0.567 |
| | | adversarial | **78.5%** | 0.809 | 0.910 | 0.660 | 0.625 |
| W4A16 | 4 / 16 | random | **88.3%** | 0.888 | 0.923 | 0.843 | 0.540 |
| | | popular | **82.8%** | 0.843 | 0.923 | 0.733 | 0.595 |
| | | adversarial | **77.0%** | 0.801 | 0.923 | 0.617 | 0.653 |
| W4A8 | 4 / 8 | random | **89.5%** | 0.898 | 0.927 | 0.863 | 0.532 |
| | | popular | **83.8%** | 0.851 | 0.927 | 0.750 | 0.588 |
| | | adversarial | **77.0%** | 0.801 | 0.923 | 0.617 | 0.653 |
| W4A4 | 4 / 4 | random | **50.8%** | 0.069 | 0.037 | 0.980* | 0.028 |
| | | popular | **50.2%** | 0.069 | 0.037 | 0.967* | 0.035 |
| | | adversarial | **50.0%** | 0.062 | 0.033 | 0.967* | 0.033 |

\* Not real: W4A4 outputs garbage text, and the parser scores any non-"yes" output as "no".

**Average over the 3 splits:** FP16 83.8% · W8A8 84.4% · W4A16 82.7% · W4A8 83.4% · W4A4 50.3% (chance).

**By axis (average accuracy):**

| Step | What changes | Accuracy change |
|---|---|---|
| FP16 → W8A8 | 8-bit weights + 8-bit activations | +0.6 pts (noise) |
| FP16 → W4A16 | weights 16 → 4 bits | −1.1 pts (noise; predicted direction) |
| W4A16 → W4A8 | activations 16 → 8 bits | +0.7 pts (noise) |
| W4A8 → W4A4 | activations 8 → 4 bits | −33 pts (collapse) |

**Question-type buckets:** accuracy on "yes" questions is flat (0.91–0.93) on every rung and split. All of the random → popular → adversarial gap is on "no" questions (FP16: 0.87 → 0.76 → 0.64), which is the language prior producing false "yes" answers. W4 weights lower "no"-question accuracy by a further 2–3 pts on every split: the H1 direction, but within noise at n=100.

### 5.2 CHAIR accuracy (100 captions, greedy)

| Rung | Correct object mentions (1 − CHAIR_i) | Captions with no hallucination (1 − CHAIR_s) |
|---|---|---|
| FP16 | 0.865 | 0.61 |
| W8A8 | 0.847 | 0.54 |
| W4A16 | 0.885 | 0.68 |
| W4A8 | 0.855 | 0.61 |
| W4A4 | not usable (3 mentions in 100 captions) | not usable |

### 5.3 Visual-token attention (H2)

Share of attention that goes to the 576 image tokens, and the change vs FP16 on the same questions/captions.

| Rung | Image attention (POPE) | Change vs FP16 | Image attention (CHAIR) | Change vs FP16 | Reading |
|---|---|---|---|---|---|
| FP16 | 10.3% | — | 10.0% | — | Baseline |
| W8A8 | 10.5% | +2.0% [+1.7, +2.4] | 10.4% | +3.9% [+2.6, +5.3] | No real change |
| W4A16 | 8.9% | **−14.0%** [−14.7, −13.2] | 8.2% | **−18.3%** [−19.6, −17.0] | Looks at the image less |
| W4A8 | 9.0% | **−12.9%** [−13.7, −12.2] | 8.2% | **−17.5%** [−18.8, −16.1] | Same drop as W4A16, so the cause is the weights |
| W4A4 | — | — | — | — | Broken model, excluded |

**Per split (paired change vs FP16):**

| Rung | Random | Popular | Adversarial |
|---|---|---|---|
| W8A8 | +2.1% [+1.7, +2.4] | +2.1% [+1.7, +2.4] | +2.0% [+1.7, +2.4] |
| W4A16 | −14.3% [−15.1, −13.6] | −13.7% [−14.5, −13.0] | −13.8% [−14.5, −13.1] |
| W4A8 | −13.4% [−14.2, −12.6] | −12.7% [−13.5, −11.8] | −12.7% [−13.5, −11.9] |

The drop is the same size on all three splits: quantization reduces attention to the image evenly, while the splits differ in accuracy.

**Spread, stability and decay:**

| Rung | Entropy POPE / CHAIR (bits, max 9.17) | Drift POPE / CHAIR | CHAIR attention: first 10% → last 10% of caption (ratio) |
|---|---|---|---|
| FP16 | 5.63 / 5.49 | 0.264 / 0.270 | 0.153 → 0.068 (0.44) |
| W8A8 | 5.26 / 5.15 | 0.248 / 0.252 | 0.159 → 0.072 (0.46) |
| W4A16 | 6.41 / 6.29 | 0.328 / 0.332 | 0.133 → 0.054 (0.41) |
| W4A8 | 6.13 / 6.00 | 0.314 / 0.315 | 0.133 → 0.055 (0.42) |

W4 weights make image attention more scattered (+0.8 bits entropy) and less stable (drift +25%). Attention fades through every caption, even at FP16; W4 lowers the whole curve, and the fade is only slightly steeper (H4 partial).

### 5.4 How the attention numbers are calculated

1. **Per generated token** (`AttentionTracker` in `experiment/decoding.py`): in each of the 32 layers × 32 heads, take the new token's attention row (sums to 1 over all previous positions), sum the part on the 576 image-token positions, then average over heads and layers.
2. **Per question / caption:** average step 1 over all generated tokens (≈7 for a POPE answer, up to 256 for a CHAIR caption). Entropy = entropy (bits) of the attention renormalised over the 576 image tokens; drift = how often the most-attended image patch changes between consecutive tokens.
3. **Per rung:** mean over questions (POPE = mean of the three split means). **Change vs FP16** is paired, because every rung sees the same images, questions and seeds: (Σ rung − Σ FP16) / Σ FP16 over the same items.
4. **95% CI:** image-cluster bootstrap. Resample the 100 images with replacement (keeping each image's questions together), recompute step 3, repeat 2,000 times, and take the 2.5th–97.5th percentiles.

> Within a rung, per-question attention mass on POPE mostly reflects whether the model says "yes": generating "Yes, there is a …" pulls attention onto the image. FP16 "no" questions: accuracy 0.998 in the lowest-attention quartile vs 0.00 in the highest; "yes" questions: 0.17 vs 1.00. So POPE attention buckets **cannot** be used as grounding evidence, and H3 is measured on CHAIR object mentions only. The H2 rung-level comparison survives this confound: W4A16 says "yes" slightly *more* often (54.0% vs 52.3%), which would raise attention, yet attention fell 14%. The CHAIR drop (−18%) has no yes/no answer involved at all.

## Related Notes

- [[Study Experiment]] — the cells this log tracks
- [[Research Ideation]] — hypotheses, metrics, figures
- [[Method Study]] — the methods phase (LoRAS / A-CAB): PTQ validation, Gate 1/2 verdicts, 2×2 ablation, A-CAB sweep — artifacts in `method-study/runs/`
- [[README]] — vault index
