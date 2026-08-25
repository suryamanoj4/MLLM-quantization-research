---
title: Visualization Spec
date: 2026-08-25
tags:
  - research/methodology
  - experiment/figures
status: in-progress
---

# Visualization Spec

Eight figures. Every figure maps to ≥1 hypothesis ([[01-Problem-and-Hypotheses#Hypotheses]]) and consumes metrics defined in [[03-Metrics-Definitions]]. Style: consistent color per precision level across all figures (FP16 = green, W8A8 = amber, W4A8 = orange, W4A4 = red); error bars = 95% bootstrap CI; 300 dpi, publication font sizes.

## F1 — Hallucination vs Precision (H1)

- **Type:** grouped bar chart (or line with markers), one panel per model.
- **Axes:** x = precision (FP16, W8A8, W4A8, W4A4); y = CHAIR_s, CHAIR_i (left), POPE-F1 (right).
- **Series:** quantizer per bar group (AWQ / GPTQ / MQuant) at each precision where applicable.
- **Purpose:** headline existence claim — hallucination rises as precision drops, within the same dataset/prompts. Monotone order FP16 < W8A8 < W4A8 < W4A4 = H1 confirmed.
- **Anti-cheat:** include FP16 published reference values (LLaVA-1.5: POPE-F1 87.3/86.1/84.2) as dashed lines to validate the harness.

## F2 — Fallback Timeline: attention mass over decoding steps (H4, H2)

- **Type:** line plot, one panel per model; x = generation step (decile of length), y = $\bar{a}_v$ (mean visual attention mass, [[03-Metrics-Definitions#2.2]]).
- **Series:** FP16 vs W4A4 (add W4A8 if trends demand).
- **Purpose:** the *mechanism picture* — does visual grounding collapse during generation? Quantized curves should decay earlier and steeper than FP16; late-step collapse = lexical fallback in action.
- **Supporting inset:** Drift rate ([[03-Metrics-Definitions#2.4]]) per decile to show argmax instability.

## F3 — Token-level Attention ↔ Hallucination Scatter (H3)

- **Type:** scatter with logistic fit + CI band; one panel per cell (or per precision, pooled over quantizers).
- **Axes:** x = mention-level $\bar{a}_v$; y = hallucination label {0,1} (jittered); overlay logistic curve.
- **Purpose:** direct evidence that hallucinated mentions are visually ungrounded. A steep negative slope shows attention tracks hallucination — the core mechanism claim of the study.

## F4 — Binned Grounding Curve (attention → hallucination function)

- **Type:** bar chart over deciles; x = $\bar{a}_v$ decile, y = hallucination rate per bin.
- **Purpose:** the quantitative relationship between grounding and hallucination (H3). Compare FP16 vs W4A4 curves: quantization should *shift* the curve (same attention → more hallucination), quantifying the "corrupted representation" penalty. The functional form is a core evidence output of this study.

## F5 — Attention Entropy Distributions (H2)

- **Type:** overlaid KDE/box plots; x = $H_v$ per token ([[03-Metrics-Definitions#2.3]]), split by precision.
- **Purpose:** quantization diffuses visual focus; right-shift in entropy = less discriminative visual attention. Supports LUQ's entropy story at the *attention* level.

## F6 — Attention Heatmaps (qualitative, H2/H3)

- **Type:** image grid — per example: (a) input image, (b) FP16 generation + attention heatmap (argmax patch trajectory or mean $\bar{a}_v$ over visual grid), (c) W4A4 same.
- **Selection:** 6 curated cases: 2 where W4A4 hallucinates (attention visibly off-object), 2 where both correct, 2 where quantized attention is visibly diffuse even when correct.
- **Purpose:** reviewer-friendly mechanism intuition; heatmaps of the 24×24 (LLaVA) visual grid resized to image.

## F7 — Reasoning Trade-off (context)

- **Type:** grouped bar; x = precision, y = ScienceQA acc / TextVQA acc.
- **Purpose:** shows the *cost* of quantization on correctness beyond hallucination; frames hallucination as the sharper degradation. Used in the introduction of the paper.

## F8 — Case Study Table

- **Type:** table (paper-ready) — 4–6 rows; columns: image ID, prompt, FP16 caption, W4A4 caption, FP16 $\bar{A}_v$, W4A4 $\bar{A}_v$, hallucinated mentions (CHAIR-labeled), linguistic-prior probe result (S2, [[02-Evidence-Experiment-Design#Support Experiments]]).
- **Purpose:** concrete, human-readable existence proof; ties S2 (text-only prior probability) into a narrative.

## Figure → Hypothesis Map

| Figure | H1 | H2 | H3 | H4 | S1/S2/S3 |
|---|---|---|---|---|---|
| F1 | ✅ | | | | |
| F2 | | ✅ | | ✅ | S3 (layer panels optional) |
| F3 | | | ✅ | | |
| F4 | | | ✅ | | |
| F5 | | ✅ | | | |
| F6 | | ✅ | ✅ | | |
| F7 | | | | | context |
| F8 | ✅ | | ✅ | | S2 ✅ |

## Related Notes

- [[03-Metrics-Definitions]] — every y-axis defined here
- [[02-Evidence-Experiment-Design]] — cells & protocols the figures summarize
- [[01-Problem-and-Hypotheses]] — hypotheses being tested
- [[05-Execution-Plan]] — when each figure is produced