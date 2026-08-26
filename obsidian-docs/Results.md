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

## 1. Cell Checklist (pending)

- [ ] LLaVA-1.5-7B / **FP16** / POPE (3 splits) + CHAIR (500)
- [ ] LLaVA-1.5-7B / **W8** (GPTQ) / POPE (3 splits) + CHAIR (500)
- [ ] LLaVA-1.5-7B / **W4** (GPTQ) / POPE (3 splits) + CHAIR (500)
- [ ] Text-only probe (S2/S2a/S2b) — FP16 + W4, POPE + CHAIR
- [ ] S3 — layer-depth attention profile (derived from captured attention)

> [!note] Resampling
> If a resampled run is used (config `sample_images` < full set), state the sample sizes here — e.g., "POPE 100 img/split, CHAIR 100" — and link the logged image IDs.

## 2. Entry Template

> [!todo] Entry — `LLaVA-1.5-7B / <precision> / GPTQ / full|resampled`
> - **Date / status:** YYYY-MM-DD / complete | partial | collapsed
> - **CHAIR:** CHAIR_s = _ , CHAIR_i = _ (n = 500 images)
> - **POPE F1:** random _ · popular _ · adversarial _ (yes-ratio _ per split)
> - **Attention:** $\bar{A}_v$ = _ , entropy $\bar{H}_v$ = _ , drift rate = _ (per [[Research Ideation#Metrics at a Glance]])
> - **F2 fallback timeline:** decay onset decile = _ , slope vs FP16 = _
> - **S2/S2b:** prior-slope = _ , ΔKL = _
> - **Hypotheses touched:** H1 / H2 / H3 / H4
> - **Figures produced:** F_ — artifact path: `results/<precision>/`

## 3. Entries

_No completed entries yet — cells land here as phases P0–P3 complete (see [[Study Experiment]])._

## 4. Hypothesis Verdict Tracker

| Hypothesis | Status | Evidence (entry refs) |
|---|---|---|
| H1 — Precision monotonicity | ⏳ pending | — |
| H2 — Attention degradation | ⏳ pending | — |
| H3 — Token-level grounding coupling | ⏳ pending | — |
| H4 — Temporal fallback | ⏳ pending | — |
| S2/S2a/S2b — Prior attribution | ⏳ pending | — |
| S3 — Layer localization | ⏳ pending | — |

## Related Notes

- [[Study Experiment]] — the cells this log tracks
- [[Research Ideation]] — hypotheses, metrics, figures
- [[README]] — vault index