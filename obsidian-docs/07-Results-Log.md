---
title: Results Log — Evidence Study
date: 2026-08-25
tags:
  - research/results
  - project/log
status: in-progress
---

# Results Log

> [!info] Purpose
> Append-only log for every experiment cell of the evidence study ([[02-Evidence-Experiment-Design]]). Each completed cell gets an entry below; hypothesis verdicts are tracked in the tracker at the bottom. Entries are added as experiments complete — this note is the growth point of the vault.

## 1. Cell Checklist (pending)

- [ ] LLaVA-1.5-7B / FP16 / — / 3 seeds
- [ ] LLaVA-1.5-7B / W8A8 / AWQ
- [ ] LLaVA-1.5-7B / W8A8 / GPTQ
- [ ] LLaVA-1.5-7B / W4A8 / AWQ
- [ ] LLaVA-1.5-7B / W4A8 / MQuant
- [ ] LLaVA-1.5-7B / W4A4 / MQuant
- [ ] LLaVA-1.5-13B / FP16
- [ ] LLaVA-1.5-13B / W4A8 / MQuant
- [ ] LLaVA-1.5-13B / W4A4 / MQuant
- [ ] Qwen2-VL-7B / FP16
- [ ] Qwen2-VL-7B / W4A8 / MQuant
- [ ] Qwen2-VL-7B / W4A4 / MQuant
- [ ] S1 — noise-injection control (2–3 σ levels)
- [ ] S2 — linguistic-prior probe
- [ ] S3 — layer-depth attention profile

## 2. Entry Template

> [!todo] Entry — `<model> / <precision> / <quantizer> / seed(s)`
> - **Date / status:** YYYY-MM-DD / complete | partial | collapsed
> - **CHAIR:** CHAIR_s = _ , CHAIR_i = _ (n = 500 images)
> - **POPE F1:** random _ · popular _ · adversarial _ (yes-ratio _)
> - **ScienceQA acc:** _ % · **TextVQA acc:** _ %
> - **Attention:** $\bar{A}_v$ = _ , entropy $\bar{H}_v$ = _ , drift rate = _ (per [[03-Metrics-Definitions]])
> - **F2 fallback timeline:** decay onset decile = _ , slope vs FP16 = _
> - **Hypotheses touched:** H1 / H2 / H3 / H4
> - **Figures produced:** F_ — artifact path: `results/<model>/<precision>/<quantizer>/<seed>/`

## 3. Entries

_No completed entries yet — cells land here as phases P1–P4 complete (see [[05-Execution-Plan]])._

## 4. Hypothesis Verdict Tracker

| Hypothesis | Status | Evidence (entry refs) |
|---|---|---|
| H1 — Precision monotonicity | ⏳ pending | — |
| H2 — Attention degradation | ⏳ pending | — |
| H3 — Token-level grounding coupling | ⏳ pending | — |
| H4 — Temporal fallback | ⏳ pending | — |
| S1 — Noise attribution | ⏳ pending | — |
| S2 — Linguistic-prior provenance | ⏳ pending | — |
| S3 — Layer localization | ⏳ pending | — |

## Related Notes

- [[02-Evidence-Experiment-Design]] — the cells this log tracks
- [[03-Metrics-Definitions]] — metric definitions used in entries
- [[04-Visualization-Spec]] — figures referenced in entries
- [[05-Execution-Plan]] — phases producing these results
- [[README]] — vault index