---
title: Qwen2.5-VL-7B method-study run — 2026-10-25
date: 2026-10-25
tags:
  - research/results
  - project/method-study
  - model/qwen2.5-vl
status: complete-gate2-partial-gate1
---

# Qwen2.5-VL-7B method-study artifacts (2026-10-25)

Provenance: `~/Downloads/qwen_results.tgz`, extracted and reduced to the compact
artifacts below. Full raw runs (including `gate1/ladder.json`, `gate2/ladder.json`,
logs and `loras.pt`) remain in the archive.

## What is here

| File | Contents |
|---|---|
| `gate2_ladder_report.json` | 13-rung Gate-2 paired report vs FP16 (300 captions, 100 POPE images, 1,800 questions) |
| `gate2_object_analysis.json` | Per-object drop rates by size bin (`<1%`, `1–5%`, `5–20%`, `>20%`) |
| `eval/base.json`, `eval/loras.json`, `eval/acab.json`, `eval/both.json` | 2×2 ablation cells on W4A8 |
| `loras_diagnostics.json` | KV-LoRAS ceilings and fit/val reductions (16 correctors) |
| `drift.json` | Layer-depth drift profile and selected layers |
| `lorast.json` | LoRAS-T calibration metadata and held-out KL |
| `acab_sweep.json` | A-CAB α × τ sweep (15 rows) |
| `validate_ptq.json` | WikiText-2 PPL per quantizer (the reliable LM-health measure) |
| `token_probe.json` | FP16 outlier census and blind-fluency probe |
| `latency.json` | TTFT/decode throughput/VRAM at W4A8 |
| `figures/` | The four generated figures |

## Run configuration

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`, bfloat16, 28 decoder layers, 28/4 GQA heads,
  196 wrapped decoder Linears (6.53B parameters).
- Shared pipeline entry point: `run_all_qwen.sh` (same stages as LLaVA `run_all.sh`).
- Gate 2: 300 caption images, 100 POPE images (1,800 questions), 100 bootstrap
  resamples of images.
- Gate 1: **partial** — 11/17 rungs. The run stopped at
  `w3a16:method=gptq:calib=text` with
  `torch._C._LinAlgError: ... leading minor of order 18917 is not positive-definite`.
  W8A8/SmoothQuant and the rotation arms were not reached; Gate 2 covers
  `w4a4:method=rot+gptq` instead.

## Known measurement caveats

- **Gate-2 blind PPL and eval caption PPL are not usable for Qwen.** They report
  FP16 blind PPL ≈146 and caption PPL ≈253, which contradicts the WikiText-2 PPL
  (8.28 at FP16). This is consistent with token-boundary misalignment when a
  caption is appended to a chat-template prompt (BPE merges the leading space), so
  the scored target tokens shift. Use `validate_ptq.json` for language health and
  treat the `lm_damaged` flags in `gate2_ladder_report.json` as provisional.
- No visual-attention-mass / `r_pb` measurement is present in this archive, so the
  LLaVA attention result (Table 2) has **no Qwen counterpart yet**.
- Qwen's FP16 baseline is more conservative than LLaVA's: coverage 0.559 vs 0.744,
  CHAIR_i 0.109 vs 0.220, POPE-F1 ≈0.833 vs 0.854. Compare **deltas**, not absolute
  scores, across models.
