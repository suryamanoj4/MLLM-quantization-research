---
title: Qwen2.5-VL-7B method-study run — 2026-10-25
date: 2026-10-25
tags:
  - research/results
  - project/method-study
  - model/qwen2.5-vl
status: complete
---

# Qwen2.5-VL-7B method-study artifacts (2026-10-25)

Provenance: `~/Downloads/qwen_results.tgz` (first pass) and `qwen_results_v2.tgz`
(second pass: the five missing Gate-1 rungs and the diagnostic / attribution ladder),
extracted and reduced to the compact artifacts below. Full raw runs (including
`gate1/ladder.json`, `gate2/ladder.json`, `diagnostic/ladder.json`, logs and
`loras.pt`) remain in the archives.

## What is here

| File | Contents |
|---|---|
| `gate1_ladder_report.json` | 16-rung Gate-1 paired report vs FP16 (100 captions, 100 POPE images, 1,800 questions) — v2 |
| `gate1_object_analysis.json` | Gate-1 per-object drop rates by size bin — v2 |
| `diagnostic_ladder_report.json` | Fine activation ladder + attribution arms (`abits_text=8` / `abits_image=8`; 40 captions, 60 POPE images) — v2 |
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
- Gate 2: 300 caption images, 100 POPE images (1,800 questions), 2,000 bootstrap
  resamples of images.
- Gate 1: **16/17 rungs** (100 captions, 100 POPE images, 2,000 resamples). The first
  pass stopped at `w3a16:method=gptq:calib=text` with
  `torch._C._LinAlgError: ... leading minor of order 18917 is not positive-definite`;
  the second pass ran the five remaining rungs (W8A8, W8A8+SQ, W4A8+SQ, W4A4 rot,
  W4A4 rot+GPTQ) on an A100-80GB. A retry of the text-calibrated GPTQ rung with
  10× damping (`damp=0.1`) failed the same way, so that rung is a known gap.
  `w4a4:method=rot+gptq` reproduces its Gate-2 POPE deltas exactly (F1 −0.063,
  recall −0.089) across the two GPUs.
- Diagnostic ladder: `--preset diagnostic`, 60 POPE / 40 caption images.

## Known measurement caveats

- **Gate-2 blind PPL and eval caption PPL are not usable for Qwen as an absolute
  LM-health threshold.** They report
  FP16 blind PPL ≈146 and caption PPL ≈253, which contradicts the WikiText-2 PPL
  (8.28 at FP16). This is consistent with token-boundary misalignment when a
  caption is appended to a chat-template prompt (BPE merges the leading space), so
  the scored target tokens shift. Use `validate_ptq.json` for language health and
  treat the `lm_damaged` flags in the ladder reports as provisional. The measure is
  still internally consistent for gross failures: blind PPL is identical for
  `w4a4` and `w4a4:abits_image=8` (30,260; text tokens at A4 in both) and for
  `w4a8` and `w4a4:abits_text=8` (117.9), as it must be with no image present.
- No visual-attention-mass / `r_pb` measurement is present in this archive, so the
  LLaVA attention result (Table 2) has **no Qwen counterpart yet**.
- Qwen's FP16 baseline is more conservative than LLaVA's: coverage 0.559 vs 0.744,
  CHAIR_i 0.109 vs 0.220, POPE-F1 ≈0.833 vs 0.854. Compare **deltas**, not absolute
  scores, across models.
