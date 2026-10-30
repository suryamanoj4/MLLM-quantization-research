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

## Calibration, PPL and seed details

### LoRAS-T was calibrated on captions only (no yes/no questions)

`lorast.json` records `source: gen`, `qa_per_image: 0`, `n_qa_batches: 0`,
`n_calib: 192`: the correctors were fitted to FP16-generated captions on 192
calibration images, rank 64, all 28 layers, ridge 1e-2, 12.9M parameters. The
question-augmented calibration from the LLaVA Table 4 ablation was **not** run for
Qwen, so there is no “captions + QA” Qwen row.

POPE for the W3-RTN + LoRAS-T cell (scale 1.0; FP16 reference in brackets):

| Split | F1 | Precision | Recall | Yes-ratio | AUROC |
|---|---:|---:|---:|---:|---:|
| random | 0.8346 [0.8378] | 0.9864 [0.9954] | 0.7233 [0.7233] | 0.3667 [0.3633] | 0.9727 [0.9748] |
| popular | 0.8298 [0.8298] | 0.9731 [0.9731] | 0.7233 [0.7233] | 0.3717 [0.3717] | 0.9550 [0.9581] |
| adversarial | 0.8298 [0.8314] | 0.9731 [0.9775] | 0.7233 [0.7233] | 0.3717 [0.3700] | 0.9386 [0.9399] |
| mean / pooled | **0.8314** [0.8330] | — | — | ~0.370 [0.368] | ~0.9554 [0.9576] |

Paired deltas vs FP16: ΔF1 −0.002 [−0.018,+0.016], Δyes +0.002 [−0.011,+0.014],
ΔAUROC −0.002 [−0.007,+0.003], all n.s. Coverage −0.009 [−0.031,+0.011] n.s.
The scale-0.5 variant has mean F1 0.8380 but coverage 0.514 (−0.045*), so it recovers
POPE F1 by giving up coverage.

### WikiText PPL: what exists and what does not

`validate_ptq.json` contains only the **wiki-calibrated** W3 GPTQ row (9.324).
There is **no WikiText PPL for W3 GPTQ calibrated on image+caption (mm)** in this
archive, and **no WikiText PPL for either noise rung** (`w4a16:method=noise`,
`w3a16:method=noise`). Gate-2 “blind PPL” is not a substitute (see caveats).

To fill the gap on the GPU box:

```bash
python scripts/09_validate_ptq.py --coco-root data/coco-mini \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 \
    --out runs_qwen/validate_extra \
    --specs "w3a16:method=gptq,w3a16:method=gptq:calib=wiki,w4a16:method=noise,w3a16:method=noise"
```

### Noise seed

The Qwen Gate-2 noise rungs used the default **noise seed 0**
(`QuantConfig.rot_seed = 0`, global `seed = 0`); `matched_noise` derives a per-layer
generator from `rot_seed * 1_000_003 + crc32(layer_name)`. That is **one noise draw
per bit width**, not a multi-seed average. The LLaVA extra seed-1/seed-2 object
analysis was not run for Qwen. To request another draw, append `:seed=1` (or `2`)
to the rung spec.

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
