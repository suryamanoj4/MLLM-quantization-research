# Q1 — does LoRAS-T work because of *where* it corrects?

Run 2026-10-31, one A100-SXM4-80GB (Jarvis Labs), `method-study/scripts/run_q1.sh`
(package `phoenix_q1_lorast_positions`, see `method-study/README_Q1.md`).

## Question

LoRAS-T corrects the decoder's hidden states at **text positions only**. Is that placement
what repairs grounding, or would any rank-64 correction of the same size do as well?
Q1 fits the **same** corrector — same rank (64), same layers (all), same calibration data
(192 FP16 captions + 2 yes/no questions per image, `qa_per_image=2`), same ridge — at
**all positions**, image tokens included, and evaluates both on the Study-2 set.
The two variants have identical parameter counts (LLaVA 16.9M, Qwen 12.9M).

- text ≥ all → the text-side placement repairs grounding, not the added capacity.
- all > text → placement is not the reason.

## Setup

| | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---|---|
| Base | W3 RTN (`w3a16`), g128, language model only | same |
| Study-2 ladder reused | `runs/gate2/ladder.json` (Surya, 20 rungs) | `runs_qwen/gate2/ladder.json` (v3, 19 rungs) |
| Text-only corrector | `runs/lorast/w3_rtn_qa.pt` — the paper's; its rung was already evaluated, so reused by name (the `.pt` itself was not needed) | `runs_qwen/lorast_qa` (v3) |
| New: all-positions corrector | `runs/lorast/w3_rtn_qa_allpos.pt` | `runs_qwen/lorast_qa_allpos.pt` |
| Eval | 300 captions, 100 POPE images × 3 splits (1,800 questions), greedy, 128 tokens | same, bf16 |
| Stats | paired, image-clustered bootstrap, 2,000 resamples, 95% CI unadjusted (`*` excludes 0) | same |

Checks passed on both: `tests/test_lorast_positions.py` passed; the calibration ran with
`positions=all: FP16 targets recomputed per block`; the ladder resumed the existing rungs
(`[resume] reusing 20 / 19 rung(s)`) and evaluated only the new rung; the LLaVA text-only
row reproduces the paper exactly (ΔCov −0.012).

## Results

### Change vs FP16

| | LLaVA ΔCov | ΔCHAIR_i | ΔF1 | Δyes | ΔAUROC | Qwen ΔCov | ΔCHAIR_i | ΔF1 | Δyes | ΔAUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W3 RTN | −.072* | −.035* | +.013 | −.031* | −.001 | −.061* | −.004 | +.003 | +.007 | −.002 |
| + LoRAS-T, **text** positions | **−.012** | +.009 | +.005 | +.006 | −.000 | **−.017** | +.002 | **−.003** | −.001 | −.000 |
| + same, **all** positions | −.029* | −.002 | +.005 | −.002 | +.003 | −.019 | −.008 | **−.034*** | **−.026*** | +.003 |

### All positions minus text positions (paired, same items)

| | ΔCov | ΔCHAIR_i | ΔF1 | Δyes | ΔAUROC | Δrecall | Δlength (words) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaVA | **−.016*** [−.033, −.001] | −.011 | +.000 | −.008 | +.003* | −.007 | −0.7 |
| Qwen | −.002 [−.021, +.017] | −.010 | **−.031*** [−.047, −.015] | **−.025*** | +.003 | **−.047*** | +1.2* |

### Recovery vs W3 RTN

| | LLaVA ΔCov | ΔCHAIR_i | ΔF1 | Qwen ΔCov | ΔCHAIR_i | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| text | +.060* | +.045* | −.008 | +.044* | +.006 | −.006 |
| all | +.044* | +.034* | −.008 | +.042* | −.004 | **−.037*** |

(The CHAIR_i rise on LLaVA is a return to FP16 level: W3 RTN had *lowered* CHAIR_i by
naming fewer objects; the text-corrected model is +.009 n.s. vs FP16.)

### Object drop (share of FP16-named objects no longer named)

| | LLaVA small <1% | large >20% | all | Qwen small <1% | large >20% | all |
|---|---:|---:|---:|---:|---:|---:|
| W3 RTN | .279 | .061 | .158 | .360 | .118 | .205 |
| + text | **.140** | .046 | **.080** | **.245** | .059 | **.125** |
| + all | .181 | .053 | .094 | .252 | .017 | .134 |

### Calibration (held-out sequences, same split for both variants)

| | LLaVA text | LLaVA all | Qwen text | Qwen all |
|---|---:|---:|---:|---:|
| Tokens fitted per layer | ~41k | ~309k (7.5×) | ~34k | ~201k (5.9×) |
| Held-out KL(FP16 ‖ W3), before | .067 | .067 | .276 | .276 |
| … after correction | **.031** | .036 | **.075** | .089 |
| Top-1 agreement after | **.913** | .909 | **.862** | .852 |
| Yes/no \|ΔP(yes)\|, before → after | .046 → **.023** | .046 → .033 | .027 → **.014** | .027 → .023 |

Qwen text-only numbers are from `runs_qwen/lorast_qa.log` (its `.json` was not archived, so
`q1_summary.txt` prints "missing" for that row).

## Comparison

- **Coverage (omission).** Both placements recover most of the W3 coverage loss. On LLaVA
  the text corrector recovers more (+.060* vs +.044*; all − text −.016*) and brings
  coverage back to FP16 level, while all-positions leaves a significant −.029*. On Qwen the
  two are tied (−.002 n.s.).
- **POPE.** On LLaVA they are tied. On Qwen, correcting image positions **damages POPE**:
  F1 −.031*, yes-ratio −.025*, recall −.047* vs text-only — the model starts answering
  "no" for objects that are present. Text-only stays at FP16 level (ΔF1 −.003 n.s.).
- **Small objects.** Text-only halves the W3 small-object drop on both models
  (.279→.140, .360→.245); all-positions recovers less on LLaVA (.181) and about the same on
  Qwen (.252).
- **Fit quality.** With 6–7.5× more tokens to fit, all-positions still ends with a *higher*
  held-out KL, lower top-1 agreement and a worse yes/no shift on both models: the rank-64
  budget spent on image tokens is not spent on the text-side error that drives the output.

## Conclusion

**Text ≥ all on both models, at equal parameters, so the placement — not the capacity — is
what makes LoRAS-T work.** Extending the same corrector to image positions never improves a
grounding metric and costs something on each model: coverage on LLaVA, POPE on Qwen. It also
fits the FP16 output distribution less well on held-out data. This agrees with the
attribution results (text-position weight quantization drives the omission; image-position
quantization does not), and it rules out the reading that LoRAS-T only helps because it adds
16.9M / 12.9M parameters.

## Caveats

- One calibration draw per variant; CIs are unadjusted. The LLaVA coverage gap is
  significant but close to 0 (upper bound −.001); the Qwen POPE gap is clearer.
- The cost of all-positions shows up on a different metric per model, so "text ≥ all" is the
  robust claim; "all hurts coverage" or "all hurts POPE" alone is model-specific.
- W3 RTN base only (not W3 GPTQ). Qwen bf16, LLaVA fp16.

## Files

`llava/`, `qwen/` — the `results_q1_<model>` folders returned by `run_q1.sh`:
`q1_summary.txt|json`, `ladder_report_vs_fp16.json`, `pairwise_all_vs_text.json`,
`pairwise_vs_w3a16.json`, `object_analysis.json`, `calib_all.json` (+ LLaVA
`calib_text.json`), `ladder_q1_rungs.json` (the four rungs with captions), `logs/`.
Code: `method-study/scripts/phoenix/loras_text.py` (`positions="text"|"all"`, default text),
`scripts/12_calibrate_lorast.py --positions`, `scripts/run_q1.sh`, `q1_summary.py`,
`q1_resume_args.py`, `tests/test_lorast_positions.py`; run notes in `method-study/README_Q1.md`
(in this repo the tests and `run_q1.sh` need `export PYTHONPATH=scripts`, and the package
files are already in place — skip its `patch` step).
