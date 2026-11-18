# W4 rows for the paper table (Results Comparison §0.1)

Run 2026-11-18, `method-study/scripts/run_4bit_table.sh`. Adds the W4 quantizer rows that
mirror the W3 ones, on the Study-2 settings, for both models:

| Row | Spec |
|---|---|
| W4 RTN | `w4a16` |
| W4 AWQ (wiki) | `w4a16:method=awq:calib=wiki` |
| W4 GPTQ (img+cap) | `w4a16:method=gptq` (calib=mm) |

Eval: 300 captions, 100 POPE images × 3 splits (1,800 questions), greedy, 128 tokens, seed 0,
g128, language model only. LLaVA fp16, Qwen bf16. Paired, image-clustered bootstrap,
2,000 resamples, 95% CI (`*` excludes 0).

**FP16 was re-run.** No Study-2 ladder was on the box, so each `ladder.json` has its own FP16
row, and the deltas are paired against it. Qwen reproduces the existing Gate-2 FP16 exactly
(Cov .559, CHAIR_i .109, F1 .833). LLaVA is within noise of it (Cov .741 / CHAIR_i .225 /
yes .543 vs .744 / .220 / .542).

## Files

- `{llava,qwen}/ladder.json`: per-rung metrics, including the re-run FP16
- `{llava,qwen}/ladder_report.json`: deltas vs FP16, CIs, verdicts, blind-PPL ratio (LLaVA PPL∅)
- `{llava,qwen}/object_analysis.json`: object-drop rate by size (`<1%` = the table's "small")
- `qwen/validate.json`: WikiText-2 PPL, FP16 8.281 and W4 GPTQ (img+cap) 8.674 (1.05×).
  W4 RTN (8.826) and W4 AWQ-wiki (8.657) are in `../qwen-2.5vl-2026-10-25/validate_ptq.json`.
- `logs/table.out`: full run log

## Result

There is no significant W4 coverage or CHAIR_i change on either model for RTN, AWQ (wiki) or
GPTQ (img+cap). The only starred cells are LLaVA yes-ratio: RTN +.017*, GPTQ (img+cap)
−.012*. Small-object drop: GPTQ (img+cap) < AWQ < RTN on both models (LLaVA .10 / .13 / .15,
Qwen .17 / .19 / .20).
