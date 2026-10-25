---
title: Results Comparison — LLaVA vs Qwen2.5-VL
date: 2026-10-25
tags:
  - research/results
  - project/comparison
aliases:
  - Results Comparison
  - Cross-model comparison
status: complete
---

# Results Comparison — LLaVA-1.5-7B vs Qwen2.5-VL-7B

> [!abstract] Purpose
> The **single comparison table** for the method-study replication. LLaVA numbers are
> from `method-study/runs/` (Gate 2 / 2×2, 2026-10-02); Qwen numbers are from
> `obsidian-docs/results/qwen-2.5vl-2026-10-25/` (Gate 2 / 2×2, 2026-10-25). Both use
> the same shared pipeline (`run_all.sh` / `run_all_qwen.sh`). Qwen is a more
> conservative FP16 baseline, so **compare deltas, not absolute scores**.

## 1. FP16 baseline

| Metric | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---:|---:|
| Caption coverage | 0.744 | 0.559 |
| CHAIR_i | 0.220 | 0.109 |
| CHAIR_s | 0.390* | 0.250 |
| POPE-F1 (3-split mean) | 0.854 | 0.833 |
| POPE yes-ratio | 0.542 | 0.368 |
| Caption length | 92.5 words | 106.9 words |
| WikiText-2 PPL | 7.27 | 8.28 |

\* LLaVA CHAIR_s from the separate 100-image evidence run; other LLaVA values are
Gate-2. Qwen values are Gate-2 (300 captions, 100 POPE images).

## 2. Master table — Gate 2 common rungs

`*` = paired 95% image-clustered CI excludes zero. Qwen absolute columns are shown
because the baseline differs; LLaVA absolute values are in [[Results]] §5 and the
paper table. Qwen `blind PPL` is deliberately omitted: it is not usable (see §7).

| Rung | LLaVA ΔCOV | LLaVA ΔCHAIR_i | LLaVA ΔF1 | Qwen COV | Qwen CHAIR_i | Qwen F1 | Qwen ΔCOV | Qwen ΔCHAIR_i | Qwen ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | — | — | — | 0.559 | 0.109 | 0.833 | — | — | — |
| W4A16 GPTQ (wiki) | −0.018* | −0.026* | +0.001 | 0.575 | 0.102 | 0.826 | +0.015 | −0.007 | −0.007 |
| W3 RTN | −0.072* | −0.035* | +0.013 | 0.498 | 0.105 | 0.836 | **−0.061*** | −0.004 | +0.003 |
| W3 GPTQ (wiki) | −0.045* | −0.028* | +0.005 | 0.513 | 0.116 | 0.816 | **−0.046*** | +0.006 | −0.017 |
| W3 GPTQ (mm) | −0.025* | −0.036* | −0.005 | 0.555 | 0.128 | 0.849 | −0.004 | +0.019 | +0.016 |
| W3 AWQ (wiki) | −0.042* | −0.014 | −0.011 | 0.535 | 0.118 | 0.837 | −0.025* | +0.009 | +0.004 |
| W4A4 rot+GPTQ | −0.002 | −0.001 | +0.002 | 0.540 | 0.097 | 0.770 | −0.019 | −0.013 | **−0.063*** |
| W3 matched noise | −0.097* | −0.046* | −0.011 | 0.518 | 0.113 | 0.797 | **−0.042*** | +0.004 | −0.036* |
| W4 matched noise | −0.027* | −0.030* | −0.003 | 0.550 | 0.120 | 0.826 | −0.009 | +0.010 | −0.007 |
| W3 weights @ image positions | −0.008 | −0.006 | +0.004 | 0.563 | 0.114 | 0.821 | +0.004 | +0.005 | −0.012 |
| W3 weights @ text positions | −0.069* | −0.040* | +0.008 | 0.501 | 0.103 | 0.817 | **−0.059*** | −0.006 | −0.016 |

**Reading.** The omission signature replicates: W3 RTN loses coverage on both
models with CHAIR_i flat; W3 GPTQ-wiki loses coverage; text-position weights drive
it and image-position weights do not; matched noise reproduces it. Two differences:
Qwen’s W4A16 GPTQ is harmless (coverage +1.5, n.s.), and Qwen’s realistic W4A4
(`rot+GPTQ`) loses **6.3 POPE-F1 points**, unlike LLaVA’s W4A4 cell.

## 3. Object drop by size

Share of FP16-mentioned objects the rung no longer names. Qwen’s FP16 already
mentions fewer small objects, but the same small-first tail appears.

| Model | FP16 mention <1% / >20% | W3 RTN <1% / >20% / all | W3 GPTQ-wiki all | W3 text-weights all | LoRAS-T all |
|---|---|---:|---:|---:|---:|
| LLaVA | 0.608 / 0.923 | 0.279 / 0.061 / 0.158 | 0.130 | 0.154 | 0.080 |
| Qwen | 0.319 / 0.838 | 0.360 / 0.118 / 0.205 | 0.167 | 0.172 | 0.110 |

## 4. 2×2 ablation at W4A8

| Cell | LLaVA coverage | LLaVA CHAIR_i | LLaVA CHAIR_s | Qwen coverage | Qwen CHAIR_i | Qwen CHAIR_s |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.771 | 0.202 | 0.59 | 0.545 | 0.103 | 0.21 |
| + LoRAS | 0.756 | 0.215 | 0.59 | **0.574** | 0.099 | 0.27 |
| + A-CAB (add, τ-pct 60) | 0.753 | 0.220 | 0.60 | 0.556 | 0.114 | 0.27 |
| + both | 0.720 | 0.221 | 0.59 | 0.552 | 0.092 | 0.21 |

Qwen paired CHAIR changes: LoRAS −0.005 [−0.039,+0.030], CHAIR_s +0.060
[−0.020,+0.140]; A-CAB +0.011 [−0.020,+0.039], CHAIR_s +0.060 [−0.010,+0.130];
both −0.011 [−0.050,+0.028], CHAIR_s 0.000 [−0.080,+0.070]. Coverage has no CI in
the artifact. LLaVA’s 2×2 has no per-item artifact for a paired CI.

## 5. Methods summary

| Method | LLaVA | Qwen2.5-VL |
|---|---|---|
| KV-LoRAS ceilings | K 0.67 / V 0.58 | K/V 0.22–0.37 |
| KV-LoRAS rank-16 error removed | K 42% / V 25% | 10–21% |
| KV-LoRAS 2×2 Δcoverage | −1.6 points | **+2.9 points** (no CI) |
| KV-LoRAS 2×2 ΔCHAIR_i | +1.3 n.s. | −0.5 n.s. |
| LoRAS-T Δcoverage vs FP16 | −1.2 n.s. | −0.9 n.s. |
| LoRAS-T small-object drop | 0.28→0.14 | 0.36→0.21 |
| LoRAS-T all-object drop | 0.16→0.08 | 0.205→0.110 |
| LoRAS-T params | 16.9M | 12.9M |
| A-CAB best honest point | α0.5/τ40: CHAIR_i 0.188 vs 0.210, coverage 0.788 vs 0.770 | τ60/α1: 0.078 vs 0.091, coverage 0.558 vs 0.580 |
| A-CAB high α | brevity trap (coverage 0.52, rep-4 0.31) | CHAIR worsens, no brevity trap (length ~106, rep-4 ≤0.006) |

LoRAS-T on Qwen: held-out KL 0.276→0.071, top-1 0.729→0.872, rank 64, all 28 layers.
LLaVA LoRAS-T: held-out KL ≈0.031 vs FP16 text, top-1 0.913.

## 6. Quantizer health — WikiText-2 PPL

| Rung | LLaVA | Qwen |
|---|---:|---:|
| FP16 | 7.27 | 8.28 |
| NF4 | 7.39 | 8.38 |
| W8A8 / W8A8+SQ | 7.30 / 7.31 | 8.44 / 8.35 |
| W4A16 / GPTQ | 7.50 / 7.56 | 8.83 / 8.43 |
| W3 RTN | 9.14 | 10.66 |
| W3 AWQ (wiki) | 8.47 | 10.29 |
| W3 GPTQ (wiki) | 8.49 | 9.32 |
| W4A4 plain | 1000.8 | 59988.7 |
| W4A4 rot / rot+GPTQ | 10.64 / 8.53 | 11.18 / 10.36 |

Rotation rescues W4A4 on both. SmoothQuant W8A8 is near FP16. GPTQ@W3 closes ~56%
of RTN’s gap on Qwen (9.32 vs 10.66; FP16 8.28); AWQ@W3 closes ~15% (10.29).

## 7. A-CAB sweep (selected rows)

Qwen uses 60 captions / 50 POPE-adversarial; LLaVA same protocol. `length` in words;
`rep4` is the 4-gram repetition rate.

| Config | Qwen CHAIR_i | Qwen cov | Qwen len | Qwen rep4 | LLaVA CHAIR_i | LLaVA cov | LLaVA len | LLaVA rep4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (τ40/α0) | 0.091 | 0.580 | 106.3 | 0.002 | 0.210 | 0.770 | 90.8 | 0.005 |
| τ40 / α0.5 | 0.102 | 0.547 | 105.6 | 0.003 | **0.188** | **0.788** | 89.4 | 0.007 |
| τ60 / α1 | **0.078** | 0.558 | 105.8 | 0.004 | 0.219 | 0.741 | 88.5 | 0.008 |
| τ40 / α2 | 0.125 | 0.588 | 106.9 | 0.005 | 0.178 | 0.573 | 84.8 | **0.231** |
| τ40 / α4 | 0.126 | 0.580 | 106.8 | 0.004 | 0.139 | 0.522 | 81.0 | **0.311** |
| τ80 / α1 | 0.109 | 0.569 | 106.1 | 0.004 | 0.177 | 0.748 | 91.2 | 0.007 |

Qwen’s best CHAIR point costs 2.2 coverage points; LLaVA’s α0.5 point is the only
coverage-preserving win. High α buys brevity/looping on LLaVA, not on Qwen.

## 8. Gate 1 (Qwen partial — 11/17)

No CIs because the run stopped before the report. FP16 coverage 0.565.

| Rung | Qwen coverage | Δ vs FP16 |
|---|---:|---:|
| FP16 | 0.565 | — |
| NF4 | 0.545 | −0.020 |
| W4A16 (RTN) | 0.543 | −0.022 |
| W4A16 AWQ (wiki) | 0.527 | −0.038 |
| W4A16 GPTQ (wiki) | 0.570 | +0.005 |
| W3 RTN | 0.475 | −0.090 |
| W3 AWQ (wiki) | 0.516 | −0.049 |
| W3 AWQ (mm) | 0.570 | +0.005 |
| W3 AWQ (text) | 0.563 | −0.002 |
| W3 GPTQ (wiki) | 0.509 | −0.056 |
| W3 GPTQ (mm) | 0.558 | −0.007 |

Missing: `w3a16:method=gptq:calib=text` (crashed: non-positive-definite Hessian),
W8A8, W8A8+SQ, W4A8+SQ, W4A4 rot, W4A4 rot+GPTQ (the last is covered by Gate 2).

## 9. Caveats

- **Qwen Gate-2 blind/caption PPL is not usable.** FP16 blind PPL ≈146 vs WikiText
  8.28, consistent with token-boundary misalignment when a caption is appended to a
  chat-template prompt. Use `validate_ptq.json`; `lm_damaged` flags in the Gate-2
  report are provisional.
- **No Qwen attention / `r_pb` measurement** in the archive; the LLaVA attention
  comparison (Table 2 / `r_pb`) has no cross-model counterpart yet.
- Qwen FP16 is more conservative (coverage 0.559 vs 0.744; CHAIR_i 0.109 vs 0.220);
  do not compare absolute scores across models.
- Qwen Gate 1 is partial (11/17); Gate 2 is complete.
- Qwen KV-LoRAS coverage gain (+2.9 points) has no paired CI; treat as provisional.

## Related Notes

- [[Results Summary]] — synthesis and verdict dashboard
- [[Method Study]] §9 — Qwen detail
- [[Results]] §6 — Qwen append-only entry
- `results/qwen-2.5vl-2026-10-25/` — archived artifacts
