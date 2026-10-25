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
> `obsidian-docs/results/qwen-2.5vl-2026-10-25/` (Gate 1 16/17, Gate 2, diagnostic
> ladder, 2×2, 2026-10-25; Gate 1 completion and the diagnostic ladder are the v2
> pass). Both use
> the same shared pipeline (`run_all.sh` / `run_all_qwen.sh`). Qwen is a more
> conservative FP16 baseline, so **compare deltas, not absolute scores**.

## 0. Does the LLaVA picture carry over?

| Finding | Direction | Magnitude | Where |
|---|---|---|---|
| Weight-only quantization → omission, POPE flat | ✅ | ✅ similar (W3 RTN ΔCOV −0.072 vs −0.061) | §2, §8 |
| Small objects dropped first | ✅ | Qwen drops 1.4–2.5× more of its FP16 mentions | §3, §8 |
| Text-position weights drive the drop, image-position weights don't | ✅ | ✅ (0.154 vs 0.172 all-object drop) | §2, §3 |
| Text-position A4 activations are the collapse trigger | ✅ | Qwen breaks earlier (A5 broken, A6 degraded) | §9 |
| Matched noise reproduces the omission | ✅ | weaker on Qwen (noise 0.142 vs RTN 0.205; LLaVA noise ≥ RTN) | §2, §3 |
| LoRAS-T halves the W3 object drop | ✅ | ✅ (0.16→0.08 vs 0.205→0.110) | §5 |
| KV-LoRAS / A-CAB give no clean CHAIR win at W4A8 | ✅ (null) | KV-LoRAS ceilings ~half of LLaVA's | §4, §5 |
| Rotation rescues W4A4 PPL; SmoothQuant W8A8 near-lossless | ✅ | ✅ | §6 |
| AWQ/GPTQ fail to beat RTN | ❌ they pass on Qwen (except AWQ@W3) | — | §6 |
| Per-rung Gate-1 verdicts (w4a16 fallback, `w4a4:rot` −2.0 F1) | ❌ | — | §8 |
| Realistic W4A4 (`rot+GPTQ`) is graceful | ❌ Qwen −6.3 POPE-F1* | — | §2, §8 |
| A-CAB high-α brevity trap | ❌ no brevity trap on Qwen | — | §7 |

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
| W4A16 RTN / GPTQ | 7.50 / 7.56 (GPTQ mm-calib) | 8.83 / 8.43 (GPTQ wiki-calib) |
| W3 RTN | 9.14 | 10.66 |
| W3 AWQ (wiki) | 8.47 | 10.29 |
| W3 GPTQ (wiki) | 8.49 | 9.32 |
| W4A4 plain | 1000.8 | 59988.7 |
| W4A4 rot / rot+GPTQ | 10.64 / 8.53 | 11.18 / 10.36 |

Rotation rescues W4A4 on both. SmoothQuant W8A8 is near FP16. **The published-method
checks behave differently:** on LLaVA, AWQ and GPTQ failed to beat RTN (GPTQ@W3 −39%
of the gap with mm calibration); on Qwen, GPTQ passes at W4 (closes 72%) and W3
(56%; 9.32 vs 10.66, FP16 8.28), AWQ passes at W4 (31%) and fails only at W3 (15%,
expect ≥25%). The re-implementations behave as published on Qwen.

## 7. A-CAB sweep (selected rows)

Qwen uses 60 captions / 50 POPE-adversarial; LLaVA same protocol. `length` in words;
`rep4` is the 4-gram repetition rate.

| Config | Qwen CHAIR_i | Qwen cov | Qwen len | Qwen rep4 | LLaVA CHAIR_i | LLaVA cov | LLaVA len | LLaVA rep4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (τ40/α0) | 0.091 | 0.580 | 106.3 | 0.002 | 0.210 | 0.770 | 90.8 | 0.005 |
| τ40 / α0.5 | 0.102 | 0.547 | 105.6 | 0.003 | **0.188** | **0.788** | 89.4 | 0.007 |
| τ60 / α1 | **0.078** | 0.558 | 105.8 | 0.004 | 0.219 | 0.741 | 88.5 | 0.008 |
| τ40 / α2 | 0.125 | 0.588 | 106.9 | 0.005 | 0.178 | 0.573 | 84.8 | **0.231** |
| τ40 / α4 | 0.126 | 0.580 | 106.8 | 0.003 | 0.139 | 0.522 | 81.0 | **0.311** |
| τ80 / α1 | 0.109 | 0.569 | 106.0 | 0.004 | 0.177 | 0.748 | 91.2 | 0.007 |

Qwen’s best CHAIR point costs 2.2 coverage points; LLaVA’s α0.5 point is the only
coverage-preserving win. High α buys brevity/looping on LLaVA, not on Qwen.

## 8. Gate 1 — published PTQ methods (Qwen 16/17)

Qwen: 100 captions + 100 POPE images, paired image-clustered bootstrap (2,000
resamples), `*` = CI excludes 0. LLaVA verdicts from [[Method Study]] §2 (same
protocol). Qwen `LM damaged` flags rely on blind PPL and are provisional (§12).

| Rung | LLaVA verdict / direction | Qwen ΔF1 | Qwen ΔCOV | Qwen ΔCHAIR_i | Qwen verdict / direction |
|---|---|---:|---:|---:|---|
| NF4 | ok / none | **+0.018*** | −0.020 | −0.006 | degraded / **fallback** (recall +0.033*, yes +0.021*) |
| W4A16 RTN | degraded / **fallback** | −0.007 | −0.022 | +0.001 | **ok** |
| W4A16 AWQ (wiki) | ok / none | +0.003 | −0.038* | −0.006 | degraded / omission |
| W4A16 GPTQ (wiki) | degraded / omission | −0.007 | +0.004 | −0.017 | degraded / none (len −1.5*) |
| W3 RTN | degraded / omission | +0.003 | **−0.090*** | +0.003 | degraded / omission (blind PPL 1.75×) |
| W3 AWQ (wiki) | degraded / mixed–omission, LM damaged | +0.004 | −0.049* | +0.019 | degraded / omission |
| W3 AWQ (mm) | degraded / mixed–omission | −0.003 | +0.004 | +0.009 | ok |
| W3 AWQ (text) | degraded / mixed–omission | +0.007 | −0.002 | +0.010 | ok |
| W3 GPTQ (wiki) | degraded / omission | −0.017 | −0.056* | −0.010 | degraded / omission |
| W3 GPTQ (mm) | degraded / mixed, LM damaged | +0.016 | −0.007 | −0.005 | degraded / **fallback** (recall +0.028*) |
| W3 GPTQ (text) | degraded / omission | — | — | — | **crashed** (non-positive-definite Hessian, also at `damp=0.1`) |
| W8A8 | ok | −0.010 | −0.013 | +0.002 | ok |
| W8A8 + SQ | ok | −0.002 | −0.009 | +0.011 | ok |
| W4A8 + SQ | ok | **+0.014*** | −0.004 | −0.012 | degraded / **fallback** (recall +0.030*, yes +0.021*) |
| W4A4 rot | **degraded / fallback, F1 −0.020*** (the only significant LLaVA cell) | +0.005 | **−0.070*** | −0.027 | degraded / mixed |
| W4A4 rot+GPTQ (wiki) | degraded / omission, F1 +0.002 | **−0.063*** | −0.011 | **−0.035*** | degraded / omission (recall −0.089*) |

**Reading.** The omission direction for weight-only RTN/AWQ/GPTQ-wiki carries over,
but per-rung verdicts do not: LLaVA's only significant POPE cell (`w4a4:rot`, −2.0)
does not replicate on POPE (Qwen loses coverage instead), while Qwen's realistic
W4A4 (`rot+GPTQ`) is the largest POPE drop in either model. Qwen also shows a
**fallback cluster** (NF4, W3 GPTQ-mm, W4A8+SQ: more "yes", higher recall, lower
precision) that LLaVA showed only at W4A16 RTN and `w4a4:rot`. ~6 false stars are
expected across the 120 Qwen comparisons, so trust the repeated patterns, not single
cells. Gate 2 reproduces the `rot+GPTQ` deltas exactly (F1 −0.063, recall −0.089).

Gate-1 object drop (share of FP16-mentioned objects dropped; Qwen 95% CI on "all"):

| Rung | LLaVA all / <1% / >20% | Qwen all [CI] | Qwen <1% / >20% |
|---|---|---|---|
| NF4 | 0.064 / 0.110 / 0.049 | 0.148 [0.103, 0.193] | 0.312 / 0.000 |
| W4A16 RTN | 0.098 / 0.193 / 0.024 | 0.139 [0.096, 0.181] | 0.292 / 0.057 |
| W3 RTN | 0.160 / 0.284 / 0.098 | 0.230 [0.176, 0.287] | 0.354 / 0.114 |
| W4A4 rot | 0.086 / 0.165 / 0.049 | 0.213 [0.147, 0.292] | 0.333 / 0.057 |

Qwen drops **1.4–2.5× more** of its own FP16-mentioned objects than LLaVA on these
rungs, and small-minus-large is positive on every Qwen Gate-1 rung (CI excludes 0 on
13/15; not on W8A8 and W4A8+SQ). Qwen's FP16 mentions only 30% of <1% objects.

## 9. Activation attribution — fine ladder (`--preset diagnostic`)

60 POPE / 40 caption images. Arm semantics (`phoenix/quant.py`):
`abits_text=8` → **image tokens at A4**, text at A8; `abits_image=8` → **text tokens
at A4**, image at A8. LLaVA rows use the corrected arm labels in [[Method Study]] §4.

| Rung | LLaVA | Qwen | Qwen detail |
|---|---|---|---|
| W4A8 | ok | ok | F1 −0.001 |
| W4A6 | ok | **degraded, LM damaged** | F1 −0.116*, recall −0.157*, COV −0.080*, CHAIR_i +0.069*, omission |
| W4A5 | degraded (fallback candidate) | **BROKEN** | 99% unparseable POPE, captions 6 words vs 108 |
| W4A4 | BROKEN (71% unparseable) | BROKEN | 100% unparseable, coverage 0.01 vs 0.58 |
| image @A4 only (`abits_text=8`) | degraded, LM intact | degraded, LM intact | F1 −0.300*, recall −0.350*, COV −0.134*, omission |
| text @A4 only (`abits_image=8`) | **BROKEN (81%)** | **BROKEN (100%)** | identical to full W4A4 |

**Reading.** The collapse trigger replicates: **text-position 4-bit activations
break both models**, image-position A4 alone degrades but stays fluent. Magnitude
does not: Qwen breaks one bit earlier (A5 vs A4) and is already degraded at A6, and
image-only A4 costs Qwen 30 POPE-F1 points. Qwen's activations are far less
tolerant of low-bit quantization.

## 10. Token probe (FP16 outlier census + blind fluency)

16 calibration images. Mean over layers of the share of entries per-token int4
rounding zeroes, and the largest |x|:

| Input | image erased@4 | text erased@4 | image max\|x\| | text max\|x\| |
|---|---:|---:|---:|---:|
| attn_in | 0.910 | 0.895 | 50.8 | 109.5 |
| attn_out | 0.757 | 0.258 | 19.4 | 62.8 |
| mlp_in | 0.762 | 0.625 | 122.5 | 127.5 |
| down_in | 0.974 | 0.973 | 310.0 | **2288.0** |

Text positions carry the largest outliers (down_proj input 2288 vs 310), which fits
the text-token collapse trigger in §9. Image tokens lose more entries at attention
outputs (largest gaps at layers 0, 5, 23, 27: +0.57–0.65).

| Blind (no image) | LLaVA PPL / rep-4 | Qwen PPL / len / rep-4 |
|---|---|---|
| FP16 | 15.9 / — | 146.3 / 51.2 / 0.003 |
| W4A8 | 15.7 / — | 117.9 / 51.6 / 0.003 |
| W4A4 | 143 / 0.26 | **30,260** / 14.4 / 0.075 |

Naive W4A4 breaks the language model with no image on both; Qwen's absolute blind
PPL is not comparable (§12), but the 200× jump is unambiguous.

## 11. Latency (Qwen, W4A8, simulated fake-quant, H100)

| Config | TTFT (ms) | Decode tok/s | Decode overhead | Peak VRAM (GB) |
|---|---:|---:|---:|---:|
| baseline | 146.6 | 17.36 | — | 15.67 |
| + LoRAS | 155.4 | 16.59 | +4.6% | 15.67 |
| + A-CAB (add, prev-gate) | 146.6 | 15.98 | +8.6% | 15.67 |
| + A-CAB (add, two-pass) | 146.1 | 13.09 | +32.6% | 15.67 |
| + A-CAB (mult, eager) | 148.6 | 13.65 | +27.1% | 15.68 |
| + LoRAS + A-CAB | 155.2 | 15.24 | +13.9% | 15.67 |

Fake-quant runs in bf16, so absolute speed is not a deployment number; the relative
overheads are what transfer. No LLaVA latency table exists in the vault.

## 12. Caveats

- **Qwen Gate-2 blind/caption PPL is not usable.** FP16 blind PPL ≈146 vs WikiText
  8.28, consistent with token-boundary misalignment when a caption is appended to a
  chat-template prompt. Use `validate_ptq.json`; `lm_damaged` flags in the Gate-1/2
  reports are provisional. It is still consistent for gross failures: blind PPL is
  identical for `w4a4` and the text-at-A4 arm (30,260), and for `w4a8` and the
  image-at-A4 arm (117.9), as it must be with no image.
- **No Qwen attention / `r_pb` measurement** in the archive; the LLaVA attention
  comparison (Table 2 / `r_pb`) has no cross-model counterpart yet.
- Qwen FP16 is more conservative (coverage 0.559 vs 0.744; CHAIR_i 0.109 vs 0.220);
  do not compare absolute scores across models.
- Qwen Gate 1 is 16/17: `w3a16:method=gptq:calib=text` fails (non-positive-definite
  Hessian, also with `damp=0.1`). Gate 2 and the diagnostic ladder are complete.
- Qwen Gate 2 used 300 captions (LLaVA 100); Gate 1 and the diagnostic ladder use
  the same n as LLaVA.
- Qwen ran in bf16, on H100 (v1) and A100-80GB (v2); deterministic rungs reproduce
  across GPUs.
- Qwen KV-LoRAS coverage gain (+2.9 points) has no paired CI; treat as provisional.

## Related Notes

- [[Results Summary]] — synthesis and verdict dashboard
- [[Method Study]] §9 — Qwen detail; §4 — corrected LLaVA attribution-arm labels
- [[Results]] §6 — Qwen append-only entry
- `results/qwen-2.5vl-2026-10-25/` — archived artifacts
