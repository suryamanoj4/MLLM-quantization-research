---
title: Method Study — LoRAS & A-CAB (Phoenix, Contributions 2 & 3)
date: 2026-10-02
tags:
  - research/methods
  - research/results
  - project/method-study
aliases:
  - Phoenix
  - Method Study
status: in-progress
---

# Method Study — LoRAS & A-CAB (Phoenix)

> [!abstract] Purpose
> The **methods phase** of the research (Contributions 2 & 3): once the [[Study Experiment]] establishes the evidence (H1–H4), this study develops and tests two training-free countermeasures on **practically quantized** MLLMs — **LoRAS** (low-rank activation steering, closed-form ridge fit on K/V outputs) and **A-CAB** (entropy-gated additive bias on visual attention logits) — plus the profiling (drift, token probe) and validation (WikiText PPL, Gate 1/2 classification) needed to show either one works. Everything runs on one A6000, nothing is trained.

> [!seealso] Reading the results
> For the synthesis across both studies (verdicts, cross-study inference, cheat-sheet), read [[Results Summary]]. This note holds the per-rung detail.

**Code:** [`method-study/`](../method-study/) — pipeline scripts `00_…–12_…`, entry points `run_all.sh` (LLaVA) and `run_all_qwen.sh` (Qwen2.5-VL), artifacts in `method-study/runs/` and `runs_qwen/`. Environment: torch 2.6.0+cu124, transformers 4.53.3, CUDA 12.4, LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) / Qwen2.5-VL-7B (`Qwen/Qwen2.5-VL-7B-Instruct`), mini-COCO (`data/coco-mini`, 500 eval / 256 calib images, disjoint splits enforced).

---

## 1. Validation of the PTQ toolchain (WikiText-2 PPL of the LLaVA LM)

> [!info] Source
> `runs/validate.log`, `runs/validate_ptq/validate.json`, `runs/validate_calib/validate.json` (n=64 calib unless noted; the `_calib` file compares calibration sources).

| Rung | PPL | Gap to fp16 | Note |
|---|---|---|---|
| fp16 | 7.273 | — | reference |
| w8a8 (RTN) | 7.298 | +0.025 | near-lossless ✅ |
| w8a8:method=sq | 7.311 | +0.038 | SmoothQuant ✅ |
| w4a16 (RTN) | 7.496 | +0.223 | |
| w4a16:awq | 7.529 | +0.256 | |
| w4a16:gptq | 7.558 | +0.285 | |
| w3a16 (RTN) | 9.138 | +1.865 | |
| w3a16:awq | 8.748 | +1.475 | |
| w3a16:gptq | 9.869 | +2.596 | **worse than RTN** (calib=mm) |
| nf4 | 7.393 | +0.120 | |
| **w4a4 (RTN)** | **1000.8** | +993.6 | broken — naive int4 activations |
| w4a4:rot | 10.644 | +3.371 | rotation rescues 99.7% of the gap ✅ |
| w4a4:rot+gptq | 8.531 | +1.258 | rescues 99.9% ✅ — **the realistic W4A4 row** |

> [!warning] Honest caveat — checks did not all pass
> The published-method checks (relative to RTN) came out **?? for AWQ/GPTQ**: AWQ@W3 closes 20.9% of RTN's gap (expect ≥25%), GPTQ@W3 closes **−39.2%** (expect ≥10%), AWQ@W4 −15.2% (expect ≥20%), GPTQ@W4 −27.8% (expect ≥0%). Rotation and SmoothQuant pass cleanly. The re-implemented AWQ/GPTQ do **not** beat RTN on the LLaVA LM at these settings, and GPTQ calibrated on image+caption data (mm) is worse than RTN at W3 (9.87 vs 9.14) while **WikiText-calibrated GPTQ@W3 closes 35% of the gap (8.49)** — calibration source matters. State this plainly in methods; single-rung conclusions from AWQ/GPTQ cells need the larger-n confirmation.

## 2. Gate 1 — published PTQ methods, direction classification

> [!info] Source
> `runs/gate1/ladder_report.json` — 17 rungs, POPE 100 images/split + CHAIR 100 (paired, image-clustered bootstrap, 2,000 resamples), reference = fp16.

**Verdicts (vs fp16):** `ok` = no significant change · `degraded (LM intact)` = grounding-specific, Gate-1 pass · `LM damaged` = cover with the no-image PPL guard (>1.10× fp16 blind PPL).

| Rung | Verdict | Direction | POPE-F1 Δ (95% CI) |
|---|---|---|---|
| nf4, w8a8, w8a8:sq, w4a8:sq, w4a16:awq:wiki | ok | none | +0.5 pts or less, n.s. |
| **w4a16 (RTN)** | degraded, LM intact | **fallback** | +0.1 pts, n.s. |
| w4a16:gptq:wiki | degraded, LM intact | omission | +0.1 pts, n.s. |
| w3a16 (RTN) | degraded, LM intact | omission | +1.3 pts, n.s. |
| w3a16:awq (mm/text/wiki) | degraded (wiki: LM damaged) | mixed / omission | ±1.1 pts, n.s. |
| w3a16:gptq (mm) | degraded, **LM damaged** | mixed | −0.5 pts, n.s. |
| w3a16:gptq (wiki/text) | degraded, LM intact | omission | ±0.5 pts, n.s. |
| **w4a4:rot** | degraded, **LM damaged** | **fallback** | **−2.0 pts [−3.5, −0.4], p=0.015** — the only significant cell |
| w4a4:rot+gptq:wiki | degraded, LM intact | omission | +0.2 pts, n.s. |

> [!note] Reading
> At n=100, weight-only W4/W3 costs **≈0–1.3 pts POPE-F1** (all n.s.) but systematically shifts behavior toward **omission** (fewer objects mentioned) with the LM intact — the *shape* differs from the evidence study's W4A16 "fallback" on simulated activations, because these are real methods on real weights. The credible degradation signal at this n is **w4a4:rot** (−2.0 pts, fallback direction) and the consistent omission direction across rungs. Verdicts are directional at this n, not final.

## 3. Object analysis — omission concentrates on small objects

> [!info] Source
> `runs/gate1/object_analysis.json` (10_object_analysis.py, CPU). Reference mention rates: objects <1% of the image are mentioned in **68.6%** of fp16 captions vs **91.1%** for >20% objects — small objects are already the fragile tail.

| Rung | Drop all | Drop <1% size | Drop >20% size |
|---|---|---|---|
| nf4 | 0.064 | 0.110 | 0.049 |
| w4a16 (RTN) | 0.098 | **0.193** | 0.024 |
| w3a16 (RTN) | **0.160** | **0.284** | 0.098 |
| w4a4:rot | 0.086 | 0.165 | 0.049 |

Drops are **2–3× larger for small objects** on every quantized rung — the omission direction is a *small-object* phenomenon, consistent with the Evidenced fragility of high-entropy visual tokens.

## 4. Gate 2 — attribution & controls

> [!info] Sources
> `runs/gate2/ladder_report.json`, `runs/gate2/object_analysis.json`, `runs/token_probe/token_probe.json`. POPE 100/split + CHAIR 100; token probe 16 images / 40 captions; weight-noise controls are RTN-matched Gaussian with per-group error energy.

- **Noise control:** w3a16:noise drops 18.5% of objects (vs 15.8–16.0% for real W3) — matched Gaussian noise **reproduces the omission**, so the effect is error-energy, not quantizer identity (S1-style evidence).
- **Token-selective weights (wtok):** quantizing weights **only at text positions** (w3a16:wtok=text) reproduces the omission (drop 0.154, degraded); quantizing **only at image positions** (wtok=image) is indistinguishable from fp16 (drop 0.072, ok). Text-position weight noise drives the object drop, not image-position weights.
- **NF4 simulation validated:** `nf4` (simulated) and `bnb-nf4` (real bitsandbytes kernels) are statistically identical (drops 0.065 vs 0.066, POPE-F1 +0.002 both).
- **Token probe (4-bit erasure census):** at layer 0, per-token int4 rounding would zero out **~99%** of attention-input entries (image and text alike; int8 ~55–65%). Blind (no-image) fluency: fp16 PPL 15.9 and w4a8 15.7 are fluent; **w4a4 blind PPL 143 + rep-4 0.26** — the naive-W4A4 LM itself is damaged, which is why every w4a4:rtn cell is excluded.
- **Fine ladder (`runs/ladder/`, 60 POPE / 40 CHAIR):** w4a8 ok, w4a6 ok, **w4a5 degraded — fallback-regime candidate**; w4a4 **BROKEN** (71% unparseable POPE; coverage 0.01 vs 0.76; captions 21 vs 88 words; `clip=0.999` makes it worse, `skip=down_proj` still broken). **Attribution arms:** `abits_text=8` (image tokens at 4 bits) → **degraded, LM intact** (CHAIR_i +0.056 [+0.001,+0.110]); `abits_image=8` (text tokens at 4 bits) → **BROKEN (81%)** — **text-position 4-bit activations are the collapse trigger**, image-position 4-bit activations are survivable. This is direct support for the outlier/massive-activation account: the fragility is on the language side, not in the visual tokens. *(Arm labels corrected 2026-10-25; the earlier text had them swapped. Check: the blind no-image probe has no image tokens, so the text-at-A4 arm must share full W4A4's blind PPL (143, LM damaged) and cannot be the "LM intact" arm. Qwen shows the same identity and the same trigger, §9.5.)*

## 5. Drift profile — where LoRAS should sit

> [!info] Source
> `runs/drift__w4a8/drift.json` (32 images, W4A8).

Drift (per-layer K/V error) grows monotonically toward the tail: K max-drift layers **[23, 25–31]**, V **[24–31]**, upper quartile [24–31]. The measurement matches the proposal's top-quartile rule (l ≥ 0.75L), so LoRAS is calibrated on layers 23, 25–31 (K and V outputs, pre-RoPE, rank 16, ridge 0.01, sequential calibration, layers-per-pass 4).

## 6. LoRAS — activation-space corrector (W4A8)

> [!info] Source
> `runs/loras__w4a8/loras_diagnostics.json` (256 calibration images).

- **Recoverability ceilings** (fraction of quantization error that is *linearly* fixable): K ceiling **0.64–0.72 (mean 0.67)**, V ceiling **0.56–0.61 (mean 0.58)** → **partial** by the README rubric (0.3–0.7): LoRAS can remove about half of K/V error, not all — the residual is genuinely non-linear.
- **Rank-16 correction:** K relative MSE −44% (train) / −42% (held-out val_red); V −28% / −25%. Held-out ≈ train ⇒ not overfit at this rank.
- **LoRAS-T (text-side variant, W3):** 32 correctors, 16.9M params (32 MiB); held-out KL 0.031 vs fp16-text, top-1 agreement 0.913, |ΔP(yes)| 0.023 at rank 64 (ceiling ≈0.57–0.58). One anomaly: L30 fits with ceiling 0.99 and zero "massive" tokens — flagged in the log.

## 7. The 2×2 ablation — LoRAS × A-CAB (W4A8, n=100+100, seed 0)

> [!info] Source
> `runs/eval__w4a8_{base,loras,acab,both}/results.json`, `runs/acab_sweep__w4a8/sweep.json` (60 captions).

| Cell | CHAIR_s | CHAIR_i | Coverage | Caption len | rep-4 | gated_frac | PPL |
|---|---|---|---|---|---|---|---|
| base | 0.59 | 0.202 | 0.771 | 91.7 | 0.009 | 0 | 12.74 |
| + LoRAS | 0.59 | 0.214 | 0.756 | 91.5 | 0.007 | 0 | 12.74 |
| + A-CAB (add, τ-pct 60) | 0.60 | 0.220 | 0.753 | 90.2 | 0.006 | 0.354 | 12.74 |
| + both | 0.59 | 0.221 | 0.720 | 89.6 | 0.010 | 0.358 | 12.74 |

> [!warning] Honest reading — no win at default settings (n=100)
> Neither method improved CHAIR at the default hyperparameters; coverage slid slightly (0.771 → 0.72–0.76). A-CAB fired on ~35% of steps with no benefit here. The A-CAB sweep tells the cautionary story: **α=0.5** gives a small honest win (CHAIR_i 0.188 vs 0.210 with coverage *up* to 0.788), but **α≥2 just buys brevity** — coverage collapses to 0.52–0.57, length to ~81–85, rep-4 explodes to 0.23–0.31 (looping). **Report the coverage/length/rep-4 operating curve, never a CHAIR column alone.**

## 8. Current reading & next steps

- ✅ PTQ validation honest and usable: rotation (+GPTQ) is the credible W4A4 recipe; AWQ/GPTQ-as-implemented need a methods note (checks "??").
- ✅ Evidence direction at n=100: **omission** (small objects), fallback only at w4a4:rot; collapse trigger = **text-token activation quantization** (fits the outlier/massive-activation account; replicates on Qwen2.5-VL, §9.5).
- ◐ LoRAS: partial ceiling (~0.6), large error reductions on K (≈40%) but the 2×2 shows no CHAIR gain at n=100 — the corrector may need the *right* degradation regime (W4A4-with-rotation, where error is bigger) or the LoRAS-T text variant (which roughly **halves object drops** in Gate 2: w3a16+lorast 0.080–0.096 vs w3a16 0.158, at the cost of slightly longer captions).
- ◐ A-CAB: only α≈0.5 shows an honest signal; needs a proper α sweep with coverage constraints and a larger n.
- ⏳ Next: larger-n confirmation of Gate-1 verdicts (500 images), then A-CAB α sweep at n=500 with the gated-frac/coverage constraints, and the LoRAS × regime interaction (does LoRAS help where it has ceiling headroom?).

## 9. Qwen2.5-VL-7B replication (2026-10-25)

> [!info] Source
> `~/Downloads/qwen_results.tgz` → compact artifacts in
> `obsidian-docs/results/qwen-2.5vl-2026-10-25/`. Model:
> `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, 28 layers, 28/4 GQA, 196 wrapped Linears
> (6.53B params). Entry point: `run_all_qwen.sh` (same stages as LLaVA).
> **Single cross-model table: [[Results Comparison]].**
> Gate 2 complete (13 rungs, 300 captions, 100 POPE images); Gate 1 **16/17**
> (`w3a16:method=gptq:calib=text` fails with a non-positive-definite Hessian, also at
> `damp=0.1`); diagnostic / attribution ladder complete. The five late Gate-1 rungs
> and the diagnostic ladder come from the second pass (`qwen_results_v2.tgz`).

### 9.1 Baseline is not the same model

| FP16 | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---:|---:|
| Coverage | 0.744 | 0.559 |
| CHAIR_i | 0.220 | 0.109 |
| CHAIR_s | 0.390* | 0.250 |
| POPE-F1 (3-split mean) | 0.854 | 0.833 |
| POPE yes-ratio | 0.542 | 0.368 |
| Caption length | 92.5 words | 106.9 words |
| WikiText-2 PPL | 7.27 | 8.28 |

\* LLaVA CHAIR_s from the 100-image evidence run; other LLaVA numbers from Gate 2.
Qwen is more conservative at FP16 (lower coverage and CHAIR, higher precision /
lower recall), so **absolute scores cannot be compared across models**; only paired
deltas are meaningful.

**Gate 1 (16/17, 100 + 100, paired CIs; full table in [[Results Comparison]] §8).**
Weight-only RTN/AWQ-wiki/GPTQ-wiki rungs lose coverage (omission), as on LLaVA:
W3 RTN −0.090*, W3 GPTQ-wiki −0.056*, W3 AWQ-wiki −0.049*, W4 AWQ-wiki −0.038*.
W8A8 and W8A8+SQ are ok. The per-rung verdicts differ from LLaVA: W4A16 RTN is
**ok** (LLaVA: fallback); NF4 (F1 +0.018*), W3 GPTQ-mm and W4A8+SQ (F1 +0.014*) move
in the **fallback** direction (more "yes", higher recall); `w4a4:rot` loses coverage
(−0.070*) but not POPE (+0.005; LLaVA's only significant cell was −2.0); `rot+gptq`
loses **6.3 POPE-F1 points*** and CHAIR_i −0.035* (omission), reproducing Gate 2
exactly. Object drops: NF4 0.148, W4A16 0.139, W3 RTN 0.230, `w4a4:rot` 0.213 (LLaVA
0.064 / 0.098 / 0.160 / 0.086); small-minus-large is positive on all 15 measurable
rungs (CI excludes 0 on 13).

### 9.2 Gate 2 — omission and small-object tail replicate

Paired deltas vs FP16 (`*` = 95% image-clustered CI excludes 0):

| Rung | LLaVA ΔCOV | LLaVA ΔCHAIR_i | Qwen ΔCOV | Qwen ΔCHAIR_i | Qwen ΔF1 | Qwen direction |
|---|---:|---:|---:|---:|---:|---|
| W4A16 GPTQ (wiki) | −0.018* | −0.026* | +0.015 | −0.007 | −0.007 | none |
| W3 RTN | −0.072* | −0.035* | **−0.061*** | −0.004 | +0.003 | omission |
| W3 GPTQ (wiki) | −0.045* | −0.028* | **−0.046*** | +0.006 | −0.017 | omission |
| W3 GPTQ (mm) | −0.025* | −0.036* | −0.004 | +0.019 | +0.016 | fallback |
| W3 AWQ (wiki) | −0.042* | −0.014 | −0.025* | +0.009 | +0.004 | omission |
| W4A4 rot+GPTQ | −0.002 | −0.001 | −0.019 | −0.013 | **−0.063*** | omission |
| W3 noise | −0.097* | −0.046* | **−0.042*** | +0.004 | −0.036* | omission |
| W4 noise | −0.027* | −0.030* | −0.009 | +0.010 | −0.007 | none |
| W3 weights @ image | −0.008 | −0.006 | +0.004 | +0.005 | −0.012 | none |
| W3 weights @ text | −0.069* | −0.040* | **−0.059*** | −0.006 | −0.016 | omission |

Object drop rates (share of FP16-mentioned objects the rung no longer names):

| Model | FP16 mention <1% / >20% | W3 RTN <1% / >20% / all | W3 GPTQ-wiki all | W3 text-weights all | LoRAS-T all |
|---|---|---:|---:|---:|---:|
| LLaVA | 0.608 / 0.923 | 0.279 / 0.061 / 0.158 | 0.130 | 0.154 | 0.080 |
| Qwen | 0.319 / 0.838 | 0.360 / 0.118 / 0.205 | 0.167 | 0.172 | 0.110 |

The Qwen result has the same signature: coverage falls without a CHAIR rise, small
objects go first, matched noise reproduces it, and **text-position weights drive the
loss while image-position weights do not**. The realistic W4A4 row differs from
LLaVA: Qwen’s `rot+gptq` loses 6.3 POPE-F1 points, so POPE is not uniformly blind on
this model.

### 9.3 2×2, LoRAS, LoRAS-T, A-CAB

| Cell (W4A8) | LLaVA cov | LLaVA CHAIR_i | Qwen cov | Qwen CHAIR_i |
|---|---:|---:|---:|---:|
| base | 0.771 | 0.202 | 0.545 | 0.103 |
| + LoRAS | 0.756 | 0.215 | **0.574** | 0.099 |
| + A-CAB | 0.753 | 0.220 | 0.556 | 0.114 |
| + both | 0.720 | 0.221 | 0.552 | 0.092 |

- **KV-LoRAS:** Qwen ceilings are much lower (K/V 0.22–0.37; rank-16 removes
  10–21%) than LLaVA’s 0.58–0.67. The 2×2 nevertheless shows **+2.9 coverage
  points** (0.545→0.574), CHAIR_i −0.005 [−0.039,+0.030], CHAIR_s +0.060
  [−0.020,+0.140]. No coverage CI is in the artifact, so treat this as provisional.
  Drift selects layers **[17, 20–26]** (LLaVA: [23, 25–31]).
- **LoRAS-T:** W3 RTN + LoRAS-T reaches coverage −0.009 (n.s. vs FP16; **+5.2
  points vs W3 RTN**), CHAIR_i +0.007 (n.s.), small-object drop 0.360→0.209, all
  0.205→0.110, 12.9M params (rank 64, all 28 layers). Held-out KL 0.276→0.071,
  top-1 0.729→0.872. This closely matches LLaVA (−0.012 coverage, small 0.28→0.14,
  all 0.16→0.08) and is the strongest cross-model result.
- **A-CAB:** baseline (60 captions) CHAIR_i 0.091, coverage 0.580. Best CHAIR is
  τ60/α1: 0.078, but coverage 0.558 (−2.2 points). Higher α worsens CHAIR; unlike
  LLaVA there is no brevity collapse (length stays ~106, rep-4 ≤0.006), but there
  is no coverage-preserving win. LLaVA’s honest point remains α0.5/τ40.

### 9.4 Quantizer health (WikiText-2 PPL)

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

Rotation still rescues W4A4 on Qwen; SmoothQuant W8A8 is near FP16. Unlike LLaVA,
the published-method checks pass on Qwen: GPTQ closes 72% (W4) / 56% (W3) of RTN's
gap and AWQ 31% at W4; only AWQ@W3 fails (15%, expect ≥25%).

### 9.5 Activation attribution and token probe

Fine ladder (`--preset diagnostic`, 60 POPE / 40 captions):

| Rung | LLaVA | Qwen |
|---|---|---|
| W4A8 | ok | ok (F1 −0.001) |
| W4A6 | ok | **degraded, LM damaged** — F1 −0.116*, recall −0.157*, COV −0.080*, CHAIR_i +0.069* |
| W4A5 | degraded (fallback candidate) | **BROKEN** — 99% unparseable, 6-word captions |
| W4A4 | BROKEN (71%) | BROKEN (100%) |
| image @A4 only (`abits_text=8`) | degraded, LM intact | degraded, LM intact — F1 −0.300*, COV −0.134*, omission |
| text @A4 only (`abits_image=8`) | **BROKEN (81%)** | **BROKEN (100%)**, identical to W4A4 |

The collapse trigger replicates: **text-position A4 activations break both models.**
Qwen is far less tolerant of low-bit activations (broken at A5, degraded at A6; image-
only A4 still costs 30 POPE-F1 points). The token probe (16 images) explains why: text
positions carry the largest outliers (down_proj input max |x| 2288 vs 310 for image
tokens; attention input 109.5 vs 50.8), while per-token int4 rounding zeroes ~90–97%
of attention-input and down_proj-input entries for both groups. Image tokens lose
more entries at attention outputs (0.76 vs 0.26). Blind fluency: FP16 PPL 146.3,
W4A8 117.9, **W4A4 30,260** (14-word, repetitive output) — naive W4A4 breaks Qwen's
LM with no image, as on LLaVA (143 vs 15.9).

### 9.6 Latency (W4A8, simulated, H100)

Baseline TTFT 147 ms, decode 17.4 tok/s, peak VRAM 15.7 GB. Decode overhead: LoRAS
+4.6%, A-CAB add (prev-gate) +8.6%, A-CAB two-pass +32.6%, A-CAB mult (eager) +27.1%,
LoRAS + A-CAB +13.9%; TTFT +6% with LoRAS, unchanged with A-CAB. Fake-quant runs in
bf16, so only the relative overheads transfer.

### 9.7 Caveats

- **Do not use Qwen Gate-2 blind/caption PPL.** FP16 blind PPL is 146 vs WikiText
  8.28; this is consistent with token-boundary misalignment when a caption is
  appended to a chat-template prompt (the scored BPE tokens shift). Use
  `validate_ptq.json`; the `lm_damaged` flags are provisional.
- **No Qwen attention / `r_pb` measurement** in this archive, so the LLaVA
  attention-drop mechanism has no cross-model counterpart yet.
- **Gate 1 is 16/17**: GPTQ with text calibration fails on Qwen's 18,944-wide
  down_proj Hessian even at `damp=0.1`; it needs a Cholesky fallback (e.g. float64
  or adaptive damping), not just more damping.
- Qwen `lm_damaged` flags come from the blind PPL (see first caveat) and are
  provisional, although the blind measure is self-consistent for gross failures
  (identical values for `w4a4` and the text-at-A4 arm).
- Qwen’s FP16 baseline is more conservative; report deltas and state the baseline.

## Related Notes

- [[Study Experiment]] — the evidence study this builds on
- [[Research Ideation]] — hypotheses H1–H4, gaps, mechanism
- [[Results]] — evidence-study results log
- [[Claims & Evidence Chain]] — literature spine (Claim 4 = the gap this phase fills)
- [[README]] — vault index