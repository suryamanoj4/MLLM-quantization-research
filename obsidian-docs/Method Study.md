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

**Code:** [`method-study/`](../method-study/) — pipeline scripts `00_…–12_…`, entry point `run_all.sh`, artifacts in `method-study/runs/`. Environment: torch 2.6.0+cu124, transformers 4.53.3, CUDA 12.4, LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`), mini-COCO (`data/coco-mini`, 500 eval / 256 calib images, disjoint splits enforced).

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
- **Fine ladder (`runs/ladder/`, 60 POPE / 40 CHAIR):** w4a8 ok, w4a6 ok, **w4a5 degraded — fallback-regime candidate**; w4a4 **BROKEN** (71% unparseable POPE; coverage 0.01 vs 0.76; captions 21 vs 88 words; `clip=0.999` makes it worse, `skip=down_proj` still broken). **Attribution arms:** `abits_text=8` (image tokens at 4 bits) → **BROKEN (81%)**; `abits_image=8` (text tokens at 4 bits) → **degraded, LM intact** (CHAIR_i +0.056 [+0.001,+0.110]) — **image-position 4-bit activations are the collapse trigger**, text-position 4-bit activations are survivable. This is direct support for the LUQ-style claim that visual tokens carry the fragility.

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
- ✅ Evidence direction at n=100: **omission** (small objects), fallback only at w4a4:rot; collapse trigger = **image-token activation quantization** (fits the LUQ entropy story).
- ◐ LoRAS: partial ceiling (~0.6), large error reductions on K (≈40%) but the 2×2 shows no CHAIR gain at n=100 — the corrector may need the *right* degradation regime (W4A4-with-rotation, where error is bigger) or the LoRAS-T text variant (which roughly **halves object drops** in Gate 2: w3a16+lorast 0.080–0.096 vs w3a16 0.158, at the cost of slightly longer captions).
- ◐ A-CAB: only α≈0.5 shows an honest signal; needs a proper α sweep with coverage constraints and a larger n.
- ⏳ Next: larger-n confirmation of Gate-1 verdicts (500 images), then A-CAB α sweep at n=500 with the gated-frac/coverage constraints, and the LoRAS × regime interaction (does LoRAS help where it has ceiling headroom?).

## Related Notes

- [[Study Experiment]] — the evidence study this builds on
- [[Research Ideation]] — hypotheses H1–H4, gaps, mechanism
- [[Results]] — evidence-study results log
- [[Claims & Evidence Chain]] — literature spine (Claim 4 = the gap this phase fills)
- [[README]] — vault index