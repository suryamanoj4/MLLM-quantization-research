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
| W4 weights lower visual attention (H2) | ❌ Qwen attention **rises** (W4A16 +13.9%* POPE vs LLaVA −14.0%*) | — | §0.3 |
| Visual attention fades through the caption (H4) | ✅ | ✅ (last/first 10%: 0.40 vs 0.44) | §0.3 |
| Matched noise effect is draw-dependent | ✅ | ✅ (Qwen W3 draws −.042*, −.045*, +.008) | §0.1 |
| LoRAS-T's gain comes from text-side placement (Q1: text ≥ all at equal params) | ✅ | cost of all-positions differs: coverage on LLaVA, POPE on Qwen | §0.6 |

## 0.1 Paper Table 1 replication

Exact replication of the current paper table, with the same rows, values, stars and
dashes. `–` means **not run / not measured**, not zero.

| Setting | LLaVA Cov | LLaVA CHAIR_i | LLaVA yes-ratio | LLaVA AUROC | LLaVA small | LLaVA PPL∅ | Qwen Cov | Qwen CHAIR_i | Qwen F1 | Qwen PPL_wiki |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FP16 (absolute values)** | .744 | .220 | .542 | – | | 15.6 | .559 | .109 | .833 | 8.28 |
| NF4 (bitsandbytes kernels) | −.005 | +.016 | +.006 | −.001 | .11 | 1.01× | +.003 | −.006 | +.018* | 1.01× |
| W4A4 rot. + GPTQ (wiki) | −.002 | −.001 | +.007 | −.001 | .14 | 1.06× | −.019 | −.013 | −.063* | 1.21× |
| W4 GPTQ (wiki) | −.018* | −.026* | +.002 | .000 | .16 | 1.01× | +.015 | −.007 | −.007 | 1.02× |
| W3 GPTQ (img+cap) | −.025* | −.036* | +.033* | −.006* | .20 | 1.13× | −.004 | +.019 | +.016 | 1.36× |
| W3 AWQ (wiki) | −.042* | −.014 | +.061* | −.004 | .22 | 1.10× | −.025* | +.009 | +.004 | 1.24× |
| W3 GPTQ (wiki) | −.045* | −.028* | +.006 | −.007* | .24 | 1.09× | −.046* | +.006 | −.017 | 1.13× |
| W3 RTN | −.072* | −.035* | −.031* | −.001 | .28 | 1.08× | −.061* | −.004 | +.003 | 1.29× |
| *Matched noise: Gaussian weight noise with RTN's per-group error energy* | | | | | | | | | | |
| W4 noise | −.027* | −.030* | +.034* | −.003* | .13 | 1.09× | −.009 | +.010 | −.007 | 1.05× |
| W3 noise | −.097* | −.046* | +.009 | −.010* | .31 | 1.14× | −.042* | +.004 | −.036* | 1.29× |
| W3 noise, other draws | Cov −.069*, −.018; CHAIR_i −.066*, +.040* | | | | | | Cov −.045*, +.008; CHAIR_i +.001, −.001 | | | |
| *Position-selective quantization: W3 RTN weights at one position type, FP16 weights at the other* | | | | | | | | | | |
| image positions only | −.008 | −.006 | −.017* | +.001 | .13 | 1.00× | +.004 | +.005 | −.012 | 1.00× |
| text positions only | −.069* | −.040* | −.022* | .002 | .28 | 1.08× | −.059* | −.006 | −.016 | 1.29× |

> **Caption (as in the paper).** Study 2 on both models (300 caption images; 100 POPE
> images, 1,800 questions). Changes are quantized minus FP16 on the same items;
> `*`: paired 95% interval excludes 0. AUROC: of P(yes), pooled over POPE's three
> splits. Small: of the small annotated objects (<1% of the image) that FP16's caption
> names, the share the setting's caption no longer names. PPL∅: LLaVA blind
> perplexity; PPL_wiki: WikiText-2 perplexity; both relative to FP16, and equal to FP16
> or W3 RTN by construction in the last two rows, as the test has no image. Noise rows:
> first draw on each model (LLaVA, W4: a second draw gives −.018 coverage, n.s.).
> `–`: not run.
>
> PPL_wiki is corrected 1.25× → 1.21×. Qwen W4 noise second draw: Cov −.007 (n.s.).

### Dashes in this table

All Qwen dashes were filled by the v3 pass (2026-10-31, A100-80GB; `results/qwen-2.5vl-2026-10-25/`
v3 artifacts). Only one LLaVA dash remains.

| Cell(s) | Status |
|---|---|
| LLaVA FP16 **AUROC** | Still `–`. Reference row; computable from the existing LLaVA Gate-2 JSON, no GPU needed. |
| LLaVA FP16 **small** | Blank by design (a drop relative to FP16 itself). |
| Qwen NF4 **Cov / CHAIR_i / F1** | Filled: real bitsandbytes NF4 rung added to Qwen Gate 2 (`--ladder-extra bnb-nf4`). Matches simulated NF4 in Gate 1 (F1 +.018*). |
| Qwen W3 GPTQ (img+cap) **PPL_wiki** | Filled: 11.285 → **1.36×**. Worse than W3 RTN on WikiText (fitted to captions); identical to LLaVA's W3 GPTQ-mm WikiText ratio (9.869 / 7.27 = 1.36×). |
| Qwen W4 / W3 noise **PPL_wiki** | Filled: 8.666 → **1.05×**, 10.722 → **1.29×** — matched to RTN's text damage (W4 RTN ≈1.07×, W3 RTN 1.29×), so the noise control is calibrated. |
| Qwen W4A4 rot.+GPTQ (wiki) **PPL_wiki** | **Corrected 1.25× → 1.21×.** The old value came from the mm-calibrated model (`09_validate_ptq.py` default), not the wiki-calibrated one whose POPE/CHAIR are in the row. |
| Qwen "W3 noise, other draws" | Filled: `seed=1`, `seed=2` (seed 0 is the main row). Two of three draws give a significant omission (−.042*, −.045*), one none (+.008) — the same draw-dependence as LLaVA (−.097*, −.069*, −.018). Qwen W4 second draw: Cov −.007 (n.s.; first draw −.009). |

### 0.2 Qwen columns not in the paper table

Gate 2, Qwen, change vs FP16 (`*`: CI excludes 0). Small = drop rate of FP16-named objects
<1% of the image (`10_object_analysis.py`).

| Setting | Cov | CHAIR_i | F1 | yes-ratio | AUROC | small |
|---|---:|---:|---:|---:|---:|---:|
| NF4 (bitsandbytes) | +.003 | −.006 | +.018* | +.021* | −.000 | .19 |
| W4A4 rot.+GPTQ (wiki) | −.019 | −.013 | −.063* | −.044* | −.013* | .26 |
| W4 GPTQ (wiki) | +.015 | −.007 | −.007 | −.009* | +.000 | .16 |
| W3 GPTQ (img+cap) | −.004 | +.019 | +.016 | +.017* | −.001 | .17 |
| W3 AWQ (wiki) | −.025* | +.009 | +.004 | +.004 | +.003 | .25 |
| W3 GPTQ (wiki) | −.046* | +.006 | −.017 | −.010 | −.011* | .35 |
| W3 RTN | −.061* | −.004 | +.003 | +.007 | −.002 | .36 |
| W4 noise (seed 0 / 1) | −.009 / −.007 | +.010 / +.006 | −.007 / −.026* | −.004 / −.021* | +.002 / +.001 | .18 / .19 |
| W3 noise (seed 0 / 1 / 2) | −.042* / −.045* / +.008 | +.004 / +.001 / −.001 | −.036* / −.033* / −.010 | −.027* / −.023* / −.003 | −.000 / −.002 / +.004 | .25 / .29 / .27 |
| image positions only | +.004 | +.005 | −.012 | −.007 | −.001 | .12 |
| text positions only | −.059* | −.006 | −.016 | −.011 | −.003 | .33 |
| W3 + LoRAS-T (caption calib) | −.009 | +.007 | −.002 | +.002 | −.002 | .21 |
| W3 + LoRAS-T (caption + yes/no calib) | −.017 | +.002 | −.003 | −.001 | −.000 | .24 |

Qwen's FP16 small-object miss reference and the LLaVA "small" column are on the same definition, but
Qwen drops more of its own objects at every setting (cf. §8 object-drop table).

**LoRAS-T with yes/no calibration (`--qa-per-image 2`).** Held-out |ΔP(yes)| 0.027 → **0.014** with
the correctors (shift +0.017 → −0.003; KL 0.075, top-1 0.862), the Qwen counterpart of LLaVA's 0.023.
On the eval it changes nothing material relative to caption-only LoRAS-T: POPE was already at FP16 level
(W3 RTN F1 +.003), and both remove W3 RTN's significant coverage loss (−.061* → −.009 / −.017, n.s.).
At scale 0.5 both recover only partly (−.045*, −.039*).

### 0.3 Visual-token attention (H2/H4) on Qwen

`scripts/13_attention_probe.py` (new; the method-study port of the evidence study's
`AttentionTracker`, which is LLaVA-only). Same quantities: per generated token, attention mass on the
image tokens averaged over all layers and heads; entropy (bits) renormalised over image tokens; drift of
the most-attended image token. 100 POPE images × 3 splits (1,800 questions), 100 captions (256 tokens),
greedy, paired change vs FP16 with image-cluster bootstrap (2,000).

| Image attention vs FP16 | LLaVA POPE | **Qwen POPE** | LLaVA CHAIR | **Qwen CHAIR** |
|---|---:|---:|---:|---:|
| FP16 (absolute) | 10.3% | 8.0% | 10.0% | 9.1% |
| W8A8 | +2.0%* | **+7.7%*** [+6.2, +9.3] | +3.9%* | **−10.7%*** [−17.5, −3.3] |
| W4A16 (GPTQ) | **−14.0%*** | **+13.9%*** [+12.4, +15.4] | **−18.3%*** | **+8.8%*** [+3.5, +14.9] |
| W4A8 | −12.9%* | **+11.2%*** [+9.7, +12.6] | −17.5%* | +2.6% [−2.5, +8.6] |

Per POPE split (Qwen): W8A8 +5.6 / +8.7 / +9.0%; W4A16 +11.7 / +15.1 / +14.9%; W4A8 +8.4 / +12.5 /
+12.5% (random / popular / adversarial), all significant.

| Rung (Qwen) | Entropy POPE / CHAIR (bits; ceiling 8.36) | Drift POPE / CHAIR | CHAIR first → last 10% |
|---|---|---|---|
| FP16 | 6.51 / 6.43 | 0.537 / 0.524 | 0.131 → 0.052 (0.40) |
| W8A8 | 6.48 / 6.37 | 0.546 / 0.485 | 0.132 → 0.042 (0.32) |
| W4A16 (GPTQ) | 6.52 / 6.51 | 0.564 / 0.562 | 0.142 → 0.055 (0.39) |
| W4A8 | 6.51 / 6.48 | 0.546 / 0.559 | 0.140 → 0.050 (0.36) |

**Reading.** H2 does **not** replicate: W4 weights *raise* Qwen's attention to the image (+11–14% on
POPE, every split) where they lowered LLaVA's (−13–14%), and entropy does not move (LLaVA +0.8 bits).
H4 replicates: attention fades through every caption on both models (Qwen 0.40, LLaVA 0.44) and W4 does
not steepen it. So the omission / small-object pattern, which does replicate, occurs on Qwen **without**
an attention loss — reduced visual attention is a LLaVA-specific correlate, not the general mechanism;
this is consistent with the attribution result that text-position damage drives the omission.

Caveats: both models' rungs are simulated (fake-quant), but the recipes differ — see the rung
comparison below; raw entropies are not comparable across models (Qwen's image-token count varies,
mean ceiling 8.36 bits vs LLaVA's 9.17); one draw at n = 100; no r_pb (H3) on Qwen yet. W8A8's
CHAIR −10.7% has a wide CI and sits against +7.7% on POPE — treat it as unresolved.

**Rung recipes differ between the two attention studies** (LLaVA: `evidence-study/.../models/load.py`;
Qwen: `phoenix`, `13_attention_probe.py`). Scope is the same (decoder Linears only, lm_head FP16).

| Rung | LLaVA (evidence study) | Qwen (attention probe) |
|---|---|---|
| W8A8 | int8 **per-channel** RTN weights + int8 per-token acts | int8 RTN weights, **g128**, + int8 per-token acts |
| W4A16 | auto_gptq int4 g128, **act-order on, damp 0.1**, calibrated on 128 **COCO caption texts** | phoenix GPTQ int4 g128, act-order off, damp 0.01, 64 **WikiText** sequences |
| W4A8 | the **same GPTQ** weights as W4A16 + int8 acts (isolates the activation axis) | **RTN** int4 g128 weights + int8 acts (different weights from W4A16) |
| Hardware / dtype | T4, fp16, transformers 4.46 | A100, bf16, transformers 4.53 |
| Items | resampled 100 images (seed 42) | method-study eval pool, 100 POPE images/split + 100 captions (seed 0) |
| POPE tokens | ≤ 8 | ≤ 16 |

So the W4A16 rows are both GPTQ but with different calibration text and settings, and W4A8 is GPTQ on
LLaVA but RTN on Qwen. Running `13_attention_probe.py` on LLaVA with the Qwen recipe is the clean way to
tell a model difference from a recipe difference.

## 0.4 The three reviewer questions (answered, v3)

**1) Was Qwen's LoRAS-T calibrated with the yes/no questions, and what are its POPE numbers?**
Originally **no**: the v1 correctors (`runs_qwen/lorast`) used captions only (`qa_per_image=0`,
192 FP16 captions, rank 64, all 28 layers). v3 adds the **question-augmented** version
(`--qa-per-image 2`, otherwise identical; `runs_qwen/lorast_qa`), the counterpart of LLaVA's.
Held-out |ΔP(yes)|: 0.027 without → **0.014** with the correctors (LLaVA: 0.023).

| W3 RTN + LoRAS-T, scale 1.0 | random | popular | adversarial | mean F1 | ΔF1 vs FP16 | Δyes | ΔAUROC | ΔCov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 (reference) | 0.8378 | 0.8298 | 0.8314 | 0.8330 | — | — | — | — |
| captions only (v1) | 0.8346 | 0.8298 | 0.8298 | 0.8314 | −.002 | +.002 | −.002 | −.009 |
| **captions + yes/no (v3)** | **0.8362** | **0.8269** | **0.8276** | **0.8302** | **−.003** [−.016, +.011] | −.001 | −.000 | −.017 [−.039, +.002] |
| captions + yes/no, scale 0.5 | 0.8423 | 0.8403 | 0.8343 | 0.8390 | +.006 | +.004 | −.001 | −.039* |

All scale-1.0 deltas are n.s.; W3 RTN alone is ΔF1 +.003, ΔCov −.061*. Adding yes/no calibration
does not change the eval picture on Qwen: POPE stays at FP16 level and the W3 coverage loss is removed
either way.

**2) WikiText perplexity for W3 GPTQ (img+cap) and the noise settings?**
Measured in v3 (`validate_missing.json`, 40 × 2048-token windows, FP16 = 8.281):

| Setting | PPL_wiki | ratio | compare |
|---|---:|---:|---|
| W3 GPTQ (img+cap) | 11.285 | **1.36×** | W3 RTN 1.29×, W3 GPTQ-wiki 1.13×; LLaVA W3 GPTQ-mm 1.36× |
| W4 noise | 8.666 | **1.05×** | W4 RTN ≈1.07× |
| W3 noise | 10.722 | **1.29×** | W3 RTN 1.29× |
| W4A4 rot.+GPTQ (wiki) | 10.004 | **1.21×** | replaces the 1.25× that came from the mm-calibrated model |

The noise rungs match RTN's text-side damage, so they are a fair control; image+caption-calibrated
GPTQ is the worst W3 setting on WikiText, exactly as on LLaVA.

**3) Which noise seed?**
The main noise rows use **seed 0** (`QuantConfig.rot_seed = 0`; per-layer generator
`rot_seed * 1_000_003 + crc32(layer_name)`). v3 adds draws with the rung option `:seed=1/2`:

| Draw | W3 Cov | W3 CHAIR_i | W3 F1 | W4 Cov | W4 F1 |
|---|---:|---:|---:|---:|---:|
| seed 0 (table row) | −.042* | +.004 | −.036* | −.009 | −.007 |
| seed 1 | −.045* | +.001 | −.033* | −.007 | −.026* |
| seed 2 | +.008 | −.001 | −.010 | — | — |

Two of three W3 draws give a significant omission, one none — the same draw-dependence as LLaVA
(−.097*, −.069*, −.018). Report the noise rows as "usually omission", not as a fixed size.

## 0.5 Attention probe (`scripts/13_attention_probe.py`)

Results are in §0.3. **What it is:** a method-study port of the evidence study's `AttentionTracker`
(`evidence-study/.../experiment/decoding.py`, LLaVA-only: hard-codes `LlavaForConditionalGeneration`,
image token 32000, auto_gptq), so it runs on Qwen (and LLaVA) through `phoenix`.

**How it measures (same definitions as LLaVA §5.4 of [[Results]]):**
1. Load the rung exactly like the ladder (`load_model` → `apply_ptq` → `quantize_model`), switch to
   eager attention, greedy-decode one item at a time.
2. For every *generated* token (decode steps only; the prefill row belongs to the prompt), in each of
   the 28 layers × 28 heads take the attention row, sum the part on the image-token positions
   (`input_ids == image_token_id`, so Qwen's variable count is exact), average over heads and layers.
   Entropy = bits of that row renormalised over the image tokens; drift = share of (layer, head) whose
   most-attended image token changes between consecutive steps.
3. Average per item over its generated tokens; captions also keep the per-step series → 10 deciles.
4. Rung vs FP16: paired relative change (Σ rung − Σ FP16) / Σ FP16 over identical items, image-cluster
   bootstrap (2,000); POPE also per split.

**Run (v3):** fp16, w8a8, w4a16:method=gptq:calib=wiki, w4a8; 100 POPE images × 3 splits (1,800
questions, ≤16 tokens), 100 captions (≤256 tokens, as LLaVA); ~33–58 min per rung on A100-80GB.

```bash
python scripts/13_attention_probe.py --coco-root data/coco-mini \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 \
  --precisions fp16,w8a8,w4a16:method=gptq:calib=wiki,w4a8 \
  --out runs_qwen/attention --resume
```

**The POPE "yes" confound does not explain the reversal.** On LLaVA, saying "yes" pulls attention onto
the image ([[Results]] §5.3). Qwen's W4A16 says "yes" slightly *less* (Δyes −.009*), which would lower
attention, yet attention rose +13.9%; and captions (no yes/no) also rise (+8.8%*).

## 0.6 Q1 — LoRAS-T placement ablation (text vs all positions)

Same rank-64 corrector, same layers and calibration data (captions + 2 yes/no questions per
image), equal parameters; fitted at text positions (LoRAS-T) or at **all** positions, image
tokens included. W3 RTN base, Study-2 eval. Full write-up, setup and files:
`results/q1-lorast-positions-2026-10-31/README.md`.

| All minus text (paired) | LLaVA | Qwen |
|---|---:|---:|
| ΔCov | **−.016*** | −.002 |
| ΔF1 | +.000 | **−.031*** |
| Δyes-ratio | −.008 | **−.025*** |
| Δrecall | −.007 | **−.047*** |
| Small-object drop, text → all | .140 → .181 | .245 → .252 |
| Held-out KL after, text vs all | .031 vs .036 | .075 vs .089 |
| Yes/no \|ΔP(yes)\| after, text vs all | .023 vs .033 | .014 vs .023 |

vs FP16: LLaVA text ΔCov −.012 (n.s., the paper's value, reproduced), all −.029*; Qwen
text −.017 / F1 −.003, all −.019 / F1 −.034*.

**Conclusion: text ≥ all on both models.** Correcting image positions too never improves a
grounding metric and costs coverage on LLaVA and POPE on Qwen, while fitting held-out FP16
outputs less well despite 6–7.5× more tokens. LoRAS-T works because of where it corrects,
not because of its added capacity — consistent with the attribution result that
text-position damage drives the omission. One draw per variant, unadjusted CIs; the LLaVA
coverage gap is close to 0.

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
- **Qwen attention measured in v3** (`13_attention_probe.py`, [[Results Comparison]] §0.3): the LLaVA
  attention drop does **not** replicate (W4A16 +13.9%* on POPE); `r_pb` (H3) is still not measured.
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
