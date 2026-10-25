---
title: Results Summary — Evidence & Methods, Inferred
date: 2026-10-25
tags:
  - research/results
  - project/log
aliases:
  - Results Summary
  - Synthesis
status: in-progress
---

# Results Summary — Where We Stand

> [!abstract] Purpose
> The **one-place read** for everything measured so far. The [[Study Experiment]] evidence (H1–H4, 100-img run) and the [[Method Study]] countermeasures (LoRAS / A-CAB, probe runs) are synthesized here — what each study found, what they say *together*, and what the combined inference is for the paper. Source of truth for details stays in [[Results]] (append-only log) and [[Method Study]] (methods phase). Every number below is traceable to those notes and to `method-study/runs/` artifacts. The Qwen2.5-VL replication (2026-10-25) is included as a cross-model check in §1.

## The storyline (one paragraph)

Quantization does not linearly break hallucination — it changes its **shape**. With weight-only W4/W3 (real methods), LLaVA-1.5-7B still reads the image (POPE-F1 within ~1 pt, LM fluent) but systematically **omits objects — especially small ones** (2–3× the drop rate) — and its visual attention mass falls ~14% at W4A16 with the entropy of image attention rising ~0.8 bits. True *fallback* (the false-“yes” / invented-object regime) appears only when **activation** quantization enters the picture, and the collapse is triggered specifically by **text-token** 4-bit activations, not image-token ones — naive simulated W4A4 breaks the decoder entirely, while the realistic rotation+GPTQ recipe stays fluent and degrades gracefully. The evidence study confirms the grounding-attention coupling (hallucinated mentions get less visual attention) at every functioning precision — the signal a decoding-time intervention can use. The counters, however, are not free wins: at probe n, A-CAB only helps at small α (α≥2 just buys shorter, loopier captions: coverage 0.77→0.52, rep-4 0.01→0.31), and LoRAS's error is only ~60% linearly recoverable (K ceiling 0.67 / V 0.58) with no CHAIR gain yet at n=100 — except the text-side variant, which roughly halves the object drop on W3. Decoding choice itself moved CHAIR_s more (0.39 greedy vs 0.51 nucleus) than any W8/W4 rung — the strongest single lever measured so far remains **decoding-time**, which is exactly where GAD/A-CAB sit. On **Qwen2.5-VL-7B** the omission/small-object/text-position pattern replicates, and LoRAS-T again recovers the W3 loss; A-CAB does not produce a coverage-preserving gain, and the attention comparison is still missing.

---

## 1. Verdict dashboard

### Evidence study — [[Results]] (LLaVA-1.5-7B, 5-rung simulated ladder, 100 img + CHAIR-100, 2026-09-29)

| Hypothesis / probe | Verdict | The numbers |
|---|---|---|
| **H1** — hallucination rises with precision loss | ⚠️ unresolved at n=100 | POPE-F1 shifts ≤1.1 pts vs ~±3 pt CI; CHAIR non-monotone (W8A8 0.46 worst, W4A16 0.32 best — both within ~±10 pt CI); W4A16 yes-ratio +1.2–1.7 pts (predicted direction, small). Needs the full 500. |
| **H2** — attention degradation | ✅ supported, **weight axis** | Visual attention mass vs FP16: W4A16 **−14.0%** (POPE), −18.3% (CHAIR); W4A8 −12.9% / −17.5%; W8A8 +2.0%. Entropy 5.49 → 6.29 bits. CIs exclude 0, same size on all 3 splits. |
| — axis attribution | weight rounding, not int8 act rounding | W4A16 → W4A8 leaves attention unchanged (0.0815 → 0.0823). |
| **H3** — token-level grounding coupling | ✅ at every functioning rung | r_pb = +0.19 … +0.27 on CHAIR mentions, all CIs > 0. Coupling persists under quantization; not strengthened. |
| **H4** — temporal fallback | ◐ decay is a baseline property | Attention decays through every caption at every precision (FP16 0.153 → 0.068, ratio 0.44); W4 lowers the whole curve, decay only marginally steeper (0.41–0.42). |
| **S2 / S2a / S2b** — prior attribution | ⏳ pending | Text-only probe never run. |
| **S3** — layer-depth profile | ⏳ pending at evidence-study level | Method study's drift profile ([23, 25–31]) is the closest proxy so far. |
| W4A4 cell | collapsed (simulated) | Naive per-token int4 activations → near-degenerate output; excluded from H1–H4. *Reconciled in §2.1.* |

### Method study — [[Method Study]] (real PTQ methods, A6000, probe n)

| Question | Verdict | The numbers |
|---|---|---|
| PTQ toolchain usable? | ⚠️ partially | Rotation rescues W4A4 (PPL 1000.8 → 10.64, and rot+gptq 8.53, 99.9% of the gap); SmoothQuant ✅ (7.31, +0.04). **AWQ/GPTQ re-implementations failed their checks** (GPTQ@W3 −39.2% of gap) — state in methods; WikiText-calibrated GPTQ@W3 is the good one (8.49 vs 9.87 for mm-calib). |
| Gate 1 — direction of real-method degradation | **omission dominates** | 11 degraded rungs; only w4a4:rot significantly drops POPE-F1 (−2.0 pts [−3.5, −0.4]); w4a16 (RTN) shows the *fallback* direction with LM intact; everything else: omission, LM intact. |
| What gets omitted? | small objects (2–3×) | w3a16: <1%-size objects drop 0.284, >20% drop 0.098; w4a16: 0.193 vs 0.024. fp16 itself mentions small objects in only 68.6% of captions. |
| Collapse trigger | **text-token** 4-bit activations | abits_image=8 (text@4) → 81% unparseable; abits_text=8 (image@4) → degraded, LM intact (CHAIR_i +0.056 [+0.001, +0.110]). Weight-side: text-position weights drive omission, image-position weights don't. |
| Noise vs quantizer identity | noise reproduces it | w3a16:noise drops 18.5% of objects vs 15.8–16.0% real — error energy, not method. NF4 sim == real bnb kernels (0.065 vs 0.066). |
| LoRAS (W4A8) | ◐ partial | Ceilings K 0.67 / V 0.58 → only ~half the error is linearly fixable. Rank-16 removes 42% (K) / 25% (V) of error, held-out ≈ train. No CHAIR gain in the 2×2 at n=100. LoRAS-T (W3) halves object drops (0.080–0.096 vs 0.158). |
| A-CAB (W4A8) | ◐ brevity trap | α=0.5: honest small win (CHAIR_i 0.188 vs 0.210, coverage 0.788). α≥2: CHAIR falls by *shorter loopier captions* (coverage 0.77→0.52, rep-4 →0.31). Fires ~35% of steps at τ-pct 60. |

### Qwen2.5-VL-7B replication — `results/qwen-2.5vl-2026-10-25/` (master table: [[Results Comparison]])

> [!info] Scope
> Qwen2.5-VL-7B-Instruct, bf16, 28 decoder layers, 28/4 GQA, 196 wrapped Linears
> (6.53B parameters). **Gate 2 is complete** (13 rungs, 300 captions, 100 POPE
> images); **Gate 1 is partial** (11/17 rungs — the run stopped at
> `w3a16:method=gptq:calib=text` with a non-positive-definite Hessian). No visual
> attention / `r_pb` measurement is present in the archive.

**Baseline first — Qwen is a more conservative model.** Absolute scores are not
comparable across models; compare deltas.

| FP16 metric | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---:|---:|
| Caption coverage | 0.744 | 0.559 |
| CHAIR_i | 0.220 | 0.109 |
| CHAIR_s | 0.390* | 0.250 |
| POPE F1 (3-split mean) | 0.854 | 0.833 |
| POPE yes-ratio | 0.542 | 0.368 |
| Caption length | 92.5 words | 106.9 words |
| WikiText-2 PPL | 7.27 | 8.28 |

\* LLaVA CHAIR_s from the separate 100-image evidence run; the rest are Gate-2.

**Gate 2, same rungs, paired deltas vs FP16** (`*` = 95% image-clustered CI excludes 0):

| Rung | LLaVA ΔCOV | LLaVA ΔCHAIR_i | Qwen ΔCOV | Qwen ΔCHAIR_i | Qwen ΔPOPE-F1 | Qwen direction |
|---|---:|---:|---:|---:|---:|---|
| W4A16 GPTQ (wiki) | −0.018* | −0.026* | +0.015 | −0.007 | −0.007 | none |
| W3 RTN | −0.072* | −0.035* | **−0.061*** | −0.004 | +0.003 | omission |
| W3 GPTQ (wiki) | −0.045* | −0.028* | **−0.046*** | +0.006 | −0.017 | omission |
| W3 GPTQ (mm) | −0.025* | −0.036* | −0.004 | +0.019 | +0.016 | fallback |
| W3 AWQ (wiki) | −0.042* | −0.014 | −0.025* | +0.009 | +0.004 | omission |
| W4A4 rot+GPTQ | −0.002 | −0.001 | −0.019 | −0.013 | **−0.063*** | omission |
| W3 matched noise | −0.097* | −0.046* | **−0.042*** | +0.004 | −0.036* | omission |
| W4 matched noise | −0.027* | −0.030* | −0.009 | +0.010 | −0.007 | none |
| W3 weights at image positions only | −0.008 | −0.006 | +0.004 | +0.005 | −0.012 | none |
| W3 weights at text positions only | −0.069* | −0.040* | **−0.059*** | −0.006 | −0.016 | omission |

**Omission concentrates on small objects on both models.** FP16 already mentions
fewer small objects on Qwen; quantization removes the same fragile tail.

| Model | FP16 mention rate (<1% / >20%) | W3 RTN drop (<1% / >20% / all) | W3 GPTQ-wiki all | W3 text-weights all | LoRAS-T all |
|---|---|---:|---:|---:|---:|
| LLaVA | 0.608 / 0.923 | 0.279 / 0.061 / 0.158 | 0.130 | 0.154 | 0.080 |
| Qwen | 0.319 / 0.838 | 0.360 / 0.118 / 0.205 | 0.167 | 0.172 | 0.110 |

**2×2 ablation at W4A8** (point estimates; Qwen CHAIR CIs are n.s.):

| Cell | LLaVA coverage | LLaVA CHAIR_i | Qwen coverage | Qwen CHAIR_i |
|---|---:|---:|---:|---:|
| base | 0.771 | 0.202 | 0.545 | 0.103 |
| + LoRAS | 0.756 | 0.215 | **0.574** | 0.099 |
| + A-CAB | 0.753 | 0.220 | 0.556 | 0.114 |
| + both | 0.720 | 0.221 | 0.552 | 0.092 |

Qwen LoRAS: ΔCHAIR_i −0.005 [−0.039, +0.030], ΔCHAIR_s +0.060 [−0.020, +0.140];
coverage +2.9 points (no CI in this artifact). A-CAB: ΔCHAIR_i +0.011
[−0.020, +0.039]; coverage +1.1. Both: ΔCHAIR_i −0.011 [−0.050, +0.028];
coverage +0.7. **Only LoRAS has a positive coverage signal, and it is provisional.**

**LoRAS-T transfers.** W3 RTN + LoRAS-T: coverage −0.009 (n.s. vs FP16; recovers
+5.2 points vs W3 RTN), CHAIR_i +0.007 (n.s.), small-object drop 0.360→0.209, all
0.205→0.110, 12.9M parameters (rank 64, all 28 layers). Held-out KL 0.276→0.071,
top-1 agreement 0.729→0.872. LLaVA’s comparable cell: coverage −0.012 (n.s.),
CHAIR_i +0.009, small drop 0.28→0.14, all 0.16→0.08. This is the strongest
cross-model result.

**A-CAB does not transfer as a clean win.** Qwen sweep baseline (60 captions):
CHAIR_i 0.091, coverage 0.580, length 106.3. Best CHAIR is τ60/α1: CHAIR_i 0.078
but coverage 0.558 (**−2.2 points**); higher α worsens CHAIR without the LLaVA
brevity collapse. LLaVA’s honest point was α0.5/τ40 (CHAIR_i 0.188 vs 0.210,
coverage 0.788 vs 0.770). Treat A-CAB as model-dependent and not yet a win.

**Quantizer health (WikiText-2 PPL; reliable):**

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

**Caveats that matter for the write-up:**

- Qwen Gate-2 **blind and caption PPL are not usable**: FP16 blind PPL ≈146 while
  WikiText-2 PPL is 8.28. The likely cause is token-boundary misalignment when a
  caption is appended to a chat-template prompt (the scored BPE tokens shift). Use
  `validate_ptq.json` for LM health; treat `lm_damaged` flags in
  `gate2_ladder_report.json` as provisional.
- **No Qwen attention measurement** yet, so the LLaVA attention-drop / `r_pb`
  mechanism cannot be compared.
- Qwen’s realistic W4A4 (`rot+gptq`) **does** move POPE-F1 (−6.3 points), unlike
  LLaVA; weight-only W3 still leaves POPE-F1 flat while coverage falls. The
  benchmark is not uniformly blind on Qwen.
- Gate 1 is incomplete (11/17); the missing activation/rotation rows are covered
  only by Gate 2’s `rot+gptq` cell.

---

## 2. What the two studies say together

### 2.1 The W4A4 reconciliation — a simulated-quantization artifact becomes a finding
The evidence study's W4A4 cell collapsed (near-degenerate output) under naive per-token int4 activations. The method study shows this collapse is a **toolchain artifact, not the model's regime**: with rotation (+GPTQ), W4A4 keeps the LM fluent (PPL 8.53) and degrades gracefully (omission). Two consequences: (a) the proposal's "largest drop at W4A4" predictions (H1/H2) **were never testable with the simulated rung** — the realistic W4A4 row is `rot+gptq`, and it should be folded back into the evidence ladder; (b) the collapse itself is *informative* — the fact that text-position (not image-position) 4-bit activations break the model is direct support for the outlier/massive-activation account the project rests on.

### 2.2 Two failure shapes, not one
At n=100 the data describes **omission** (weight quantization: reads less, mentions fewer/smaller objects) and **fallback** (activation collapse / w4a4:rot: more linguistically-plausible wrong output). The evidence study's yes-ratio +1.2–1.7 pts on W4A16 is the fallback seed even in the weight axis, but the dominant weight-axis response is omission. The proposal should frame the phenomenon as a **shape-shifting degradation** — omission first (weights), then fallback (activations) — rather than a monotone hallucination curve. H1 as written (monotone CHAIR/POPE rise) is the least supported claim at n=100.

### 2.3 The mechanism chain is coherent
Weight noise degrades the K/V pathway the decoder uses to read the image: attention mass −14% (evidence, H2) ↔ text-position weight quantization driving object omission (method, Gate 2). H3's grounding↔attention coupling holds at every precision — so the *signal* for a decoding-time fix is available. And the collapse trigger is at **text-token** activations (the outlier-bearing positions), while attention is spent on the 576 visual tokens — the fragility is in the language-side readout, not in the stored image tokens.

### 2.4 Decoding is the strongest lever measured
Greedy vs nucleus moved CHAIR_s by 12 pts (0.39 vs 0.51) — more than any W8/W4 rung and more than LoRAS/A-CAB at their defaults. This is the single most important inference for the research direction: **attention-gated decoding (GAD/A-CAB, QA-CD, prefix filtering) targets the largest measured effect**, and H3 says the grounding signal it needs is present even when quantized.

### 2.5 The methods' honest status
- **LoRAS** will not fully fix activation error (ceiling ~0.6) — it is a *partial, cheap* corrector; its promise is in regimes with big linear error (W4A4-rotation, W3), where Gate 2 shows the text-side variant roughly halving object drops (0.096 vs 0.158 at w3a16) — but pairwise CHAIR at n=100 is still n.s.
- **A-CAB** only works within a tiny α window (≈0.5); past it, it's a brevity/length intervention in disguise. Any A-CAB claim must report coverage/length/rep-4 with CHAIR.
- Neither method's n=100 cell should appear in the proposal as a headline; the honest headline from the methods phase is the **diagnosis** (omission shape, small-object tail, text-token collapse trigger, noise attribution), which strengthens the evidence contribution rather than the countermeasure.

---

## 3. Cheat-sheet — headline numbers

| Quantity | Value | Where |
|---|---|---|
| W4A16 visual-attention mass change | −14.0% [−14.7, −13.2] (POPE), −18.3% (CHAIR) | [[Results]] §5.3 |
| Attention entropy (FP16 → W4A16) | 5.49 → 6.29 bits | [[Results]] §5.3 |
| r_pb grounding↔attention | +0.19 … +0.27, CIs > 0 | [[Results]] §4 |
| POPE-F1 (avg over splits) | FP16 0.838 · W8A8 0.844 · W4A16 0.827 · W4A8 0.834 · W4A4 0.503 (chance) | [[Results]] §5.1 |
| decode-effect on CHAIR_s | 0.39 greedy vs 0.51 nucleus (+12 pts) | [[Results]] §4 |
| W4A4 real recipe | rot+gptq: WikiText PPL 8.53 (rtn 1000.8) | [[Method Study]] §1 |
| only significant POPE cell (n=100, real methods) | w4a4:rot −2.0 pts [−3.5, −0.4] p=0.015 | [[Method Study]] §2 |
| small-object drop (w3a16 vs fp16) | <1% size 0.284 vs >20% 0.098 | [[Method Study]] §3 |
| collapse trigger | text-token A4 (81% unparseable) vs image-token A4 (degraded, CHAIR_i +0.056) | [[Method Study]] §4 |
| LoRAS ceilings (rank 16, W4A8) | K 0.67 / V 0.58; removes 42% K, 25% V | [[Method Study]] §6 |
| LoRAS-T (W3) object drop | 0.080–0.096 vs 0.158 (≈half) | [[Method Study]] §4 |
| A-CAB α=0.5 / α≥2 | CHAIR_i 0.188 w/ coverage 0.788 / brevity trap (rep-4 0.23–0.31) | [[Method Study]] §7 |
| **Qwen FP16 baseline** | coverage 0.559 · CHAIR_i 0.109 · POPE-F1 0.833 · WikiText PPL 8.28 | `qwen-2.5vl-2026-10-25/` |
| **Qwen W3 RTN** | coverage −0.061 [−0.085, −0.037]; CHAIR_i −0.004 n.s.; small drop 0.360 vs large 0.118 | `qwen-2.5vl-2026-10-25/` |
| **Qwen text-position weights** | coverage −0.059*; image-position weights +0.004 n.s. | `qwen-2.5vl-2026-10-25/` |
| **Qwen W4A4 rot+GPTQ** | coverage −0.019 n.s.; **POPE-F1 −0.063*** | `qwen-2.5vl-2026-10-25/` |
| **Qwen LoRAS-T (W3)** | coverage −0.009 n.s. vs FP16 (recovers +5.2 vs W3); small drop 0.360→0.209; 12.9M | `qwen-2.5vl-2026-10-25/` |
| **Qwen KV-LoRAS** | ceilings 0.22–0.37 (vs LLaVA 0.58–0.67); 2×2 coverage +2.9 points, CHAIR_i −0.005 n.s. | `qwen-2.5vl-2026-10-25/` |
| **Qwen A-CAB** | best CHAIR_i 0.078 vs 0.091, but coverage 0.558 vs 0.580; no clean win | `qwen-2.5vl-2026-10-25/` |

---

## 4. What is missing before the proposal gets results

- [ ] **Full-n confirmation** (500 images) of the evidence ladder — H1 verdict, per-split effect sizes, S2/S2a/S2b text-only probe, S3 layer profile (evidence harness supports it: POPE-500 + CHAIR-500).
- [ ] **Fold the realistic W4A4 row** (`rot+gptq`) into the evidence ladder — the H2 "largest drop at W4A4" prediction is currently untested.
- [ ] **Larger-n Gate-1 confirmation** of real-method verdicts (500 images), esp. w4a16 fallback-direction and w4a4:rot.
- [ ] **A-CAB α sweep with coverage/length constraints at n=500** (find the honest operating point; kill the brevity confound).
- [ ] **LoRAS × regime interaction** — apply where the ceiling has headroom (W4A4-rot / W3-text), not where the error is non-linear (W4A8 tail layers).
- [ ] Methods note for the write-up: AWQ/GPTQ re-implementation checks did not pass; rotation + WikiText-calibrated GPTQ are the trustworthy tools.
- [ ] **Qwen attention / `r_pb` measurement** — the archive has no visual-attention probe, so the LLaVA mechanism result has no cross-model counterpart yet.
- [ ] **Fix Qwen caption/blind-PPL token alignment** and rerun the LM-health guard; current Gate-2 `lm_damaged` flags are provisional.
- [ ] **Complete Qwen Gate 1** (11/17): `w3a16:method=gptq:calib=text` hit a non-positive-definite Hessian; W8A8/SQ and the activation-attribution arms are missing.
- [ ] **Qwen LoRAS n-confirmation** — the +2.9 coverage points in the 2×2 have no paired CI in the artifact.

## Related Notes

- [[Results]] — append-only evidence log, entry template, source numbers
- [[Results Comparison]] — single LLaVA ↔ Qwen2.5-VL master table
- [[Method Study]] — methods phase detail, per-rung tables, artifact paths
- [[Study Experiment]] — the protocol the evidence numbers come from
- [[Research Ideation]] — hypotheses H1–H4, probes S1–S3, metrics
- [[README]] — vault index