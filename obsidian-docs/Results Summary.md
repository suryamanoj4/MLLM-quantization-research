---
title: Results Summary — Evidence & Methods, Inferred
date: 2026-10-02
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
> The **one-place read** for everything measured so far. The [[Study Experiment]] evidence (H1–H4, 100-img run) and the [[Method Study]] countermeasures (LoRAS / A-CAB, probe runs) are synthesized here — what each study found, what they say *together*, and what the combined inference is for the paper. Source of truth for details stays in [[Results]] (append-only log) and [[Method Study]] (methods phase). Every number below is traceable to those notes and to `method-study/runs/` artifacts.

## The storyline (one paragraph)

Quantization does not linearly break hallucination — it changes its **shape**. With weight-only W4/W3 (real methods), LLaVA-1.5-7B still reads the image (POPE-F1 within ~1 pt, LM fluent) but systematically **omits objects — especially small ones** (2–3× the drop rate) — and its visual attention mass falls ~14% at W4A16 with the entropy of image attention rising ~0.8 bits. True *fallback* (the false-“yes” / invented-object regime) appears only when **activation** quantization enters the picture, and the collapse is triggered specifically by **image-token** 4-bit activations, not text-token ones — naive simulated W4A4 breaks the decoder entirely, while the realistic rotation+GPTQ recipe stays fluent and degrades gracefully. The evidence study confirms the grounding-attention coupling (hallucinated mentions get less visual attention) at every functioning precision — the signal a decoding-time intervention can use. The counters, however, are not free wins: at probe n, A-CAB only helps at small α (α≥2 just buys shorter, loopier captions: coverage 0.77→0.52, rep-4 0.01→0.31), and LoRAS's error is only ~60% linearly recoverable (K ceiling 0.67 / V 0.58) with no CHAIR gain yet at n=100 — except the text-side variant, which roughly halves the object drop on W3. Decoding choice itself moved CHAIR_s more (0.39 greedy vs 0.51 nucleus) than any W8/W4 rung — the strongest single lever measured so far remains **decoding-time**, which is exactly where GAD/A-CAB sit.

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
| Collapse trigger | **image-token** 4-bit activations | abits_text=8 (image@4) → 81% unparseable; abits_image=8 (text@4) → degraded, LM intact (CHAIR_i +0.056 [+0.001, +0.110]). Weight-side: text-position weights drive omission, image-position weights don't. |
| Noise vs quantizer identity | noise reproduces it | w3a16:noise drops 18.5% of objects vs 15.8–16.0% real — error energy, not method. NF4 sim == real bnb kernels (0.065 vs 0.066). |
| LoRAS (W4A8) | ◐ partial | Ceilings K 0.67 / V 0.58 → only ~half the error is linearly fixable. Rank-16 removes 42% (K) / 25% (V) of error, held-out ≈ train. No CHAIR gain in the 2×2 at n=100. LoRAS-T (W3) halves object drops (0.080–0.096 vs 0.158). |
| A-CAB (W4A8) | ◐ brevity trap | α=0.5: honest small win (CHAIR_i 0.188 vs 0.210, coverage 0.788). α≥2: CHAIR falls by *shorter loopier captions* (coverage 0.77→0.52, rep-4 →0.31). Fires ~35% of steps at τ-pct 60. |

---

## 2. What the two studies say together

### 2.1 The W4A4 reconciliation — a simulated-quantization artifact becomes a finding
The evidence study's W4A4 cell collapsed (near-degenerate output) under naive per-token int4 activations. The method study shows this collapse is a **toolchain artifact, not the model's regime**: with rotation (+GPTQ), W4A4 keeps the LM fluent (PPL 8.53) and degrades gracefully (omission). Two consequences: (a) the proposal's "largest drop at W4A4" predictions (H1/H2) **were never testable with the simulated rung** — the realistic W4A4 row is `rot+gptq`, and it should be folded back into the evidence ladder; (b) the collapse itself is *informative* — the fact that image-position (not text-position) 4-bit activations break the model is direct support for the LUQ entropy claim the whole project rests on.

### 2.2 Two failure shapes, not one
At n=100 the data describes **omission** (weight quantization: reads less, mentions fewer/smaller objects) and **fallback** (activation collapse / w4a4:rot: more linguistically-plausible wrong output). The evidence study's yes-ratio +1.2–1.7 pts on W4A16 is the fallback seed even in the weight axis, but the dominant weight-axis response is omission. The proposal should frame the phenomenon as a **shape-shifting degradation** — omission first (weights), then fallback (activations) — rather than a monotone hallucination curve. H1 as written (monotone CHAIR/POPE rise) is the least supported claim at n=100.

### 2.3 The mechanism chain is coherent
Weight noise degrades the K/V pathway the decoder uses to read the image: attention mass −14% (evidence, H2) ↔ text-position weight quantization driving object omission (method, Gate 2). H3's grounding↔attention coupling holds at every precision — so the *signal* for a decoding-time fix is available. And the collapse trigger (image-token activations) matches where attention is spent (576 visual tokens) — the fragility is exactly where the model's grounding lives.

### 2.4 Decoding is the strongest lever measured
Greedy vs nucleus moved CHAIR_s by 12 pts (0.39 vs 0.51) — more than any W8/W4 rung and more than LoRAS/A-CAB at their defaults. This is the single most important inference for the research direction: **attention-gated decoding (GAD/A-CAB, QA-CD, prefix filtering) targets the largest measured effect**, and H3 says the grounding signal it needs is present even when quantized.

### 2.5 The methods' honest status
- **LoRAS** will not fully fix activation error (ceiling ~0.6) — it is a *partial, cheap* corrector; its promise is in regimes with big linear error (W4A4-rotation, W3), where Gate 2 shows the text-side variant roughly halving object drops (0.096 vs 0.158 at w3a16) — but pairwise CHAIR at n=100 is still n.s.
- **A-CAB** only works within a tiny α window (≈0.5); past it, it's a brevity/length intervention in disguise. Any A-CAB claim must report coverage/length/rep-4 with CHAIR.
- Neither method's n=100 cell should appear in the proposal as a headline; the honest headline from the methods phase is the **diagnosis** (omission shape, small-object tail, image-token collapse trigger, noise attribution), which strengthens the evidence contribution rather than the countermeasure.

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
| collapse trigger | image-token A4 (81% unparseable) vs text-token A4 (degraded, CHAIR_i +0.056) | [[Method Study]] §4 |
| LoRAS ceilings (rank 16, W4A8) | K 0.67 / V 0.58; removes 42% K, 25% V | [[Method Study]] §6 |
| LoRAS-T (W3) object drop | 0.080–0.096 vs 0.158 (≈half) | [[Method Study]] §4 |
| A-CAB α=0.5 / α≥2 | CHAIR_i 0.188 w/ coverage 0.788 / brevity trap (rep-4 0.23–0.31) | [[Method Study]] §7 |

---

## 4. What is missing before the proposal gets results

- [ ] **Full-n confirmation** (500 images) of the evidence ladder — H1 verdict, per-split effect sizes, S2/S2a/S2b text-only probe, S3 layer profile (evidence harness supports it: POPE-500 + CHAIR-500).
- [ ] **Fold the realistic W4A4 row** (`rot+gptq`) into the evidence ladder — the H2 "largest drop at W4A4" prediction is currently untested.
- [ ] **Larger-n Gate-1 confirmation** of real-method verdicts (500 images), esp. w4a16 fallback-direction and w4a4:rot.
- [ ] **A-CAB α sweep with coverage/length constraints at n=500** (find the honest operating point; kill the brevity confound).
- [ ] **LoRAS × regime interaction** — apply where the ceiling has headroom (W4A4-rot / W3-text), not where the error is non-linear (W4A8 tail layers).
- [ ] Methods note for the write-up: AWQ/GPTQ re-implementation checks did not pass; rotation + WikiText-calibrated GPTQ are the trustworthy tools.

## Related Notes

- [[Results]] — append-only evidence log, entry template, source numbers
- [[Method Study]] — methods phase detail, per-rung tables, artifact paths
- [[Study Experiment]] — the protocol the evidence numbers come from
- [[Research Ideation]] — hypotheses H1–H4, probes S1–S3, metrics
- [[README]] — vault index