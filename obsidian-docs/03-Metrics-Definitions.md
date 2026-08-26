---
title: Metrics & Statistical Protocol
date: 2026-08-25
tags:
  - research/methodology
  - experiment/metrics
status: in-progress
---

# Metrics & Statistical Protocol

## 1. Hallucination Metrics (task-level)

### 1.1 CHAIR (Rohrbach et al. 2018, MLLM-era 500-image protocol)

Let $G$ be the set of MSCOCO 80 object classes present in the ground truth (instance segmentations ∪ reference captions), and $M(c)$ the set of object classes mentioned in caption $c$.

**Sentence-level** (fraction of captions containing ≥1 hallucinated object):

$$
\text{CHAIR}_s = \frac{\left| \left\{ c : M(c) \setminus G \neq \varnothing \right\} \right|}{\left| \left\{ c \right\} \right|}
$$

**Mention-level** (fraction of object mentions that are hallucinated):

$$
\text{CHAIR}_i = \frac{\sum_c \left| M(c) \setminus G \right|}{\sum_c \left| M(c) \right|}
$$

Lower is better for both. Matching uses the official synonym list (e.g., "player" → "person", multi-word "hot dog" handled so "dog" is not double-counted) — mandatory for comparability with published numbers.

### 1.2 POPE (RUCAIBox)

Per-split (random / popular / adversarial), 3,000 yes/no questions: report Accuracy, Precision, Recall, **F1** (major metric), and **yes-ratio** (answer-rate of "yes"; drift upward signals over-trust):

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### 1.3 Reasoning & perception

- **ScienceQA**: multiple-choice accuracy (%), over the multimodal subset.
- **TextVQA**: VQA-style normalized accuracy (%).

These are *context* metrics: they show quantization hurts correctness overall, while CHAIR/POPE show the *hallucination-specific* damage (relative degradation is expected to be larger).

## 2. Attention Grounding Metrics (mechanism-level)

### 2.1 Visual token span

For model $m$ with a generated token at position $t$, let $\mathcal{V} \subset \{1 \ldots t-1\}$ be the visual-token positions:

- **LLaVA-1.5**: fixed prefix $\mathcal{V} = \{1 \ldots 576\}$ (CLIP ViT-L-14@336 patches).
- **Qwen2-VL**: positions between `<|vision_start|>` and `<|vision_end|>` special tokens — **per-input variable length**; resolved at prefill per image.

### 2.2 Per-token visual attention mass

For generated token $t$, in layer $l$ and head $h$, with attention weights $a^{(l,h)}_{t,i}$ over all keys $i$ (row-normalized, $\sum_i a^{(l,h)}_{t,i} = 1$):

$$
A^{(l,h)}_v(t) = \sum_{i \in \mathcal{V}} a^{(l,h)}_{t,i}
$$

**Aggregated grounding score** (headline grounding metric):

$$
\bar{a}_v(t) = \frac{1}{L \cdot H} \sum_{l,h} A^{(l,h)}_v(t)
$$

Layer/head ablations use per-$(l,h)$ values directly (supports S3, [[02-Evidence-Experiment-Design#Support Experiments]]).

### 2.3 Attention entropy (diffusion of visual focus)

Normalize over visual positions only: $p^{(l)}_{t,i} = a^{(l)}_{t,i} / A^{(l)}_v(t)$, then

$$
H_v(t) = -\sum_{i \in \mathcal{V}} p_{t,i} \log_2 p_{t,i}
$$

High $H_v$ = diffuse, unfocused visual attention (expected to rise with quantization; tests H2, [[01-Problem-and-Hypotheses#Hypotheses]]).

### 2.4 Stability / argmax drift

Let $i^*_t = \arg\max_{i \in \mathcal{V}} a_{t,i}$ be the most-attended patch at step $t$. Quantization noise should make $i^*_t$ jump around:

$$
\text{Drift}(t) = \mathbf{1}\left[ i^*_t \neq i^*_{t-1} \right]
$$

Reported as mean drift rate over the generation, and as patch-coordinate displacement $\lVert i^*_t - i^*_{t-1} \rVert_2$ for visualization.

### 2.5 Aggregations per cell

For each (model, precision, quantizer, seed):

- $\bar{A}_v$ = mean $\bar{a}_v(t)$ over all generated tokens (excluding prompt).
- **Step-window profile**: $\bar{a}_v$ per decile of generation length — the raw material for the fallback timeline (H4).
- **Mention-level**: for each noun/object mention $m$ in the caption (CHAIR-matched), the mean $\bar{a}_v$ over the mention's tokens, labeled grounded/hallucinated.

### 2.6 Distributional convergence to the language prior (S2b)

Same prompt $x$, with image $v$ vs image masked: $P_{img}(y \mid x, v)$ and $P_{txt}(y \mid x)$. Lexical fallback = the quantized distribution drifts toward the text-only distribution:

$$
\Delta\text{KL}_c = \text{KL}\big(P_c(\cdot \mid x, v) \,\|\, P_{txt}(\cdot \mid x)\big) - \text{KL}\big(P_{FP16}(\cdot \mid x, v) \,\|\, P_{txt}(\cdot \mid x)\big)
$$

Expected $\Delta\text{KL} < 0$ under fallback (convergence). Reported per step (over the vocabulary, at each decoding step) and aggregated per cell. Token-level companion: $P_{txt}(\text{hallucinated mention})$ high, $P_{txt}(\text{grounded mention})$ low.

## 3. Correlation & Statistical Protocol

### 3.1 Token-level coupling (tests H3)

Point-biserial correlation between mention-level grounding and hallucination label $Y \in \{0,1\}$:

$$
r_{pb} = \frac{\bar{a}_v(Y{=}1) - \bar{a}_v(Y{=}0)}{s_{\bar{a}_v}} \cdot \sqrt{p(1-p)}, \qquad p = P(Y{=}1)
$$

Reported per cell with 95% **bootstrap confidence intervals** (resample images, not tokens — tokens within a caption are dependent).

### 3.2 Binned grounding curve (attention → hallucination function)

Bin all mentions by $\bar{a}_v$ quantile (10 bins); plot hallucination rate per bin. Expect monotone decrease — the functional form of the attention↔hallucination relationship. Fit a logistic regression $\log \frac{p}{1-p} = \beta_0 + \beta_1 \bar{a}_v$ for a continuous view and report $\beta_1$ (negative, significant ⇒ hallucinated mentions are systematically less grounded).

### 3.3 Model-level monotonicity (tests H1)

Spearman rank correlation between precision level (ordered) and cell hallucination metric, across the matrix. Report per (model, quantizer) chain.

### 3.4 Pairwise significance

- **POPE**: McNemar's test on paired yes/no answers between FP16 and each precision cell (same images, same seeds).
- **CHAIR**: paired bootstrap over the 500 images; report $\Delta$CHAIR$_s$, $\Delta$CHAIR$_i$ with CIs.
- **Attention**: paired $t$-test / Wilcoxon on per-image $\bar{A}_v$ (FP16 vs quantized).

### 3.5 Multiple-comparison hygiene

Cells ≈ 14 × 3 seeds; declare significance at $\alpha = 0.05$ with **Benjamini–Hochberg** correction across the hypothesis family; report effect sizes (Cohen's $d$ for attention drop, risk ratio for hallucination rise).

## 4. Expected Results Under Each Hypothesis

| Hypothesis | Metric signature | Figure |
|---|---|---|
| H1 monotonicity | CHAIR_s/i and POPE-F1 strictly ordered FP16 < W8A8 < W4A8 < W4A4 | [[04-Visualization-Spec#F1]] |
| H2 attention degradation | $\bar{A}_v$ down, $H_v$ up, Drift up with precision drop | [[04-Visualization-Spec#F2]], [[04-Visualization-Spec#F5]] |
| H3 token coupling | $r_{pb} < 0$ significant; binned curve monotone | [[04-Visualization-Spec#F3]], [[04-Visualization-Spec#F4]] |
| H4 temporal fallback | attention decay slope steeper/earlier for quantized | [[04-Visualization-Spec#F2]] |

## Related Notes

- [[02-Evidence-Experiment-Design]] — where each metric is collected
- [[04-Visualization-Spec]] — every figure consuming these metrics
- [[06-Supporting-Evidence]] — metric protocol sources (POPE/CHAIR conventions)
- [[README]] — vault index