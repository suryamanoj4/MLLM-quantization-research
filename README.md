# Quantized Multimodal LLMs Say Less, Not More

### Locating and Repairing Text-Side Grounding Loss

**Team Phoenix — International Institute of Information Technology, Hyderabad (IIITH)**

> **Abstract.** Multimodal LLMs are increasingly compressed to 3 bits per weight and
> below, and the prevailing expectation is that quantization noise weakens visual
> evidence and makes models hallucinate. We test this on LLaVA-1.5 and Qwen2.5-VL with
> five standard quantizers and find the opposite: quantized models do not invent
> objects, they **omit** them. At 3 bits, captions name 6–7 points fewer of the objects
> in the image, small objects first, while hallucination metrics hold steady or improve
> and POPE does not move. Causal interventions locate the loss on the language side:
> quantizing weights only at image positions costs nothing, only at text positions
> everything, and restoring full-precision activations shows the loss building up
> across depth at text positions and peaking in the last MLPs. A closed-form, low-rank
> correction applied only at text positions removes the significant loss on both
> models; applied at every position, the same correction recovers less or breaks
> yes/no answering.

**Paper:** [`Phoenix-Final.pdf`](Phoenix-Final.pdf) · **Mid report:**
[`TeamPhoenix-Mid.pdf`](TeamPhoenix-Mid.pdf) · **Code:** this repository

## Contributions

1. **Quantized MLLMs say less, not more.** With 3-bit weights, captions keep their
   length but name up to **7.2 (LLaVA)** and **6.1 (Qwen)** points fewer of the objects
   present, small objects first. No weight-only quantizer significantly raises
   hallucination on either model, and POPE registers none of the loss.
2. **The loss enters at text positions.** Using 3-bit weights only while the decoder
   processes text tokens reproduces the full loss; using them only at image tokens
   causes none. The same holds for 4-bit activations: text-position A4 breaks both
   models, image-position A4 leaves LLaVA intact and Qwen fluent.
3. **It builds up across depth and peaks in the last MLPs.** Restoring FP16 activations
   and weights inside the quantized decoder shows that image positions carry almost
   none of the loss (≤4% LLaVA, ≤5% Qwen), that it accumulates layer by layer at text
   positions, and that it sits in MLP rather than attention weights: single MLPs
   reproduce **0.86 / 0.74** of the object-word loss versus **0.35 / 0.11** for
   attention. The last MLP alone reproduces **51%** on Qwen; LLaVA's penultimate MLP
   reproduces **46%**.
4. **Repairing it where it is.** **LoRAS-T**, a low-rank correction of the decoder's
   hidden states at text positions, fitted in closed form without gradients, removes
   the significant coverage loss of 3-bit RTN on both models (34 MB LLaVA / 26 MB
   Qwen). Applied at every position instead, with the same parameters and data, it
   recovers less coverage (LLaVA) or breaks yes/no answering (Qwen): **where the repair
   is placed matters**.

## Key results

| Finding | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---|---|
| FP16 coverage / CHAIR_i / POPE-F1 | .744 / .220 / .854 | .559 / .109 / .833 |
| W3 RTN Δcoverage | **−7.2*** | **−6.1*** |
| W3 RTN ΔCHAIR_i | −3.5* | −0.4 n.s. |
| Small-object drop at W3 (<1% vs >20%) | 28% vs 6% | 36% vs 12% |
| Weights at text positions only | −6.9* | −5.9* |
| Weights at image positions only | −0.8 n.s. | +0.4 n.s. |
| Matched-noise W3 Δcoverage | −9.7* | −4.2* |
| Realistic W4A4 ΔPOPE-F1 | −0.2 n.s. | **−6.3*** |
| W4 attention-mass change | **−14%** | **+14%** |
| LoRAS-T recovery of W3 coverage loss | **83%** | **72%** |
| W3 + LoRAS-T Δcoverage vs FP16 | −1.2 n.s. | −1.7 n.s. |
| Same corrector at all positions | 61% recovered, −2.9* coverage | 69% recovered, POPE recall −5.0, F1 −3.4 |

\* Paired 95% image-clustered bootstrap interval excludes zero. Full tables:
[`Results Comparison.md`](obsidian-docs/Results%20Comparison.md), including the
paper-table replica and the final-paper metrics.

**Omission, not hallucination.** Every 3-bit quantizer lowers coverage on LLaVA by
2.5–7.2 points and on Qwen by up to 6.1; CHAIR_i rises for no weight-only quantizer.
POPE-F1 stays within 2 points of FP16 for weight-only quantizers, while its yes-ratio
moves in both directions with precision/recall trading off and AUROC nearly unchanged:
POPE sees a moved threshold, not fewer objects seen. The only large POPE drop is
activation quantization (Qwen W4A4, −6.3).

**Small objects are lost first.** At 3 bits, 28% (LLaVA) and 36% (Qwen) of the small
objects that FP16 names disappear from the quantized captions, against 6% and 12% of
the large ones. The drop rate falls steadily with object size.

**This is not language-model damage.** The 3-bit quantizer that damages WikiText
perplexity most — GPTQ calibrated on images+captions (1.36× on both models) — loses the
least coverage (−2.5 LLaVA, −0.4 Qwen). On Qwen, AWQ has higher perplexity than GPTQ
(1.24× vs 1.13×) but loses half as much coverage. Grounding loss is not a proxy for
perplexity.

**Patching localises the loss.** With 100 teacher-forced FP16 captions per model,
restoring image positions recovers at most 4–5% of the object-word loss. Restoring the
text-position residual stream recovers a share that grows with depth; quantizing one
layer's MLP reproduces most of the loss. The damage is distributed across the decoder,
not introduced at one site.

**Attention is not the mechanism.** 4-bit weights lower LLaVA's image attention by
14% but raise Qwen's by 14%, while both models lose objects. Reduced attention is a
model/recipe-specific correlate, not a general explanation.

## Setup

- **Models:** LLaVA-1.5-7B (CLIP ViT-L/14, 576 image tokens, 32-layer Vicuna decoder)
  and Qwen2.5-VL-7B (dynamic-resolution ViT, ~390 image tokens per COCO image,
  28-layer decoder with GQA and M-RoPE). Only the decoder is quantized (~6.5B
  parameters in each model).
- **Quantizers:** NF4 (real bitsandbytes kernels for the reference row, simulated
  otherwise), RTN, GPTQ, AWQ, and rotation+GPTQ at W4A4, on one simulated-quantization
  substrate so methods differ only in their algorithm.
- **Protocol:** greedy captions (≤128 tokens) on 300 COCO val2014 images; POPE on
  1,800 questions over 100 images. All comparisons are paired, quantized minus FP16 on
  identical items, with image-level bootstrap CIs. Corrections are calibrated on a
  disjoint image pool.
- **Measures:** CHAIR_i and caption coverage, plus per-object drop rates by size; POPE
  F1, yes-ratio and AUROC; WikiText-2 perplexity for language health.

## Repository layout

| Path | Contents |
|---|---|
| [`evidence-study/`](evidence-study/README.md) | Evidence harness for the five-rung precision ladder on LLaVA-1.5-7B: FP16/W8A8/W4A16/W4A8/W4A4, POPE + CHAIR, per-step cross-modal attention, text-only prior probe. |
| [`method-study/`](method-study/README.md) | Shared method-study pipeline for LLaVA and Qwen2.5-VL: PTQ validation, Gate 1/2, object-size analysis, position-selective quantization, matched noise, patching, LoRAS/LoRAS-T, A-CAB, latency. Entry points: [`run_all.sh`](method-study/run_all.sh) (LLaVA) and [`run_all_qwen.sh`](method-study/run_all_qwen.sh) (Qwen). |
| [`method-study/experiments/`](method-study/experiments/README.md) | Standalone follow-ups with self-contained code: `patching_qwen/` (cross-precision causal patching) and `q1_lorast_positions/` (text vs all-position LoRAS-T). Each has its own `run.sh`. |
| [`obsidian-docs/`](obsidian-docs/README.md) | Research vault: hypotheses, study design, metrics, append-only results log, per-rung method detail, and cross-model synthesis. Start with [`Results Summary.md`](obsidian-docs/Results%20Summary.md) and [`Results Comparison.md`](obsidian-docs/Results%20Comparison.md). |
| [`Phoenix-Final.pdf`](Phoenix-Final.pdf) | Final paper. [`TeamPhoenix-Mid.pdf`](TeamPhoenix-Mid.pdf) is the earlier mid-evaluation report. |

## Reproduce

```bash
cd method-study

# LLaVA-1.5-7B
bash run_all.sh

# Qwen2.5-VL-7B
bash run_all_qwen.sh
```

Standalone experiments:

```bash
cd method-study/experiments/patching_qwen
bash run.sh                  # LLaVA; Qwen: MODEL=Qwen/Qwen2.5-VL-7B-Instruct DTYPE=bfloat16

cd ../q1_lorast_positions
bash run.sh                  # text vs all-position LoRAS-T
```

Both reuse `method-study/data/coco-mini` and the shared
`method-study/.venv-phoenix`; see each experiment's README for exact paths and flags.

## Results and documentation

- [`Results Summary.md`](obsidian-docs/Results%20Summary.md) — synthesis: verdict
  dashboard, cross-model inference, Qwen replication, Q1 and attention results.
- [`Results Comparison.md`](obsidian-docs/Results%20Comparison.md) — single LLaVA ↔
  Qwen comparison note, paper-table replica, reviewer questions and attention probe.
- [`Method Study.md`](obsidian-docs/Method%20Study.md) — per-rung detail: PTQ
  validation, Gate 1/2, object analysis, activation attribution, LoRAS/LoRAS-T, A-CAB,
  drift, token probe.
- [`Results.md`](obsidian-docs/Results.md) — append-only evidence log and Qwen run.
- [`Results/qwen-2.5vl-2026-10-25/`](obsidian-docs/results/qwen-2.5vl-2026-10-25/README.md) —
  Qwen artifacts (Gate 1/2, attention, LoRAS/LoRAS-T, A-CAB, token probe, latency).
- [`Results/q1-lorast-positions-2026-10-31/`](obsidian-docs/results/q1-lorast-positions-2026-10-31/README.md) —
  Q1 text-vs-all-position ablation.

## Limitations

- All quantizers except bitsandbytes NF4 run on a simulated substrate; the KV cache is
  not quantized.
- Two 7B MLLMs, COCO captions and greedy decoding. Larger models, other domains and
  other decoding strategies may behave differently.
- Grounding measures rely on COCO annotations and a synonym list; unannotated objects
  are invisible to the metric.
- Patching scores teacher-forced FP16 captions and single sites, not free generation or
  interactions between sites.
- LoRAS-T is distilled from FP16 and inherits FP16's errors.
- Future work should test real low-bit kernels and larger models, and turn the
  localization into quantizers that spend precision on the text side and the last MLPs.
