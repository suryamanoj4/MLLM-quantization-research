# Quantized Multimodal LLMs Say Less, Not More

### Text-side grounding loss and a closed-form repair

**Team Phoenix — International Institute of Information Technology, Hyderabad**

> **Current finding.** We started from the hypothesis that quantization corrupts visual
> tokens and makes MLLMs fall back on language priors, inventing objects that are not
> there (**Lexical Fallback**). On LLaVA-1.5-7B and Qwen2.5-VL-7B the opposite happens:
> quantized models **omit objects that are present**, especially small ones, while
> inventing no more (often fewer). The damage is on the **language side that reads out
> the visual tokens**, not in the stored image tokens. A closed-form low-rank correction
> at text positions, **LoRAS-T**, restores most of the lost grounding at 3-bit weights.

## Main results

- **Omission, not hallucination.** With W3 weights, caption coverage falls by **−7.2
  points on LLaVA** and **−6.1 points on Qwen**, while CHAIR_i does **not rise** and
  POPE-F1 stays near FP16. The standard POPE benchmark largely misses the loss.
- **Small objects go first.** At W3, LLaVA drops **27.9%** of small (<1% image area)
  objects that FP16 names versus **6.1%** of large (>20%) ones; Qwen drops **36.0%**
  versus **11.8%**. The same tail appears under matched Gaussian weight noise.
- **The failure is text-side.** Quantizing only text-position weights reproduces the
  coverage loss (LLaVA −6.9, Qwen −5.9 points); quantizing only image-position weights
  does not (LLaVA −0.8 n.s., Qwen +0.4 n.s.). 4-bit **text-token** activations are the
  collapse trigger; image-token A4 is survivable.
- **LoRAS-T repairs it.** A rank-64 closed-form correction of text-position hidden
  states (16.9M / 12.9M parameters) brings W3 RTN back to FP16 level: coverage
  −0.012 n.s. on LLaVA and −0.017 n.s. on Qwen with the question-augmented corrector;
  small-object drops fall 0.279→0.140 and 0.360→0.245. Fitted in ~30–40 min without
  gradients.
- **Placement matters, not capacity.** Q1 fits the *same* corrector at all positions,
  image tokens included. It never improves a grounding metric at equal parameters and
  costs coverage on LLaVA (all−text ΔCov **−0.016***) or POPE on Qwen (ΔF1 **−0.031***).
- **A-CAB is not a clean win.** LLaVA has a narrow honest α≈0.5 point; higher α buys
  shorter, loopier captions. On Qwen there is no coverage-preserving CHAIR gain.
- **Attention is not a universal mechanism.** LLaVA's W4 attention mass falls ~14%,
  but Qwen's rises ~14% under its GPTQ recipe. The omission is robust; the attention
  correlate is model/recipe-specific.
- **W4A4 depends on the recipe.** Naive per-token A4 collapses both models. The realistic
  rotation+GPTQ recipe stays fluent, but Qwen still loses **6.3 POPE-F1 points**.

The original Lexical Fallback prediction — more false “yes”, more invented objects,
stable coverage — is **not supported** for weight-only quantization. It is replaced by
**H2, Omission**: coverage/recall fall without a CHAIR rise, and the loss localises to
text-position computation.

## Research questions

| RQ | Answer |
|---|---|
| **RQ1** Does deployed quantization change grounding beyond POPE-F1? | Yes: coverage falls up to 7.2 points at 3 bits while POPE-F1 stays within 0.02. |
| **RQ2** Is the change visual and quantization-specific, or generic weight error? | Generic weight error in magnitude (matched noise reproduces it), but grounding-specific in shape and location. |
| **RQ3** Does it enter at image or text positions? | Text positions: text-only weight quantization reproduces the loss; image-only does not. |
| **RQ4** Can a training-free correction remove it? | Yes, largely: LoRAS-T restores W3 grounding to FP16 level; the placement ablation confirms text-side correction is what matters. |

## Repository layout

| Path | Contents |
|---|---|
| [`evidence-study/`](evidence-study/README.md) | **Evidence study (H1–H4):** harness for the FP16→W8A8→W4A16→W4A8→W4A4 ladder on LLaVA-1.5-7B, POPE + CHAIR with per-step cross-modal attention capture and a text-only prior probe. |
| [`method-study/`](method-study/README.md) | **Method study (C2/C3):** shared pipeline for LLaVA-1.5-7B and Qwen2.5-VL-7B — NF4/AWQ/GPTQ/SmoothQuant/QuaRot-style quantization, Gate 1/2, object-size analysis, LoRAS/LoRAS-T, A-CAB, drift profiling and latency. Entry points: [`run_all.sh`](method-study/run_all.sh) (LLaVA) and [`run_all_qwen.sh`](method-study/run_all_qwen.sh) (Qwen). |
| [`method-study/experiments/`](method-study/experiments/README.md) | **Standalone follow-ups:** `patching_qwen/` (cross-precision causal patching — where in the network grounding is lost) and `q1_lorast_positions/` (text vs all-position LoRAS-T). Self-contained `phoenix` copies and `run.sh` entry points. |
| [`obsidian-docs/`](obsidian-docs/README.md) | Research vault: problem framing, hypotheses, study design, metrics, append-only results log, method-study detail, and the cross-model synthesis. Start with [`Results Summary.md`](obsidian-docs/Results%20Summary.md) and [`Results Comparison.md`](obsidian-docs/Results%20Comparison.md). |
| [`TeamPhoenix-Mid.pdf`](TeamPhoenix-Mid.pdf) | Mid-evaluation report (the paper draft this repo supports). |

## Run it

The main pipeline is shared; only the model/dtype/output defaults differ.

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
bash run.sh                  # Q1 placement ablation
```

Both reuse `method-study/data/coco-mini` and the shared
`method-study/.venv-phoenix`; see the per-experiment READMEs for exact paths and flags.

## Results and documentation

- [`Results Summary.md`](obsidian-docs/Results%20Summary.md) — one-place synthesis:
  verdict dashboard, cross-study inference, and the Qwen replication.
- [`Results Comparison.md`](obsidian-docs/Results%20Comparison.md) — single LLaVA ↔
  Qwen comparison note and an exact replica of **paper Table 1**.
- [`Method Study.md`](obsidian-docs/Method%20Study.md) — per-rung detail: PTQ validation,
  Gate 1/2, object analysis, LoRAS/LoRAS-T, A-CAB, drift, activation attribution.
- [`Results.md`](obsidian-docs/Results.md) — append-only evidence log and Qwen run.
- [`Results/qwen-2.5vl-2026-10-25/`](obsidian-docs/results/qwen-2.5vl-2026-10-25/README.md) —
  Qwen artifacts (Gate 1/2, attention, LoRAS/LoRAS-T, A-CAB, token probe, latency).
- [`Results/q1-lorast-positions-2026-10-31/`](obsidian-docs/results/q1-lorast-positions-2026-10-31/README.md) —
  Q1 text-vs-all-position ablation.

## Status

- Background research and fact verification: ✅
- Evidence-study design: ✅
- Evidence-study execution: ✅ 100-image, 5-rung ladder; full 500 pending
- Method study — LLaVA: ✅ PTQ validation, Gate 1/2, object analysis, LoRAS, LoRAS-T, A-CAB, latency
- Method study — Qwen2.5-VL: ✅ Gate 2 (13 core rungs; 19 with the v3 additions) plus
  v3 attention/noise/PPL runs; Gate 1 is **16/17** (text-calibrated GPTQ hits a
  non-positive-definite Hessian)
- Q1 placement ablation: ✅ LLaVA + Qwen
- Cross-precision patching: ✅ code and tests; results pending
- GAD / QA-CD / prefix-guided filtering: next research phase, not yet run

## Next

1. Full-n confirmation (500 images) of the evidence ladder and the key Gate-1 rungs.
2. Run the patching experiment on LLaVA and Qwen to localise the text-side failure by
   layer/component.
3. Extend LoRAS-T to a second model/regime and test rank/layer/calibration-size ablations.
4. Evaluate GAD / QA-CD / prefix filtering on top of the corrected or uncorrected W3/W4
   models, with coverage/length/rep-4 guards rather than CHAIR alone.
