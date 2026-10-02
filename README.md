# Decoding the Compressed Mind: Counteracting Lexical Fallback in Quantized MLLMs via Grounding-Aware Decoding

**Team Phoenix — International Institute of Information Technology, Hyderabad**

## Research Topic

Multimodal Large Language Models (MLLMs) extend language models with vision by projecting continuous visual tokens into an autoregressive decoder. Deploying them under constrained compute budgets requires aggressive **Post-Training Quantization (PTQ)** — but extreme precision reduction (e.g., W4A8/W4A4) disproportionately injects noise into cross-modal representations, inducing **Lexical Fallback**: a failure mode where the decoder loses visual fidelity and defaults to over-parameterized *linguistic priors*, producing severe object hallucinations.

Current PTQ frameworks (QSLaw, MQuant, QuaRot, LUQ) optimize hardware-level weight/activation scaling and outlier handling but remain **agnostic to generation dynamics**. The open question this project addresses: *can hallucination in quantized MLLMs be counteracted at decoding time, at zero training cost?*

## Background & Gaps

- **Quantization ↔ Decoding are decoupled** — existing PTQ optimizes reconstruction error, not downstream sampling behavior.
- **Generation heads are agnostic to noise** — quantized MLLMs still use sampling strategies designed for full-precision backbones.
- **Hallucination mitigations assume full precision** — contrastive-decoding methods (VCD, DoLa, OPERA) are benchmarked on uncompressed models and ignore low-bit PTQ noise profiles.
- **No hallucination-focused ablation across precision levels** — no published work measures hallucination + cross-modal attention across a precision grid (FP16 → W8A8 → W4A8 → W4A4) on the same data.

## Research Direction

1. **Evidence study (current phase)** — prove the phenomenon and mechanism: same-dataset ablation of hallucination (CHAIR, POPE) and cross-modal attention metrics across a precision grid on LLaVA-1.5 (7B/13B) and Qwen2-VL-7B, quantized with AWQ, GPTQ, and MQuant. Hypotheses: monotone hallucination rise with precision loss (H1), attention degradation (H2), token-level grounding↔hallucination coupling (H3), and attention collapse over decoding steps (H4).
2. **Grounding-Aware Decoding (GAD)** — a training-free decoding algorithm that monitors cross-modal attention at each autoregressive step and penalizes visually ungrounded candidates:

   $$\tilde{S}(y_t) = \log P_{LM}(y_t \mid y_{<t}, v) - \lambda \cdot \max(0, \tau - A(y_t, v))$$

   where $A(y_t, v)$ is the aggregated cross-modal attention over visual tokens and $\tau$ an adaptive threshold.
3. **Quantization-Aware Contrastive Decoding (QA-CD)** — contrast the target model against an over-quantized proxy (e.g., W2A4) that retains language fluency but loses visual conditioning, isolating the pure hallucination distribution.
4. **Prefix-Guided Vocabulary Filtering** — pre-filter high-entropy, context-irrelevant entity tokens during prefill to keep Time-to-First-Token low.

## Repository Layout

| Path | Contents |
|---|---|
| [`evidence-study/`](evidence-study/README.md) | **Evidence Study** (H1–H4): modular, reproducible harness for the precision-ladder ablation — FP16/W8A8/W4A16/W4A8/W4A4 on LLaVA-1.5-7B, POPE + CHAIR, per-step attention capture, text-only prior probe. `uv`-managed package; run via `uv run experiments` |
| [`method-study/`](method-study/README.md) | **Method Study** (Phoenix, Contributions 2 & 3): LoRAS (low-rank activation steering) + A-CAB (entropy-gated attention biasing) on realistically quantized (NF4/AWQ/GPTQ/SmoothQuant/QuaRot-style) LLaVA-1.5-7B — drift profiling, PTQ validation, 2×2 method ablation. Numbered pipeline scripts (`00_…–10_…`), entry point `run_all.sh` |
| `obsidian-docs/` | Research vault — problem framing, hypotheses, study design, metric definitions, supporting evidence, and the running results log |
| `README.md` | This file — global research summary |

Start with [`obsidian-docs/README.md`](obsidian-docs/README.md) for the vault index.

## Status

- Background research & fact verification: ✅ (see [`obsidian-docs/Claims & Evidence Chain.md`](obsidian-docs/Claims%20%26%20Evidence%20Chain.md))
- Evidence study design: ✅ (see [`obsidian-docs/Study Experiment.md`](obsidian-docs/Study%20Experiment.md))
- Evidence study execution: results logged in [`obsidian-docs/Results.md`](obsidian-docs/Results.md)
- Method study (LoRAS / A-CAB): in progress — see [`method-study/README.md`](method-study/README.md)
- GAD / QA-CD / prefix filtering: later research phases (documented in the vault as they begin)