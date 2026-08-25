---
title: Supporting Evidence — Verified Facts with Primary Sources
date: 2026-08-25
tags:
  - research/literature
  - project/evidence
status: verified
---

# Supporting Evidence

> [!info] How this note was produced
> Verified by a background research agent against primary sources (arXiv papers/HTML, official GitHub repos, HuggingFace model cards, official docs) on 2026-08-25. Every claim below carries its source. **Corrections to earlier assumptions are flagged.**

## Architecture Facts

### LLaVA-1.5 — projector + self-attention, no cross-attention
- Two-layer **MLP vision-language connector**; CLIP **ViT-L-14@336** vision encoder; **Vicuna-1.5** (LLaMA-2-based) LLM; 336² image → **576 visual tokens** concatenated into the LLM input; fusion via the LLM's own causal **self-attention**.
- "Attention to visual tokens" is measurable directly in LLM self-attention rows (columns 1–576).
- Published POPE-F1 (7B): 87.3 / 86.1 / 84.2; (13B): 87.1 / 86.2 / 84.5 (random/popular/adversarial) — our harness validation targets.
- Sources: https://arxiv.org/abs/2310.03744 · https://huggingface.co/liuhaotian/llava-v1.5-7b · https://github.com/haotian-liu/LLaVA

### Qwen2-VL — ViT + MLP merger + self-attention (no cross-attention either)
- 675M-parameter ViT (DFN init, 2D-RoPE) → **MLP merger compressing adjacent 2×2 tokens into one** → Qwen2 LLM; `<|vision_start|>`/`<|vision_end|>` special tokens; 224² image → **66 tokens into the LLM** (64 merged + 2 specials).
- **Dynamic resolution**: visual token count is input-dependent (min_pixels=100×28², max_pixels=16384×28²) ⇒ attention metrics must resolve the visual span per input.
- Official quantized checkpoints exist (Qwen2-VL-72B AWQ / GPTQ-Int4/Int8 on HF).
- Sources: https://arxiv.org/abs/2409.12191 · https://github.com/QwenLM/Qwen3-VL (repo redirect) · https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4

> [!warning] Correction
> Neither model family has cross-attention layers. "Cross-modal attention" in the literature = LLM self-attention over visual-token positions. Our attention term $A(y_t, v)$ is measured in self-attention.

## Benchmarks

### POPE (EMNLP 2023)
- Official repo: **https://github.com/RUCAIBox/POPE** (shikohome/POPE is a dead 404). Paper: arXiv:2305.10355.
- Protocol: **500 MSCOCO val2014 images** (>3 GT objects) × 6 questions (3 yes / 3 no, 1:1 GT:non-existent) = **3,000 questions** per split; splits: **random / popular / adversarial**; metrics Accuracy, Precision, Recall, **F1** (major) + yes-ratio.
- Also available as `pope` task in lmms-eval.

### CHAIR (Rohrbach et al., EMNLP 2018)
- Official repo: **https://github.com/LisaAnne/Hallucination** (`utils/chair.py`); paper arXiv:1809.02156.
- Matching: tokenize + singularize, map to the **80 MSCOCO segmentation-challenge classes** via a synonym list (Lu et al. 2018); GT = **union of instance segmentations and reference captions**.
- The paper's original protocol was the Karpathy test split (beam-5); the **500-image MSCOCO val2014, one detailed caption per image** protocol is the **MLLM-era community convention** (used by VCD, ReWEIGH, SelfVal, FocusMatters) — state it as such in methods.

### ScienceQA & TextVQA
- ScienceQA: arXiv:2209.09513, NeurIPS 2022, ~21k multimodal multiple-choice, accuracy; official site https://scienceqa.github.io.
- TextVQA: official repo (faverogian) is **dead** → cite https://textvqa.org + facebookresearch/mmf (LoRRA); standard VQA-style normalized accuracy.
- Both available in **lmms-eval** (`scienceqa_img`, `textvqa_val`).

### lmms-eval coverage (harness decision)
- Repo: https://github.com/EvolvingLMMs-Lab/lmms-eval (arXiv:2407.12772).
- Supports: **POPE, ScienceQA, TextVQA** + model wrappers for `llava`/`llava_hf` and `qwen2_vl`.
- **CHAIR is NOT among its tasks** ⇒ custom CHAIR harness required (on LisaAnne/Hallucination utils).

## Quantizers

### AWQ (MLSys 2024 best paper)
- Official repo: **https://github.com/mit-han-lab/llm-awq** — "supports instruction-tuned and multi-modal LMs"; LLaVA-1.5-7B requires **self-run AWQ search**; default 4-bit group size 128.
- ⚠️ **Weight-only (W4A16)** — no activation quantization ⇒ true W4A8/W4A4 needs MQuant-style pipelines.
- AutoAWQ (casper-hansen) supports `llava` and `qwen2_vl`, loadable via transformers `AwqConfig`.

### GPTQ
- **AutoGPTQ archived April 11, 2025** → maintained successor **GPTQModel** (https://github.com/ModelCloud/GPTQModel).
- Transformers integration: `GPTQConfig`, `modules_to_not_convert` to exclude the vision tower; Vicuna is `model_type=llama` so quantizes cleanly.

### MQuant (ACM MM '25) — headline W4A4/W4A8 tool
- Paper: **arXiv:2502.00425**; ACM MM 2025 DOI **10.1145/3746027.3755433**; official code **https://github.com/StiphyJay/MQuant** (released).
- Built on QuaRot Hadamard rotations + Modality-Specific Static Quantization (MSQ) + Attention-Invariant Flexible Switching (AIFS) + Rotation Magnitude Suppression (RMS).
- Reports **W4A8 and W4A4**; W4A4 on **LLaVA-1.5-13B** (ScienceQA +6.4% over Q-VLM); W4A8 near-lossless on Qwen2-VL-72B.
- ⚠️ Eval suite is TextVQA/DocVQA/OCRBench/MME/ScienceQA — **no POPE/CHAIR** ⇒ strong quantizer baseline but not a hallucination study. Research codebase, not a pip package.

### QuaRot
- Official repo: **https://github.com/spcl/QuaRot** (NOT IST-DASLab); arXiv:2404.00456.
- **LLM-only** (LLaMA-2 class, W4A4 incl. KV cache); MQuant states QuaRot "is not applicable to MLLMs due to inherent modality differences."

## LUQ (arXiv:2509.23729, TMLR 2026)

- Confirmed claims: (1) **multimodal token activations have significantly higher entropy than text tokens** (Qwen-2.5-VL analysis, K-means entropy estimation); (2) **sub-4-bit performance collapse** on multimodal tasks — "frequently generating incoherent outputs."
- **Evaluated on LLaVA-1.5 and Qwen-2.5-VL** (not Qwen2-VL) over 9 VQA benchmarks **including POPE** — the closest existing measurement of hallucination on quantized MLLMs.
- ⚠️ No public code found; treat as paper-only baseline. Project page: https://shubhangb97.github.io/LUQ/

## Attention ↔ Hallucination Evidence (mechanism anchors)

| Work                      | Finding                                                                                                                                                                                | Source                                        |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| **OPERA** (CVPR 2024)     | MLLMs over-trust a few "summary tokens" and **neglect image tokens**; attention-penalty + retrospection mitigates hallucination                                                        | arXiv:2311.17911 · github.com/shikiw/OPERA    |
| **VCD** (CVPR 2024)       | Hallucination from statistical bias / **unimodal language priors**; contrastive decoding with distorted images (+7.4 POPE-F1 on LLaVA-1.5) — supports our "linguistic prior" mechanism | arXiv:2311.16922 · github.com/DAMO-NLP-SG/VCD |
| **RBD** (2024)            | In LLaVA-1.5-7B, **visual tokens receive only ~25% of total attention**, imbalance intensifies in deeper layers                                                                        | arXiv:2409.06485                              |
| **ASCD** (AAAI 2026)      | Contrastive decoding methods *lower* attention on visual tokens; explicit attention steering changes hallucination; "text inertia"                                                     | arXiv:2506.14766                              |
| **MIHBench** (ACM MM '25) | Multi-image hallucination scales with cross-image attention imbalance; balancing image attention reduces hallucination                                                                 | arXiv:2508.00726                              |
| **HDPO** (2024)           | Lowest-attention visual tokens construct hallucinated negatives — attention weights track hallucination                                                                                | arXiv:2411.10436                              |

> [!note] Terminology gap
> There is **no canonical "attention-to-image grounding" metric** in the literature. Our per-token visual attention mass $\bar{a}_v(t)$ ([[03-Metrics-Definitions#2.2]]) is itself a small contribution.

## Gap Analysis (novelty positioning)

Searches for hallucination in *quantized* MLLMs return very little:

1. **LUQ** — POPE as one column of 9 benchmarks (sub-4-bit LLaVA-1.5/Qwen-2.5-VL); a quantization-paper benchmark, not a hallucination study; no FP16-vs-bitwidth ablation of hallucination metrics; no attention analysis.
2. **"Evaluating the Impact of PTQ on Reliable VQA with MLLMs"** (arXiv:2602.13289) — accuracy + confidence *reliability* on quantized Qwen2-VL-7B / Idefics3-8B (HQQ vs MBQ); not hallucination metrics.
3. **"Quantized but Deceptive?"** (EMNLP 2025, aclanthology.org/2025.emnlp-main.1548) — truthfulness of quantized *text-only* LLMs.
4. **"Tethered Reasoning"** (arXiv:2602.17691) — entropy–hallucination in quantized text LLMs.

> [!success] Novelty claim
> **No published work** performs a same-dataset ablation of hallucination (POPE/CHAIR) across a precision grid (FP16/W8A8/W4A8/W4A4) on LLaVA-1.5-7B/13B + Qwen2-VL-7B **and** connects quantization error to cross-modal attention metrics. Our differentiators vs LUQ: hallucination as object of study, full grid incl. W4A8, attention mechanism, two model families.

## Correction Log (assumptions overturned by verification)

1. Qwen2-VL has **no cross-attention** (ViT + MLP merger + self-attention).
2. POPE repo = **RUCAIBox/POPE** (shikohome/POPE dead).
3. QuaRot repo = **spcl/QuaRot**, LLM-only.
4. **AutoGPTQ archived** → GPTQModel.
5. **AWQ is W4A16 weight-only** — W4A8/W4A4 needs MQuant-style pipelines.
6. TextVQA repo (faverogian) dead → textvqa.org + lmms-eval.
7. lmms-eval covers POPE/ScienceQA/TextVQA but **not CHAIR**.
8. LUQ evaluated **Qwen-2.5-VL**, not Qwen2-VL.

## Related Notes

- [[01-Problem-and-Hypotheses]] — hypotheses anchored on this evidence
- [[02-Evidence-Experiment-Design]] — decisions that follow (harness, quantizers)
- [[05-Execution-Plan]] — risks R1–R6 trace to this note
- [[README]] — vault index