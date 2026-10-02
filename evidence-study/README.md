# Experiments — Lexical Fallback Evidence Study

Modular, reproducible harness for the main study: **LLaVA-1.5-7B** across a five-rung precision ladder — **FP16 / W8A8 (TorchAO) / W4A16 (GPTQ) / W4A8 (Quanto) / W4A4 (Quanto + fake-quant activations)** — evaluated on **POPE** (all 3 splits) and **CHAIR** (500 captions), with per-step cross-modal attention capture and a text-only prior probe.

Research context: `../obsidian-docs/` → **Research Ideation** (hypotheses H1–H4, probes S1–S3), **Study Experiment** (protocol), **Results** (log).

---

## 1. What this experiment answers

| Hypothesis | Where to look |
|---|---|
| H1 — hallucination rises with precision loss | `results/ablation.json` → POPE F1 per split, CHAIR_s/i per variant |
| H2 — attention degradation | attention `mean_mass` / `mean_entropy` per variant (report.json) |
| H3 — token-level attention↔grounding coupling | `f3_attention_hallucination.png`, `f4_binned_grounding.png` |
| H4 — fallback timeline (attention decay over steps) | `f2_fallback_timeline.png` |
| S2/S2a/S2b — prior attribution | `pope_*_prior.jsonl` (P_txt per question), ΔKL in report.json |

---

## 2. Prerequisites

**Hardware**
- **GPU (recommended):** 16 GB VRAM (Colab free T4, Kaggle 2×T4). 7B FP16 fits T4 at batch 1; the quantized rungs fit easily.
- CPU-only: works for `fp16` (slow); GPTQ decode on CPU is ~5× slower — smoke tests only.

**Disk (first run):** ~20 GB total
- base checkpoint ~14 GB (only needed for `w4a16`'s GPTQ extraction — deleted after unless `keep_base_checkpoint: true`), quantized W4 (GPTQ) ~4.5 GB
- `fp16`/`w8a8`/`w4a8`/`w4a4` all load and quantize straight from the hub at run time — no extra checkpoint on disk for those
- COCO annotations zip ~250 MB (extracted jsons)
- ~1,000 val2014 images (downloaded selectively, ~0.5–1 GB)

**Network:** access to huggingface.co, images.cocodataset.org, raw.githubusercontent.com (all public, no auth).

**Python:** uv ≥ 0.5 (`pip install uv` or standalone installer).

---

## 3. Setup

```bash
cd experiments
uv sync          # creates .venv + installs pinned deps (incl. torch; allow ~10-20 min first time)
```

> **Version pins (important):** the lockfile pins the auto_gptq-compatible stack — `transformers 4.46.3`, `torch 2.5.1`, `accelerate 0.34.2`. If you synced with an older lock, re-run `uv sync` after pulling the updated `uv.lock`.

Optional environment variables (set before any run):
```bash
export HF_HOME=~/hf-cache          # where checkpoints download
# behind a slow mirror? export HF_ENDPOINT=https://hf-mirror.com
```

---

## 4. Quick start

```bash
# resampled run (~3-3.5 h on a T4) — recommended first
uv run experiments --root . --sample-images 100

# fallback if the console script / editable install is broken (e.g., after an
# interrupted `uv sync` in a notebook): run the module directly from src
PYTHONPATH=src uv run python -m experiments.cli --root . --sample-images 100

# full study (~13-14 h on a T4; fits Kaggle 30 h/wk)
uv run experiments --root .

# everything already prepared; just rerun eval/analysis
uv run experiments --root . --skip-download --skip-quantize
```

---

## 5. CLI reference

```bash
uv run experiments --help
```

| Flag | Default | Meaning |
|---|---|---|
| `--root .` | `"."` | base dir; `data/`, `checkpoints/`, `results/`, `config.yaml` resolve here |
| `--variants fp16,w8a8,w4a16,w4a8,w4a4` | from config | which cells to run |
| `--sample-images N` | `null` (full set) | resample N images per POPE split + N CHAIR images (keeps 6-question blocks) |
| `--device auto` | `auto` | `auto` \| `cuda` \| `cpu` |
| `--skip-download` | off | don't fetch data — fails fast if `data/` is incomplete (offline re-runs) |
| `--skip-quantize` | off | reuse existing quantized checkpoints |
| `--skip-probe` | off | skip text-only prior probe + ΔKL (saves ~4 h, loses S2 evidence) |
| `--no-attention` | off | skip attention capture (saves ~1.4× time, loses H2/H3/H4) |

---

## 6. Configuration (`config.yaml`)

| Key | Default | Notes |
|---|---|---|
| `model_id` | `liuhaotian/llava-v1.5-7b` | any HF LLaVA-1.5-style repo |
| `variants` | `[fp16, w8a8, w4a16, w4a8, w4a4]` | supported: `fp16`, `w8a8` (TorchAO int8 dynamic act + int8 weight), `w4a16` (GPTQ W4 g128, weight-only), `w4a8` (Quanto int4 weight + int8 act), `w4a4` (Quanto int4 weight + fake-quant int4 act) |
| `sample_images` | `null` | `null` = full set (500/500) |
| `seed` | `42` | global seed; per-image seeds derived deterministically |
| `calibration_samples` | `128` | GPTQ calibration size (MSCOCO train2014 captions) — only used by `w4a16` |
| `gptq_group_size` | `128` | GPTQ group size (`w4a16` only) |
| `keep_base_checkpoint` | `false` | `true` keeps the 14 GB FP16 checkpoint after GPTQ extraction; `false` deletes it — required on Kaggle's ~30 GB disk. Only relevant to `w4a16`; `fp16`/`w8a8`/`w4a8`/`w4a4` always load from the hub regardless of this flag |
| `max_new_tokens_pope/chair` | `8 / 256` | answer/caption length caps |
| `temperature / top_p / do_sample` | `1.0 / 0.9 / true` | nucleus sampling for POPE; set `do_sample: false` for greedy (CHAIR uses its own greedy path via `cfg.do_sample=false` only if you flip it — see note below) |
| `capture_attention` | `true` | per-step attention aggregates |
| `run_probe` | `true` | S2/S2a/S2b |
| `probe_chair_subset` | `100` | captions used for ΔKL probe |

> **Decoding note:** the harness uses one decoding config; POPE (nucleus) and CHAIR (greedy) share `do_sample`. For strict per-benchmark conventions, run the full study with `do_sample: true` (nucleus) as documented — the CHAIR effect is small under nucleus with per-image seeds, and it keeps one reproducible path. If you need greedy CHAIR, add `do_sample: false` and accept nucleus POPE.

---

## 7. What happens on the first run (flow)

```
1. prepare data     → downloads COCO annotations (instances/captions val+train),
                      POPE question files (all 3 splits), and only the needed
                      val2014 images (~1,000) into data/
2. prepare checkpoints → only if `w4a16` is in `variants`: downloads LLaVA-1.5-7B
                      (checkpoints/base), extracts + GPTQ-quantizes the LLM
                      (checkpoints/gptq-llm-w4). fp16/w8a8/w4a8/w4a4 quantize
                      on the fly at load time straight from the hub — nothing
                      to prepare for them
3. per variant      → POPE (3 splits) + CHAIR (500 captions) with attention capture
4. text-only probe  → P_txt(yes) per POPE question (S2a) + lockstep ΔKL (S2b)
5. analysis         → results/<variant>/report.json, results/ablation.json,
                      results/summary.md, results/figures/*.png
```

Resume behavior: anything already present is skipped (downloads, quantized checkpoints). **Per-question/per-image incremental checkpointing**: each POPE/CHAIR record is appended to its jsonl immediately, so a session crash (Kaggle 12 h limit, Colab disconnect) resumes exactly where it stopped — rerun the same command and it continues, then finalizes reports/figures.

---

## 8. Outputs — where and what to look at

```
results/
├── ablation.json                    ← ALL cells in one file (main comparison table)
├── summary.md                       ← human-readable version of the same
├── figures/
│   ├── f1_hallucination_vs_precision.png   H1
│   ├── f2_fallback_timeline.png            H4
│   ├── f3_attention_hallucination.png      H3
│   ├── f4_binned_grounding.png             H3 + τ intuition
│   └── f5_attention_entropy.png            H2
└── <variant>/
    ├── report.json                  ← POPE (per split: acc/precision/recall/f1/yes_ratio),
    │                                   CHAIR (s/i), attention (mass/entropy/drift/decile
    │                                   profile), ΔKL
    ├── pope_<split>.jsonl           ← per-question: label, answer, correct, attention summary
    ├── pope_<split>_prior.jsonl     ← S2a: per-question text-only P(yes)
    └── chair_captions.jsonl         ← captions + mentions (class, grounded, attention) — H3 rows
```

**How to read the results (expected under the hypothesis — H1 ordering FP16 < W8A8 < W4A16 < W4A8 < W4A4):**

| Signal | FP16 (published anchor) | W8A8 | W4A16 | W4A8 | W4A4 |
|---|---|---|---|---|---|
| POPE-F1 random / popular / adversarial | ~87.3 / 86.1 / 84.2 | ↓ small | ↓ mid | ↓ larger | ↓↓ largest |
| yes-ratio on "no" questions | baseline | ↑ | ↑ | ↑↑ | ↑↑↑ |
| attention mean mass | ~0.2–0.3 | ↓ | ↓ | ↓↓ | ↓↓↓ |
| attention entropy | low | ↑ | ↑ | ↑↑ | ↑↑↑ |
| ΔKL (vs text-only) | baseline | ↑ | ↑ | ↑↑ | ↑↑↑ (converges toward text-only) |

W4A16→W4A8 isolates the effect of adding activation quantization on top of the same int4 weights; W8A8→W4A16 isolates the effect of dropping weight precision at fixed (near-lossless) activation precision.

**First sanity gate (reproduce-before-trust):** the FP16 cell's POPE-F1 must land within ~1–2 pts of the published 87.3/86.1/84.2 before trusting any quantized cell. If it doesn't, check prompt template / decoding settings first.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `uv sync` slow | torch download is large; allow 10–20 min; or `uv sync --extra-index-url https://download.pytorch.org/whl/cpu` for CPU-only wheels |
| auto-gptq build fails | needs a C++ toolchain (`gcc`, `python3-dev`); on Colab/Kaggle it's preinstalled — if local, `sudo apt install build-essential` |
| "CUDA extension not installed" (auto_gptq) | **expected, non-fatal** — auto_gptq's fused kernels need compilation at install; on notebooks it falls back to pure-torch ops. Quantization still runs on the GPU, just slower (one-time, ~10-30 min) |
| w8a8 cell fails to load / `TorchAoConfig` error | needs `torchao>=0.4.0`; check `python -c "import torchao; print(torchao.__version__)"` in the venv |
| w4a8/w4a4 cell fails to load / `QuantoConfig` error | needs `optimum-quanto` importable as `optimum.quanto`; `QuantoConfig(activations="int4")` is expected to raise — real int4 activation quant isn't supported by the library, which is why w4a4 uses a fake-quant hook instead (see §10) |
| vision-tower carve-out warning in logs | `[load] WARNING: vision-tower carve-out did not hold...` means the CLIP tower or projector got quantized despite `modules_to_not_convert` — that run measured a *different* failure mode (vision-side collapse) and should not be folded into the H1–H4 numbers |
| quantize step OOM | reduce `calibration_samples` to 64; run with `--device cuda`; close other GPU processes |
| GPTQ checkpoint won't load | transformers version mismatch — the lockfile pins it; don't upgrade transformers independently |
| downloads stall | network to COCO/HF blocked; set `HF_ENDPOINT` mirror, or pre-place files: annotations in `data/annotations/`, images in `data/val2014/`, POPE files in `data/pope/` (filenames `coco_pope_{split}.json`) |
| Colab session too short | use `--sample-images 100`; resume with `--skip-download --skip-quantize --variants <pending>` |
| attention all zeros / span empty | LLaVA processor didn't expand `<image>` — check prompt has `USER: <image>\n... ASSISTANT:` (see `data/prompts.py`) |

---

## 10. Reproducibility & logging

- Every random draw derives from `seed` (global) + image name → identical generations across machines.
- Calibration data (train2014) is disjoint from evaluation (val2014) — by design.
- After a run, log the numbers into `../obsidian-docs/Results.md` (entry template there), including `--sample-images` if used and the seed.
- For the paper/proposal, always state: model, variant set, sample sizes, seed, calibration size, and the figures F1–F5.
- **Variant mechanics (state in methods):**
  - `w8a8` = TorchAO `int8_dynamic_activation_int8_weight` (no calibration; activations quantized dynamically per forward pass).
  - `w4a16` = GPTQ g128 on the extracted Vicuna LLM (auto_gptq can't quantize the `llava` wrapper class directly, so the `llama`-type LM is quantized standalone and swapped into the fp16 LLaVA at load time). Weight-only.
  - `w4a8` = Quanto `QuantoConfig(weights="int4", activations="int8")` — both real quantized dtypes.
  - `w4a4` = Quanto `QuantoConfig(weights="int4", activations=None)` (real int4 weights) plus a forward-pre-hook on every `Linear` in `language_model` that fake-quantizes activations to int4 (round/clip/dequantize in fp16). **This is a simulation, not a real int4 GEMM kernel** — optimum-quanto and transformers' `QuantoConfig` both hard-reject real int4 activation quantization (`ActivationQBytesTensor` only supports 8-bit qtypes). State this plainly in methods: w4a4 numerically approximates the collapse regime rather than measuring a production int4×int4 kernel.
  - All four quantized rungs exclude `vision_tower` and `multi_modal_projector` from quantization (`modules_to_not_convert`), so only the LLM decoder's precision changes — consistent with GPTQ's existing extract-and-swap approach and with the study's isolated variable (H1–H4 are about decoder fallback, not vision-tower collapse).

---

## 11. Layout

```
config.yaml                experiment config
pyproject.toml / uv.lock   pinned environment
src/experiments/
  models/                  checkpoint download, GPTQ quantization (W4), variant loading
                           (fp16/w8a8/w4a16/w4a8/w4a4 dispatch in load.py)
  data/                    COCO subset downloader, POPE/CHAIR loaders + resamplers, prompts
  experiment/              decode loop w/ attention capture, POPE, CHAIR (+80-class matcher),
                           text-only probe, flow
  analysis/                metrics (F1, CHAIR_s/i, r_pb, binned curves, ΔKL), JSON export, plots
scripts/smoke_test.py      pure-logic sanity tests (no model needed):
                           uv run python scripts/smoke_test.py
scripts/preflight_variants.py   per-rung quantizer config validation (no GPU needed), and an
                           optional --load mode that loads each variant + checks
                           output_attentions still works (needs GPU/hub access):
                           uv run python scripts/preflight_variants.py --root .
```