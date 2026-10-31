# Qwen2.5-VL entry point

This directory holds **only the Qwen-specific pieces**: this note and the adapter
test. The experiment code, PTQ methods, LoRAS/LoRAS-T, A-CAB, data, CHAIR/POPE
scoring and object analysis are the **shared** code in `../scripts/`.

There are two entry points in `method-study/`:

| Entry point | Model | Results |
|---|---|---|
| `../run_all.sh` | LLaVA-1.5-7B (unchanged) | `runs/`, `figures/` |
| `../run_all_qwen.sh` | Qwen2.5-VL-7B | `runs_qwen/`, `figures_qwen/` |

## Why reuse instead of copying the pipeline

A full copy of `scripts/` was deliberately **not** kept. The copied version was
removed after this refactor. Reasons:

1. **One source of truth.** The experiment logic (ladder, PTQ wrappers, LoRAS,
   LoRAS-T, A-CAB, CHAIR/POPE, bootstrap statistics, object analysis) is identical
   for both models. A second copy would drift the moment either side is fixed.
2. **The model differences are small and local.** Qwen needs a different model
   class, chat template, MRoPE decode positions, yes/no token ids, variable visual
   tokens and bf16. Those live in the shared model adapter (`phoenix/model.py`) plus
   three small compatibility fixes; no experiment script needs a Qwen fork.
3. **Two entry points, same stages.** `run_all.sh` and `run_all_qwen.sh` run the same
   scripts in the same order, with the same output layout, so results are directly
   comparable. Only `--model-id`, `--dtype`, `--out` and batch/memory defaults differ.
4. **No regression risk for the LLaVA results.** LLaVA remains the default family:
   `build_prompt` keeps the original template, `DEFAULT_MODEL` is still LLaVA, and
   the Qwen branches are selected by the checkpoint's `model_type`. The existing
   `run_all.sh` and its artifact paths are unchanged.

The cost is that Qwen support is a small, backward-compatible addition to shared
files rather than a physically isolated tree. That is intentional: it keeps the
codebase lean and makes the model comparison honest, because both models execute
the same experiment code.

## What is Qwen-specific in the shared code

| Area | Location |
|---|---|
| Model class selection (`Qwen2_5_VLForConditionalGeneration`) | `phoenix/model.py` |
| Chat-template prompt (`<|vision_start|><|image_pad|><|vision_end|>`) | `phoenix/model.py` |
| `max_pixels` cap (`QWEN_MAX_PIXELS`, default 1,003,520) | `phoenix/model.py` |
| `visual` / `visual.merger` submodule aliases | `phoenix/quant.py` |
| Explicit `cache_position` on decode steps (MRoPE) | `phoenix/acab.py` |
| Qwen `Yes`/`ĠYes` token ids for P(yes)/AUROC | `phoenix/metrics.py` |
| Text-only prompts for blind PPL / text calibration | `phoenix/probes.py`, `phoenix/ptq/calib.py` |
| VLM detection in PTQ validation | `scripts/09_validate_ptq.py` |

Everything else is shared unchanged. Qwen2.5-VL's decoder is a Qwen2 stack with the
same Llama-style module names (`self_attn.q/k/v/o_proj`, `mlp.gate/up/down_proj`,
`input_layernorm`, `post_attention_layernorm`), which is why the PTQ and steering
code carries over directly.

## Results (2026-10-25)

The run is archived in `obsidian-docs/results/qwen-2.5vl-2026-10-25/`; the synthesis
is in `obsidian-docs/Results Summary.md` §1 and the per-rung detail in
`obsidian-docs/Method Study.md` §9.

| FP16 baseline | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---:|---:|
| Coverage | 0.744 | 0.559 |
| CHAIR_i | 0.220 | 0.109 |
| POPE-F1 (3-split mean) | 0.854 | 0.833 |
| WikiText-2 PPL | 7.27 | 8.28 |

Gate 2 (300 captions, 100 POPE images) replicates the LLaVA shape:

- W3 RTN: coverage **−0.061 [−0.085, −0.037]**, CHAIR_i −0.004 n.s.; small-object
  drop 0.360 vs large 0.118.
- W3 GPTQ (wiki): coverage −0.046*, CHAIR_i +0.006 n.s.
- Text-position weights: coverage −0.059*; image-position weights +0.004 n.s.
- W3 matched noise: coverage −0.042* — error energy, not method identity.
- Realistic W4A4 (`rot+gptq`): coverage −0.019 n.s., but **POPE-F1 −0.063***.
- LoRAS-T: coverage −0.009 n.s. vs FP16 (recovers +5.2 vs W3), small-object drop
  0.360→0.209, all 0.205→0.110; 12.9M params.
- KV-LoRAS: ceilings 0.22–0.37 (much lower than LLaVA’s 0.58–0.67); 2×2 coverage
  +2.9 points, CHAIR_i −0.005 n.s. — provisional (no coverage CI).
- A-CAB: best CHAIR_i 0.078 vs 0.091, but coverage 0.558 vs 0.580 — no
  coverage-preserving win.
- Gate 1 (16/17, v2): weight-only omission replicates (W3 RTN coverage −0.090*);
  W4A16 RTN is ok; NF4 / W3 GPTQ-mm / W4A8+SQ move toward fallback; `w4a4:rot`
  coverage −0.070*, POPE +0.005; `rot+gptq` POPE-F1 −0.063*.
- Activation attribution (v2): text tokens at A4 → BROKEN (100%), image tokens at
  A4 → degraded, LM intact (F1 −0.300*) — the same trigger as LLaVA. Qwen is
  already broken at W4A5 and degraded at W4A6.

**Two caveats.** Qwen Gate-1 is 16/17: GPTQ with text calibration hits a
non-positive-definite Hessian on the 18,944-wide down_proj, even with `damp=0.1`.
On a 40 GB card GPTQ also needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Qwen Gate-2 blind/caption PPL is not usable (FP16
blind PPL ≈146 vs WikiText 8.28), consistent with token-boundary misalignment when
a caption is appended to the chat template; use `validate_ptq.json` for LM health.
No visual attention / `r_pb` measurement exists for Qwen yet.

**Run details often asked for.** LoRAS-T was calibrated **without** yes/no questions
(`qa_per_image=0`, 192 FP16 captions, rank 64, all 28 layers). Its scale-1.0 POPE is
random 0.8346 / popular 0.8298 / adversarial 0.8298 (mean 0.8314 vs FP16 0.8330;
Δ −0.002 n.s.), yes-ratio ~0.370 vs 0.368. WikiText PPL for **W3 GPTQ (img+cap)** and
for both **noise** rungs was **not measured**; the archive only has wiki-calibrated
W3 GPTQ = 9.324. The noise rungs used **seed 0** (one draw per bit width; add
`:seed=1` for another). See `obsidian-docs/results/qwen-2.5vl-2026-10-25/README.md`
for the table and the command to fill the missing PPL rows.
**v3 update (2026-10-31):** all three gaps are filled — QA-calibrated LoRAS-T (`--qa-per-image 2`,
mean F1 0.8302, |ΔP(yes)| 0.014), PPL_wiki W3 GPTQ-mm 1.36× / W4 noise 1.05× / W3 noise 1.29×,
noise seeds 1 and 2 (`:seed=N`). Visual attention: `scripts/13_attention_probe.py`. See
`obsidian-docs/Results Comparison.md` §0.3–0.5.

## Setup

The existing `.venv` is a copy from another machine (its `bin/python` is not
executable and its shebang points at `/home/sushant/...`). Rebuild it before running:

```bash
cd method-study
bash scripts/setup_env.sh
source .venv-phoenix/bin/activate
python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
# expect transformers 4.53.3, which supports Qwen2.5-VL
```

Then:

```bash
python qwen/test_qwen_adapter.py       # CPU, no weights
python scripts/check_env.py --coco-root data/coco-mini \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct
```

## Run

```bash
# LLaVA, unchanged
bash run_all.sh

# Qwen2.5-VL, same stages, separate outputs
bash run_all_qwen.sh

# if the default 8/4 batches OOM on large images
POPE_BS=4 CHAIR_BS=2 bash run_all_qwen.sh

# smaller image budget
QWEN_MAX_PIXELS=501760 POPE_BS=4 bash run_all_qwen.sh
```

Stage by stage, using the shared scripts:

```bash
COCO=data/coco-mini
MODEL=Qwen/Qwen2.5-VL-7B-Instruct

python scripts/00_precision_ladder.py --coco-root $COCO --model-id $MODEL \
    --dtype bfloat16 --ladder fp16,w8a8,w4a16,w4a8,w4a4 \
    --n-pope-images 60 --n-chair-images 40 --out runs_qwen/ladder

python scripts/01_profile_drift.py --coco-root $COCO --model-id $MODEL \
    --dtype bfloat16 --precision w4a8 --n-images 32 --out runs_qwen
python scripts/02_calibrate_loras.py --coco-root $COCO --model-id $MODEL \
    --dtype bfloat16 --precision w4a8 --drift-json runs_qwen/drift__w4a8/drift.json \
    --n-calib 256 --rank 16 --out runs_qwen

python scripts/03_eval.py --coco-root $COCO --model-id $MODEL --dtype bfloat16 \
    --precision w4a8 --tag qwen_base --ppl --out runs_qwen

python scripts/00_precision_ladder.py --coco-root $COCO --model-id $MODEL \
    --dtype bfloat16 --preset gate2 --n-pope-images 100 --n-chair-images 300 \
    --out runs_qwen/gate2 --resume

python scripts/08_token_probe.py --coco-root $COCO --model-id $MODEL \
    --dtype bfloat16 --precisions fp16,w4a8,w4a4 --out runs_qwen/token_probe
```

The Gate-1/Gate-2 presets are unchanged. `09_validate_ptq.py` also accepts Qwen
(it recognises `qwen2_5_vl` as a VLM and defaults to `calib=mm`).

## Sanity checks before trusting numbers

1. **Visual mask.** `(input_ids == image_token_id).sum()` must equal
   `image_grid_thw.prod(-1) // 4`. Zero means the LLaVA prompt was used.
2. **Decode positions.** Greedy captions must be coherent. Without the
   `cache_position` fix Qwen still emits fluent-looking text but every generated
   token has MRoPE position 0.
3. **POPE baseline.** FP16 POPE F1 should be near published Qwen2.5-VL numbers and
   `unparsed` near zero.
4. **P(yes).** Not constant, AUROC high on FP16. The adapter resolves Qwen's
   `Yes`/`ĠYes` ids explicitly.
5. **Quantizers.** `09_validate_ptq.py` should show rotation rescuing W4A4,
   SmoothQuant W8A8 near FP16, GPTQ/AWQ beating RTN.

## Caveats

- Qwen2.5-VL-7B uses **GQA 28/4**, so AWQ/SmoothQuant skip the `v_proj → o_proj`
  fusion used on LLaVA's MHA. Results are valid but not directly comparable across
  models.
- Visual-token counts are **variable**. Attention shares are comparable; absolute
  token counts and per-token costs are not.
- Use `--dtype bfloat16`; Qwen is native bf16.
- Cap `QWEN_MAX_PIXELS` for a consistent image budget.
- Qwen has 28 decoder layers vs LLaVA's 32, so drift-selected LoRAS layers and
  LoRAS-T parameter counts differ.
- Qwen may be more verbose; check POPE `unparsed` and CHAIR caption length before
  comparing absolute scores.
