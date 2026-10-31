# Cross-precision causal patching

**Question:** where in the decoder does quantization lose grounding — which layer,
which component (residual / attention / MLP), and which token positions?

This is a **self-contained side experiment**. Its `phoenix/` package is a copy of the
main package with the patching-specific `model.py`, `acab.py`, `quant.py` and the new
`patching.py` from `phoenix_patching_qwen.zip`. Keeping it separate means the main
`method-study/scripts/` pipeline is not modified by these experimental changes.

## Layout

```
patching_qwen/
  phoenix/
    patching.py      activation / weight patching, token tagging, summaries
    model.py         Qwen/LLaVA loader + prompt adapter used by this experiment
    acab.py          generation loop with multi-EOS handling
    quant.py         fake-quant substrate
    ...              unchanged shared support modules
  scripts/13_patching.py
  tests/test_patching.py
  run.sh
```

## What it does

For each evaluation image:

1. Generate FP16's greedy caption.
2. Teacher-force that caption through the quantized model.
3. Restore FP16 at one site at a time:
   - activation patching: layer × {residual, attention, MLP} × {text, image}
   - weight patching: restore / damage one layer's attention or MLP
4. Measure how much of the lost log-probability on **object tokens** comes back, versus
   other caption tokens, with caption-bootstrap CIs.

## Setup

It uses the shared environment and data from the main study:

```bash
cd method-study
bash scripts/setup_env.sh
python scripts/fetch_data.py --root data/coco-mini --preset dev
```

The experiment wrapper looks for `../../.venv-phoenix/bin/python`, then
`../../.venv/bin/python`, then `python3`.

## Run

LLaVA:

```bash
cd method-study/experiments/patching_qwen
bash run.sh
# defaults: LLaVA-1.5-7B, w3a16, 100 captions, float16
```

Qwen:

```bash
MODEL=Qwen/Qwen2.5-VL-7B-Instruct DTYPE=bfloat16 \
  bash run.sh
```

From a local snapshot:

```bash
MODEL=/path/to/Qwen2.5-VL-7B-Instruct DTYPE=bfloat16 \
  PHOENIX_QWEN_MAX_PIXELS=1003520 bash run.sh
```

Extra flags are passed through, e.g.:

```bash
N=20 PREC=w4a16 MODEL=Qwen/Qwen2.5-VL-7B-Instruct DTYPE=bfloat16 \
  bash run.sh --layers 0:28:2 --modes act,w
```

## Outputs

Default: `runs/patching/<family>_<precision>/`

- `captions.jsonl` — per-image statistics (resumable)
- `patching_report.json` — gaps, per-site restoration/damage, CIs
- `patching_tables.txt` — human-readable tables
- `patching_heatmap.pdf`

`--summarize-only` recomputes the report from `captions.jsonl` without a model.

## Tests

```bash
python tests/test_patching.py
```

Covered: decoder-only replay equals the full HF forward, activation hooks restore
FP16 exactly at the last residual stream, weight restore/damage works, Qwen M-RoPE
position ids are preserved, token tagging, bootstrap summaries.

## Notes

- Qwen is forced to bfloat16 by the loader; `--dtype float16` is corrected with a log.
- `PHOENIX_QWEN_MAX_PIXELS` caps the dynamic image-token budget.
- The experiment is separate because its `model.py` / `acab.py` / `quant.py` differ
  from the main pipeline; do not import this directory from the main scripts.
