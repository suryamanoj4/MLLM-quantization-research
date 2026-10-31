# Q1 — does LoRAS-T work because of *where* it corrects?

LoRAS-T corrects the decoder's hidden states at **text positions only**. Q1 fits the
**same** corrector — same rank 64, same layers, same calibration images, captions and
yes/no questions — at **all positions**, image tokens included, and evaluates both on
the Study-2 set.

The two variants have the same parameter count, so:

- **text ≥ all** → the text-side placement repairs grounding, not the added capacity.
- **all > text** → placement is not the reason.

This directory is a **self-contained side experiment**. The patch from
`phoenix_q1_lorast_positions.zip` is already applied; do not re-apply it. The main
`method-study/scripts/` pipeline is untouched.

## Layout

```
q1_lorast_positions/
  phoenix/
    loras_text.py          positions="text"|"all"; default "text"
    ...                    unchanged shared support modules
  scripts/
    12_calibrate_lorast.py --positions {text,all}
    q1_resume_args.py      reproduce an existing ladder's settings
    q1_summary.py          result table + export payload
    run_q1.sh              resumable end-to-end driver
    ...                    unchanged shared scripts used by the driver
  tests/
    test_lorast_positions.py
    ...                    shared regression tests
  run.sh                   entry point
```

## Setup

Use the shared environment/data:

```bash
cd method-study
bash scripts/setup_env.sh
python scripts/fetch_data.py --root data/coco-mini --preset dev
```

The wrapper finds `../../.venv-phoenix/bin/python`, then `../../.venv/bin/python`,
then `python3`. Defaults point at `../../data/coco-mini` and `../../runs/gate2`; new
Q1 artifacts stay in this experiment directory.

## Run

### LLaVA (required)

```bash
cd method-study/experiments/q1_lorast_positions
TEXT_PT=../../runs/lorast/w3_rtn_qa.pt \
  bash run.sh
```

Defaults: LLaVA-1.5-7B, float16, w3a16, `QA=2`, `GATE2=../../runs/gate2`.

### Qwen (optional)

```bash
TAG=qwen MODEL=Qwen/Qwen2.5-VL-7B-Instruct DTYPE=bfloat16 \
  GATE2=../../runs_qwen/gate2 \
  TEXT_PT=../../runs_qwen/lorast_qa/w3_rtn_qa.pt \
  ALL_PT=runs/lorast/w3_rtn_qa_allpos.pt \
  bash run.sh
```

If you do not have the Study-2 ladder or the text correctors yet, leave them unset:
the driver runs `fp16` and `w3a16` and fits the text variant, adding roughly 1.5–2 h.

Every step is skipped if its output exists, so rerunning resumes.

## What it does

| Step | Action | Time |
|---|---|---|
| 0 | CPU tests | ~2 min |
| 1 | Copy the Study-2 ladder and reproduce its settings | seconds |
| 2 | Text-only LoRAS-T (+QA); reused if `TEXT_PT` exists | 0 or ~40 min |
| 3 | All-positions corrector (same rank/layers/data) | ~1–1.5 h |
| 4 | Ladder `--resume`: evaluate only the new rungs | ~35–45 min per rung |
| 5 | Pairwise comparisons + object drops | minutes |
| 6 | Summary and zip | seconds |

## Outputs

`results_q1_<tag>/` contains:

- `q1_summary.txt` / `q1_summary.json`
- `ladder_report_vs_fp16.json`
- `pairwise_all_vs_text.json`
- `pairwise_vs_w3a16.json`
- `object_analysis.json`
- `calib_text.json`, `calib_all.json`
- `ladder_q1_rungs.json`
- `logs/`
- `<results_q1_tag>.zip` next to the directory

## Sanity checks

- Step 0 prints `all LoRAS-T position tests passed`.
- `logs/03_calib_all.log` starts with `positions=all: FP16 targets recomputed per block`;
  per-layer token counts are ~5× the text run.
- Step 4 prints `[resume] reusing N rung(s)` and only evaluates the new rungs.
- `q1_summary.txt` has a `+ LoRAS-T text` row close to the paper's `dCov ≈ −0.012`.
  If not, `TEXT_PT` is not the paper's question-augmented corrector.
