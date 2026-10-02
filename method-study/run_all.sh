#!/usr/bin/env bash
# Phoenix C2/C3 fast dev loop. Roughly 3-5 h end to end on one A6000 at the
# default sizes; every stage writes JSON into runs/ and is independently re-runnable.
#
#   bash run_all.sh
#       builds data/coco-mini if it is missing, checks the GPU, then runs everything.
#   COCO=/path/to/coco bash run_all.sh
#       alternative: use an existing full COCO checkout instead.
#
set -euo pipefail
# Python block-buffers stdout when it is a pipe (e.g. `| tee run.log`), which hides
# every print() until 8 KB accumulate. Unbuffered keeps the log live.
export PYTHONUNBUFFERED=1

COCO="${COCO:-data/coco-mini}"
PREC="${PREC:-w4a8}"
OUT="${OUT:-runs}"
if [ -z "${PY:-}" ] && [ -x .venv-phoenix/bin/python ]; then
  PY=.venv-phoenix/bin/python          # built by scripts/setup_env.sh
fi
PY="${PY:-python}"
PRESET="${PRESET:-dev}"

if [ ! -d "$COCO/annotations" ]; then
  if [ "$COCO" = "data/coco-mini" ]; then
    echo "############ building the mini-COCO (~110 MB) ############"
    $PY scripts/fetch_data.py --root "$COCO" --preset "$PRESET"
  else
    echo "[run_all] COCO=$COCO has no annotations/ directory." >&2
    echo "          Point COCO at an existing COCO checkout, or unset it to build data/coco-mini." >&2
    exit 1
  fi
fi

echo "############ preflight ############"
$PY scripts/check_env.py --coco-root "$COCO"

echo "############ 0. where does the model actually break? ############"
$PY scripts/00_precision_ladder.py --coco-root "$COCO" \
    --ladder fp16,w8a8,w4a16,w4a8,w4a4 \
    --n-pope-images 60 --n-chair-images 40 --out "$OUT/ladder"

echo "############ 1. layer-depth drift profiling ############"
$PY scripts/01_profile_drift.py --coco-root "$COCO" --precision "$PREC" \
    --n-images 32 --batch-size 2 --topk 8 --out "$OUT"

echo "############ 2. LoRAS calibration (closed form) ############"
$PY scripts/02_calibrate_loras.py --coco-root "$COCO" --precision "$PREC" \
    --drift-json "$OUT/drift__$PREC/drift.json" \
    --n-calib 256 --rank 16 --layers-per-pass 4 --out "$OUT"

LORAS="$OUT/loras__$PREC/loras.pt"

echo "############ 3. the four ablation cells ############"
$PY scripts/03_eval.py --coco-root "$COCO" --precision "$PREC" \
    --tag "${PREC}_base"  --out "$OUT" --ppl
$PY scripts/03_eval.py --coco-root "$COCO" --precision "$PREC" --loras "$LORAS" \
    --tag "${PREC}_loras" --out "$OUT" --ppl
$PY scripts/03_eval.py --coco-root "$COCO" --precision "$PREC" \
    --acab add --alpha 1.0 --tau-percentile 60 \
    --tag "${PREC}_acab"  --out "$OUT" --ppl
$PY scripts/03_eval.py --coco-root "$COCO" --precision "$PREC" --loras "$LORAS" \
    --acab add --alpha 1.0 --tau-percentile 60 \
    --tag "${PREC}_both"  --out "$OUT" --ppl

echo "############ 4. A-CAB sweep ############"
$PY scripts/04_acab_sweep.py --coco-root "$COCO" --precision "$PREC" --loras "$LORAS" \
    --acab add --alphas 0.0,0.5,1.0,2.0,4.0 --tau-percentiles 40,60,80 \
    --with-pope --out "$OUT" --tag "$PREC"

echo "############ 5. systems metrics ############"
$PY scripts/05_latency.py --coco-root "$COCO" --precision "$PREC" --loras "$LORAS" \
    --out "$OUT" --tag "$PREC"

echo "############ 6. figures ############"
$PY scripts/06_figures.py --runs "$OUT" --out figures

echo "done. results in $OUT/, figures in figures/"
