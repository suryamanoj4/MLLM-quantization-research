#!/usr/bin/env bash
# Qwen2.5-VL entry point for the Phoenix C2/C3 pipeline.
#
# This is the second entry point next to the existing LLaVA `run_all.sh`.  Both run
# the same scripts in scripts/; only the model id, dtype and a few batch/memory
# defaults differ.  Results go to runs_qwen/ so the LLaVA runs stay untouched.
#
#   bash run_all_qwen.sh
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct DTYPE=bfloat16 POPE_BS=4 CHAIR_BS=2 \
#       bash run_all_qwen.sh
#   COCO=/path/to/coco bash run_all_qwen.sh
#
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

COCO="${COCO:-data/coco-mini}"
PREC="${PREC:-w4a8}"
OUT="${OUT:-runs_qwen}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
POPE_BS="${POPE_BS:-8}"
CHAIR_BS="${CHAIR_BS:-4}"
PRESET="${PRESET:-dev}"

if [ -z "${PY:-}" ] && [ -x .venv-phoenix/bin/python ]; then
  PY=.venv-phoenix/bin/python
fi
PY="${PY:-python}"

if [ ! -d "$COCO/annotations" ]; then
  if [ "$COCO" = "data/coco-mini" ]; then
    echo "############ building the mini-COCO (~110 MB) ############"
    $PY scripts/fetch_data.py --root "$COCO" --preset "$PRESET"
  else
    echo "[run_all_qwen] COCO=$COCO has no annotations/ directory." >&2
    echo "               Point COCO at an existing COCO checkout, or unset it." >&2
    exit 1
  fi
fi

echo "############ preflight ############"
$PY scripts/check_env.py --coco-root "$COCO" --model-id "$MODEL"

echo "############ 0. where does the model actually break? ############"
$PY scripts/00_precision_ladder.py --coco-root "$COCO" --model-id "$MODEL" \
    --dtype "$DTYPE" --ladder fp16,w8a8,w4a16,w4a8,w4a4 \
    --pope-batch-size "$POPE_BS" --chair-batch-size "$CHAIR_BS" \
    --n-pope-images 60 --n-chair-images 40 --out "$OUT/ladder"

echo "############ 1. layer-depth drift profiling ############"
$PY scripts/01_profile_drift.py --coco-root "$COCO" --model-id "$MODEL" \
    --precision "$PREC" --dtype "$DTYPE" \
    --n-images 32 --batch-size 2 --topk 8 --out "$OUT"

echo "############ 2. LoRAS calibration (closed form) ############"
$PY scripts/02_calibrate_loras.py --coco-root "$COCO" --model-id "$MODEL" \
    --precision "$PREC" --dtype "$DTYPE" \
    --drift-json "$OUT/drift__$PREC/drift.json" \
    --n-calib 256 --rank 16 --layers-per-pass 4 --out "$OUT"

LORAS="$OUT/loras__$PREC/loras.pt"

echo "############ 3. the four ablation cells ############"
$PY scripts/03_eval.py --coco-root "$COCO" --model-id "$MODEL" --dtype "$DTYPE" \
    --precision "$PREC" --pope-batch-size "$POPE_BS" --chair-batch-size "$CHAIR_BS" \
    --tag "${PREC}_base" --out "$OUT" --ppl
$PY scripts/03_eval.py --coco-root "$COCO" --model-id "$MODEL" --dtype "$DTYPE" \
    --precision "$PREC" --pope-batch-size "$POPE_BS" --chair-batch-size "$CHAIR_BS" \
    --loras "$LORAS" --tag "${PREC}_loras" --out "$OUT" --ppl
$PY scripts/03_eval.py --coco-root "$COCO" --model-id "$MODEL" --dtype "$DTYPE" \
    --precision "$PREC" --pope-batch-size "$POPE_BS" --chair-batch-size "$CHAIR_BS" \
    --acab add --alpha 1.0 --tau-percentile 60 \
    --tag "${PREC}_acab" --out "$OUT" --ppl
$PY scripts/03_eval.py --coco-root "$COCO" --model-id "$MODEL" --dtype "$DTYPE" \
    --precision "$PREC" --pope-batch-size "$POPE_BS" --chair-batch-size "$CHAIR_BS" \
    --loras "$LORAS" --acab add --alpha 1.0 --tau-percentile 60 \
    --tag "${PREC}_both" --out "$OUT" --ppl

echo "############ 4. A-CAB sweep ############"
$PY scripts/04_acab_sweep.py --coco-root "$COCO" --model-id "$MODEL" --dtype "$DTYPE" \
    --precision "$PREC" --pope-batch-size "$POPE_BS" --chair-batch-size "$CHAIR_BS" \
    --loras "$LORAS" --acab add --alphas 0.0,0.5,1.0,2.0,4.0 \
    --tau-percentiles 40,60,80 --with-pope --out "$OUT" --tag "$PREC"

echo "############ 5. systems metrics ############"
$PY scripts/05_latency.py --coco-root "$COCO" --model-id "$MODEL" --dtype "$DTYPE" \
    --precision "$PREC" --loras "$LORAS" --out "$OUT" --tag "$PREC"

echo "############ 6. figures ############"
$PY scripts/06_figures.py --runs "$OUT" --out figures_qwen

echo "done. results in $OUT/, figures in figures_qwen/"
