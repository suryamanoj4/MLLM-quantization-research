#!/usr/bin/env bash
# Q1: is LoRAS-T's TEXT-position placement what repairs grounding, or just its capacity?
# Fits the same rank-64 corrector at ALL positions (image tokens included) and evaluates
# it next to text-only LoRAS-T on the Study-2 set. See README_Q1.md.
#
#   bash scripts/run_q1.sh                     # LLaVA, defaults below
#   TAG=qwen MODEL=/path/to/qwen DTYPE=bfloat16 GATE2=runs_qwen/gate2 \
#     TEXT_PT=runs_qwen/lorast_qa/w3_rtn_qa.pt ALL_PT=runs_qwen/lorast_qa/w3_rtn_qa_allpos.pt \
#     bash scripts/run_q1.sh
#
# Every step is skipped if its output already exists, so a re-run resumes.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${PY:-}" ]; then
  for c in .venv-phoenix/bin/python .venv/bin/python; do [ -x "$c" ] && { PY="$c"; break; }; done
  PY="${PY:-python3}"
fi
COCO="${COCO:-data/coco-mini}"
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
DTYPE="${DTYPE:-float16}"
TAG="${TAG:-llava}"
PREC="${PREC:-w3a16}"
GATE2="${GATE2:-runs/gate2}"                       # existing Study-2 ladder dir (has fp16 + w3a16)
TEXT_PT="${TEXT_PT:-runs/lorast/w3_rtn_qa.pt}"     # existing text-only LoRAS-T (+QA); refit if missing
ALL_PT="${ALL_PT:-runs/lorast/w3_rtn_qa_allpos.pt}"
QA="${QA:-2}"                                      # yes/no questions per calibration image (paper: 2)
WORK="${WORK:-runs/q1_${TAG}}"                     # COPY of GATE2; the original is never modified
OUT="${OUT:-results_q1_${TAG}}"

say() { printf '\n[q1] %s\n' "$*"; }
mkdir -p "$OUT/logs" "$(dirname "$ALL_PT")"
LOG="$OUT/logs"

say "python: $PY  model: $MODEL ($DTYPE)  coco: $COCO  tag: $TAG"
"$PY" -c "import torch, transformers; print('[q1] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

# 0. the patch is in and works (CPU, ~2 min)
if [ ! -f "$LOG/00_tests.ok" ]; then
  say "0/6 CPU tests"
  "$PY" tests/test_lorast_positions.py 2>&1 | tee "$LOG/00_tests.log" | tail -3
  grep -q "all LoRAS-T position tests passed" "$LOG/00_tests.log" || { echo "[q1] tests failed, see $LOG/00_tests.log"; exit 1; }
  touch "$LOG/00_tests.ok"
fi

# 1. work on a copy of the Study-2 ladder (fresh run if there is none)
if [ ! -f "$WORK/ladder.json" ]; then
  if [ -f "$GATE2/ladder.json" ]; then
    say "1/6 copying $GATE2 -> $WORK (original untouched)"
    mkdir -p "$WORK" && cp "$GATE2/ladder.json" "$WORK/ladder.json"
  else
    say "1/6 no $GATE2/ladder.json: Q1 will also run fp16 and $PREC (about +1.5 h)"
    mkdir -p "$WORK"
  fi
fi
if [ -f "$WORK/ladder.json" ]; then
  ARGS=$("$PY" scripts/q1_resume_args.py "$WORK/ladder.json" 2> >(tee "$LOG/01_existing_rungs.log" >&2))
else
  ARGS="--n-pope-images 100 --n-chair-images 300 --max-new-tokens 128"
fi
echo "$ARGS" > "$LOG/01_ladder_args.txt"

# 2. text-only LoRAS-T (reuse the Study-2 correctors if present)
if [ ! -f "$TEXT_PT" ]; then
  say "2/6 calibrating text-only LoRAS-T -> $TEXT_PT (~30-45 min)"
  "$PY" scripts/12_calibrate_lorast.py --coco-root "$COCO" --model-id "$MODEL" --dtype "$DTYPE" \
    --precision "$PREC" --qa-per-image "$QA" --positions text --out "$TEXT_PT" 2>&1 | tee "$LOG/02_calib_text.log"
else
  say "2/6 reusing $TEXT_PT"
fi

# 3. the ablation: same corrector at ALL positions
if [ ! -f "$ALL_PT" ]; then
  say "3/6 calibrating ALL-positions corrector -> $ALL_PT (~1-1.5 h)"
  "$PY" scripts/12_calibrate_lorast.py --coco-root "$COCO" --model-id "$MODEL" --dtype "$DTYPE" \
    --precision "$PREC" --qa-per-image "$QA" --positions all --out "$ALL_PT" 2>&1 | tee "$LOG/03_calib_all.log"
fi

# 4. evaluate both as ladder rungs next to fp16 / w3a16 (only missing rungs run)
TEXT_RUNG="$PREC+lorast=$TEXT_PT"
ALL_RUNG="$PREC+lorast=$ALL_PT"
say "4/6 ladder: $TEXT_RUNG , $ALL_RUNG (~35-45 min per new rung)"
# shellcheck disable=SC2086
"$PY" scripts/00_precision_ladder.py --coco-root "$COCO" --model-id "$MODEL" --dtype "$DTYPE" \
  --out "$WORK" --ladder "fp16,$PREC" --ladder-extra "$TEXT_RUNG,$ALL_RUNG" --resume $ARGS \
  2>&1 | tee "$LOG/04_ladder.log"

# 5. paired comparisons + object drops (CPU, minutes)
say "5/6 pairwise comparisons and object analysis"
"$PY" scripts/11_pairwise.py --ladder "$WORK/ladder.json" --ref "$TEXT_RUNG" --rungs "$ALL_RUNG" \
  --out "$WORK/pairwise_all_vs_text" 2>&1 | tee "$LOG/05_pairwise_all_vs_text.log"
"$PY" scripts/11_pairwise.py --ladder "$WORK/ladder.json" --ref "$PREC" --rungs "$TEXT_RUNG,$ALL_RUNG" \
  --out "$WORK/pairwise_vs_base" 2>&1 | tee "$LOG/05_pairwise_vs_base.log"
"$PY" scripts/10_object_analysis.py --coco-root "$COCO" --ladder "$WORK/ladder.json" \
  --out "$WORK/object_analysis.json" 2>&1 | tee "$LOG/05_objects.log"

# 6. summary + the zip to send back
say "6/6 summary"
"$PY" scripts/q1_summary.py --work "$WORK" --base "$PREC" --text-rung "$TEXT_RUNG" \
  --all-rung "$ALL_RUNG" --text-pt "$TEXT_PT" --all-pt "$ALL_PT" --out "$OUT"
cp "$WORK/ladder_report.json" "$OUT/ladder_report_vs_fp16.json"
cp "$WORK/pairwise_all_vs_text/ladder_report.json" "$OUT/pairwise_all_vs_text.json"
cp "$WORK/pairwise_vs_base/ladder_report.json" "$OUT/pairwise_vs_${PREC}.json"
cp "$WORK/object_analysis.json" "$OUT/object_analysis.json"
cp "${TEXT_PT%.pt}.json" "$OUT/calib_text.json" 2>/dev/null || true
cp "${ALL_PT%.pt}.json" "$OUT/calib_all.json"
"$PY" -c "import shutil; shutil.make_archive('$OUT', 'zip', '.', '$OUT')"
say "done -> send ${OUT}.zip"
