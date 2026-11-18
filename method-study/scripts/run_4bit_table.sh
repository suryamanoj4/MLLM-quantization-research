#!/usr/bin/env bash
# The W4 rows missing from the comparison table (Results Comparison §0.1), mirroring the
# W3 quantizer rows on the Study-2 settings, for both models:
#
#   W4 RTN            w4a16
#   W4 AWQ (wiki)     w4a16:method=awq:calib=wiki
#   W4 GPTQ (img+cap) w4a16:method=gptq            (calib=mm is the default)
#
# (W4 GPTQ-wiki, NF4, W4A4 rot.+GPTQ and W4 noise are already in the table.)
#
#   mkdir -p runs_4bit
#   HF_HOME=/persistent/hf nohup bash scripts/run_4bit_table.sh > runs_4bit/table.out 2>&1 &
#
# If a Study-2 ladder.json is present (LLAVA_GATE2 / QWEN_GATE2), it is COPIED and its
# exact settings and FP16 row are reused, so only the 3 new rungs run. Otherwise FP16 is
# re-run with the Study-2 settings (100 POPE images x 3 splits, 300 captions, 128 tokens).
# Every step resumes: rerun the same command after a restart.
#
# Overrides: PY, COCO, ONLY=llava|qwen, LLAVA_GATE2, QWEN_GATE2.
# Time: ~40 min per rung -> ~4 GPU-h with the Gate-2 files, ~5.5 GPU-h without.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${PY:-}" ]; then
  for c in .venv-phoenix/bin/python .venv/bin/python; do [ -x "$c" ] && { PY="$c"; break; }; done
  PY="${PY:-python3}"
fi
COCO="${COCO:-data/coco-mini}"
LLAVA_GATE2="${LLAVA_GATE2:-runs/gate2/ladder.json}"
QWEN_GATE2="${QWEN_GATE2:-runs_qwen/gate2/ladder.json}"
RUNGS="w4a16,w4a16:method=awq:calib=wiki,w4a16:method=gptq"
STUDY2="--n-pope-images 100 --n-chair-images 300 --max-new-tokens 128"

say() { printf '\n[w4-table] %s\n' "$*"; }
want() { [ -z "${ONLY:-}" ] || [ "$ONLY" = "$1" ]; }

[ -n "${HF_HOME:-}" ] || say "WARNING: HF_HOME not set; model weights go to ~/.cache/huggingface"
[ -f "$COCO/splits.json" ] || "$PY" scripts/fetch_data.py --root "$COCO" --preset full
"$PY" scripts/check_env.py --coco-root "$COCO"

run_model() {  # name model dtype gate2
  local name=$1 model=$2 dtype=$3 gate2=$4
  local out="runs_4bit/table_$name"
  mkdir -p "$out/logs"
  local args="$STUDY2"
  if [ ! -f "$out/ladder.json" ] && [ -f "$gate2" ]; then
    say "$name: copying $gate2 -> $out (original untouched; FP16 row reused)"
    cp "$gate2" "$out/ladder.json"
  fi
  if [ -f "$out/ladder.json" ] && "$PY" -c "import json,sys; sys.exit(0 if 'fp16' in json.load(open('$out/ladder.json'))['table'] else 1)"; then
    args=$("$PY" scripts/q1_resume_args.py "$out/ladder.json")
  else
    say "$name: no Study-2 ladder found; FP16 is re-run too"
  fi
  echo "$args" > "$out/logs/ladder_args.txt"

  say "$name: ladder (fp16 + $RUNGS)"
  # shellcheck disable=SC2086
  "$PY" scripts/00_precision_ladder.py --coco-root "$COCO" --model-id "$model" --dtype "$dtype" \
    --out "$out" --ladder fp16 --ladder-extra "$RUNGS" --resume $args 2>&1 | tee -a "$out/logs/ladder.log"

  say "$name: small-object drops"
  "$PY" scripts/10_object_analysis.py --coco-root "$COCO" --ladder "$out/ladder.json" \
    --out "$out/object_analysis.json" 2>&1 | tee -a "$out/logs/objects.log"
}

if want llava; then
  run_model llava llava-hf/llava-1.5-7b-hf float16 "$LLAVA_GATE2"
fi
if want qwen; then
  run_model qwen Qwen/Qwen2.5-VL-7B-Instruct bfloat16 "$QWEN_GATE2"
  # Qwen's table column is WikiText-2 PPL; W4 RTN and W4 AWQ-wiki are already measured
  # (validate_ptq.json), so only GPTQ (img+cap) is new. FP16 is re-measured for the ratio.
  say "qwen: WikiText-2 PPL"
  "$PY" scripts/09_validate_ptq.py --model-id Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 \
    --coco-root "$COCO" --specs "fp16,w4a16:method=gptq:calib=mm" \
    --out runs_4bit/table_qwen/validate 2>&1 | tee -a runs_4bit/table_qwen/logs/validate.log
fi

say "done. Send back: runs_4bit/table_*/ladder_report.json, object_analysis.json, validate/validate.json"
