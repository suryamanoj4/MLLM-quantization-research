#!/usr/bin/env bash
# Self-contained entry point for the causal-patching experiment.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PY:-}" ]; then
  for c in ../../.venv-phoenix/bin/python ../../.venv/bin/python; do
    [ -x "$c" ] && { PY="$c"; break; }
  done
  PY="${PY:-python3}"
fi

COCO="${COCO:-../../data/coco-mini}"
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
PREC="${PREC:-w3a16}"
N="${N:-100}"
DTYPE="${DTYPE:-float16}"

ARGS=(--coco-root "$COCO" --model-id "$MODEL" --precision "$PREC" --n "$N" --dtype "$DTYPE")
[ -n "${OUT:-}" ] && ARGS+=(--out "$OUT")

exec "$PY" scripts/13_patching.py "${ARGS[@]}" "$@"
