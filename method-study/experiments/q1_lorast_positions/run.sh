#!/usr/bin/env bash
# Self-contained entry point for the Q1 LoRAS-T placement ablation.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PY:-}" ]; then
  for c in ../../.venv-phoenix/bin/python ../../.venv/bin/python; do
    [ -x "$c" ] && { PY="$c"; break; }
  done
  PY="${PY:-python3}"
fi

# Defaults point at the shared method-study data/run, while new artifacts stay here.
export COCO="${COCO:-../../data/coco-mini}"
export GATE2="${GATE2:-../../runs/gate2}"
export TEXT_PT="${TEXT_PT:-../../runs/lorast/w3_rtn_qa.pt}"
export ALL_PT="${ALL_PT:-runs/lorast/w3_rtn_qa_allpos.pt}"
export PY

exec bash scripts/run_q1.sh "$@"
