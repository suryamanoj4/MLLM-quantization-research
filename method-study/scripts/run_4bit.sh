#!/usr/bin/env bash
# 4-bit settings + real-kernel latency, both models, end to end. Resumable: every step
# skips finished work (ladder/latency --resume, calibration if the .pt exists), so after
# an instance restart just run the same command again.
#
#   HF_HOME=/persistent/hf nohup bash scripts/run_4bit.sh > runs_4bit/run.out 2>&1 &
#
# Steps
#   0  environment: venv, data, pynvml, autoawq (for the official Qwen AWQ checkpoint)
#   1  ladder gate4bit-qwen  (Qwen2.5-VL-7B, bfloat16)       -> runs_4bit/qwen/gate4bit
#      CHECKPOINT: prints the table; stops if a rung is BROKEN by unparseable POPE
#      (>20%) or looping captions (CONTINUE_ON_BROKEN=1 to go on anyway)
#   2  ladder gate4bit       (LLaVA-1.5-7B, float16)          -> runs_4bit/llava/gate4bit
#   3  LoRAS-T correctors: reuse runs*/lorast*/ (w3a16 RTN, +QA), else calibrate one
#   4  ladder extras: the real-kernel rows + LoRAS-T (bnb-nf4, Qwen AWQ), for figure (b)
#   5  15_latency.py both models                              -> runs_4bit/latency/
#   6  figures, SUMMARY.md, copy of the small files to obsidian-docs/results/
#
# Overrides: PY, COCO, ONLY=qwen|llava, SKIP_LATENCY=1, REUSE_GATE2=<ladder.json> (copy
# identical rungs instead of re-running; default off: the anchors are re-run),
# LAT_REPEATS (default 5), PUBLISH_DIR, CONTINUE_ON_BROKEN=1.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${PY:-}" ]; then
  for c in .venv-phoenix/bin/python .venv/bin/python; do [ -x "$c" ] && { PY="$c"; break; }; done
  PY="${PY:-python3}"
fi
COCO="${COCO:-data/coco-mini}"
OUT="runs_4bit"
LOG="$OUT/logs"
mkdir -p "$LOG" "$OUT/lorast"
LAT_REPEATS="${LAT_REPEATS:-5}"
PUBLISH_DIR="${PUBLISH_DIR:-../obsidian-docs/results/4bit-latency-$(date +%F)}"
EVAL="--n-pope-images 100 --n-chair-images 300 --max-new-tokens 128"
QWEN=Qwen/Qwen2.5-VL-7B-Instruct
LLAVA=llava-hf/llava-1.5-7b-hf
QWEN_AWQ="w4a16:backend=packed:ckpt=Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

say() { printf '\n[4bit] %s\n' "$*"; }
stamp() { printf '%s\t%s\t%s\n' "$(date +%FT%T)" "$1" "$2" >> "$LOG/timing.tsv"; }
want() { [ -z "${ONLY:-}" ] || [ "$ONLY" = "$1" ]; }

# ------------------------------------------------------------------ 0 env ---- #
say "0/6 environment (python: $PY)"
[ -x "$PY" ] || command -v "$PY" >/dev/null || { echo "no $PY: run bash scripts/setup_env.sh"; exit 1; }
if [ -z "${HF_HOME:-}" ]; then
  echo "[4bit] WARNING: HF_HOME is not set; ~14 GB per model will go to ~/.cache/huggingface."
  echo "        Put it on persistent storage: HF_HOME=/persistent/hf bash scripts/run_4bit.sh"
fi
if [ ! -f "$COCO/splits.json" ]; then
  say "fetching data into $COCO"
  "$PY" scripts/fetch_data.py --root "$COCO" --preset full
fi
"$PY" -c "import pynvml" 2>/dev/null || "$PY" -m pip install --quiet pynvml
{
  echo "== $(date +%FT%T) dependency check"
  if ! "$PY" -c "import awq" 2>/dev/null; then
    # autoawq 0.2.9 is the last release; --no-deps keeps it from replacing torch /
    # transformers 4.53.3. transformers' AWQ path needs only awq.modules.linear.
    echo "installing autoawq==0.2.9 --no-deps"
    "$PY" -m pip install --quiet --no-deps autoawq==0.2.9 || echo "autoawq install FAILED"
  fi
  "$PY" - <<'EOF' || true
import importlib.metadata as md
for p in ("torch", "transformers", "bitsandbytes", "autoawq", "autoawq-kernels", "triton", "pynvml", "nvidia-ml-py"):
    try:
        print(f"  {p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p}: not installed")
try:
    from awq.modules.linear.gemm import WQLinear_GEMM  # noqa: F401
    print("  awq WQLinear_GEMM import: OK")
except Exception as e:                                  # noqa: BLE001
    print(f"  awq WQLinear_GEMM import: FAILED ({type(e).__name__}: {e})")
EOF
} 2>&1 | tee -a "$OUT/env_install.log"
grep -q "WQLinear_GEMM import: OK" "$OUT/env_install.log" || \
  echo "[4bit] WARNING: AWQ kernels not importable; the Qwen AWQ rung will fail (see $OUT/env_install.log)"
"$PY" scripts/check_env.py --coco-root "$COCO" || { echo "[4bit] check_env failed"; exit 1; }

say "budget estimate: ~40 min per ladder rung (300 captions + 1800 POPE), 23 Qwen + 22 LLaVA"
say "rungs + 3 LoRAS-T extras -> ~32 GPU-h for ladders; latency ~5 GPU-h. REUSE_GATE2 saves ~5 h."

REUSE=()
[ -n "${REUSE_GATE2:-}" ] && REUSE=(--reuse-from "$REUSE_GATE2")

ladder() {  # model dtype name preset [extra]
  local model=$1 dtype=$2 name=$3 preset=$4 extra=${5:-}
  local t0=$SECONDS
  "$PY" scripts/00_precision_ladder.py --coco-root "$COCO" --model-id "$model" --dtype "$dtype" \
    --preset "$preset" ${extra:+--ladder-extra "$extra"} $EVAL \
    --out "$OUT/$name/gate4bit" --resume "${REUSE[@]}" 2>&1 | tee -a "$LOG/ladder_$name.log"
  stamp "ladder_$name${extra:+_extras}" "$((SECONDS - t0))s"
}

check_health() {  # name
  "$PY" - "$OUT/$1/gate4bit/ladder_report.json" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
bad = {k: v["health_reasons"] for k, v in rep["summary"].items()
       if v["verdict"].startswith("BROKEN")
       and any("unparseable" in r or "looping" in r for r in v["health_reasons"])}
for k, v in bad.items():
    print(f"[4bit] BROKEN: {k}: {'; '.join(v)}")
sys.exit(3 if bad else 0)
EOF
}

# ------------------------------------------------------------- 1 Qwen ladder ---- #
if want qwen; then
  say "1/6 Qwen ladder (gate4bit-qwen)"
  ladder "$QWEN" bfloat16 qwen gate4bit-qwen
  say "CHECKPOINT: Qwen ladder table above (also $OUT/qwen/gate4bit/ladder_report.json)"
  if ! check_health qwen; then
    if [ "${CONTINUE_ON_BROKEN:-0}" = "1" ]; then
      say "broken rungs found; continuing because CONTINUE_ON_BROKEN=1"
    else
      say "stopping at the checkpoint: a rung is broken (see above). Inspect it, then rerun"
      say "with CONTINUE_ON_BROKEN=1 to go on."
      exit 3
    fi
  fi
fi

# ------------------------------------------------------------ 2 LLaVA ladder ---- #
if want llava; then
  say "2/6 LLaVA ladder (gate4bit)"
  ladder "$LLAVA" float16 llava gate4bit
fi

# ---------------------------------------------------------- 3 LoRAS-T files ---- #
corrector() {  # model dtype name -> path on stdout
  local model=$1 dtype=$2 name=$3
  local found
  found=$("$PY" - "$model" <<'EOF'
import importlib.util, sys
from transformers import AutoConfig
spec = importlib.util.spec_from_file_location("lat", "scripts/15_latency.py")
lat = importlib.util.module_from_spec(spec); spec.loader.exec_module(lat)
h = AutoConfig.from_pretrained(sys.argv[1]).get_text_config().hidden_size
p, _ = lat.find_corrector(h)
print(p or "")
EOF
)
  if [ -n "$found" ]; then echo "$found"; return; fi
  local pt="$OUT/lorast/${name}_w3_rtn_qa.pt"
  if [ ! -f "$pt" ]; then
    echo "[4bit] no existing LoRAS-T corrector for $name: calibrating $pt on w3a16 (~1 h)" >&2
    echo "- No existing LoRAS-T corrector for $name: calibrated $pt (w3a16 RTN, --qa-per-image 2)." \
      >> "$OUT/DECISIONS.md"
    local t0=$SECONDS
    "$PY" scripts/12_calibrate_lorast.py --coco-root "$COCO" --model-id "$model" --dtype "$dtype" \
      --precision w3a16 --qa-per-image 2 --out "$pt" >> "$LOG/lorast_$name.log" 2>&1
    stamp "lorast_calib_$name" "$((SECONDS - t0))s"
  fi
  echo "$pt"
}

# --------------------------------------------- 4 ladder extras (+LoRAS-T rows) ---- #
if want qwen; then
  QPT=$(corrector "$QWEN" bfloat16 qwen)
  say "3/6 Qwen LoRAS-T corrector: $QPT"
  say "4/6 Qwen extras: real-kernel rows + LoRAS-T"
  ladder "$QWEN" bfloat16 qwen gate4bit-qwen "bnb-nf4+lorast=$QPT,$QWEN_AWQ+lorast=$QPT"
fi
if want llava; then
  LPT=$(corrector "$LLAVA" float16 llava)
  say "3/6 LLaVA LoRAS-T corrector: $LPT"
  say "4/6 LLaVA extras: real-kernel rows + LoRAS-T"
  ladder "$LLAVA" float16 llava gate4bit "bnb-nf4+lorast=$LPT"
fi

# ---------------------------------------------------------------- 5 latency ---- #
if [ "${SKIP_LATENCY:-0}" != "1" ]; then
  if want qwen; then
    say "5/6 latency: Qwen"
    t0=$SECONDS
    "$PY" scripts/15_latency.py --coco-root "$COCO" --model-id "$QWEN" --dtype bfloat16 \
      --lorast "$QPT" --repeats "$LAT_REPEATS" --out "$OUT/latency" --name qwen --resume \
      2>&1 | tee -a "$LOG/latency_qwen.log"
    stamp latency_qwen "$((SECONDS - t0))s"
  fi
  if want llava; then
    say "5/6 latency: LLaVA"
    t0=$SECONDS
    "$PY" scripts/15_latency.py --coco-root "$COCO" --model-id "$LLAVA" --dtype float16 \
      --lorast "$LPT" --repeats "$LAT_REPEATS" --out "$OUT/latency" --name llava --resume \
      2>&1 | tee -a "$LOG/latency_llava.log"
    stamp latency_llava "$((SECONDS - t0))s"
  fi
fi

# ----------------------------------------------------- 6 figures + summary ---- #
say "6/6 figures, SUMMARY.md, publish -> $PUBLISH_DIR"
"$PY" scripts/16_figures_4bit.py --runs "$OUT" --out "$OUT/figures"
"$PY" scripts/17_summary_4bit.py --runs "$OUT" --publish "$PUBLISH_DIR"
say "done. GPU wall time per step: $LOG/timing.tsv; per rung: SUMMARY.md"
