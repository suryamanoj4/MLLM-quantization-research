#!/usr/bin/env bash
# Build an isolated Python environment for Phoenix that can actually use the GPU.
#
#   bash scripts/setup_env.sh
#
# Why a script rather than "pip install ...":
#   * On Ubuntu 23.04+ the system pip refuses to install (PEP 668,
#     "externally-managed-environment"), so every install has to go into a venv.
#   * A shell prompt showing "(.venv)" does not guarantee `python` is the venv's --
#     a moved or half-built venv leaves the prompt but runs /usr/bin/python. This
#     script never relies on PATH: it calls $VENV/bin/python explicitly.
#   * The venv is created WITHOUT --system-site-packages, so a torch that was
#     `pip install --user`-ed into ~/.local (built for a newer CUDA than this
#     driver supports) can no longer leak in.
#   * torch 2.6 + its CUDA libraries are 5.2 GB installed (measured). The script
#     checks free space before downloading anything and says what can be reclaimed.
#
# Overrides:  VENV=path  PYTHON=python3.12  TORCH_SPEC="torch==2.6.0 torchvision==0.21.0"
#             TORCH_INDEX=https://download.pytorch.org/whl/cu118   FORCE=1 (skip space check)
set -euo pipefail

cd "$(dirname "$0")/.."
VENV="${VENV:-.venv-phoenix}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0 torchvision==0.21.0}"
NEED_VENV_GB=6.5          # 5.2 GB measured + transformers & friends + headroom
NEED_TMP_GB=3.0           # pip stages ~2.7 GB of wheels in $TMPDIR during install

say()  { printf '[setup] %s\n' "$*"; }
die()  { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }
gb_free() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%.1f", $4/1048576}'; }
dev_of()  { df -Pk "$1" 2>/dev/null | awk 'NR==2 {print $1}'; }
# report sizes of whichever of the given paths exist; never fails (set -e safe)
size_of() {
  local p out=""
  for p in "$@"; do
    [ -e "$p" ] || continue
    out="$out$(du -sh "$p" 2>/dev/null | awk '{print $1 "\t" $2}')"$'\n'
  done
  [ -n "$out" ] && printf '%s' "$out" || echo "(none found)"
  return 0
}

# ---------------------------------------------------------------- python ---- #
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
  done
fi
[ -n "$PY" ] || die "no python3 found"
PYV="$("$PY" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || die "$PY is Python $PYV; need >= 3.10"
say "base interpreter: $PY (Python $PYV)"

if [ -n "${VIRTUAL_ENV:-}" ] && [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
  say "note: your prompt says a venv is active (VIRTUAL_ENV=$VIRTUAL_ENV)"
  say "      but $VIRTUAL_ENV/bin/python does not exist -- that is why 'python'"
  say "      and 'pip' were the system ones. Run 'deactivate' after this script."
fi

# ------------------------------------------------------------ torch build ---- #
DRV_CUDA=""
if command -v nvidia-smi >/dev/null 2>&1; then
  DRV_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
fi
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }
if [ -z "${TORCH_INDEX:-}" ]; then
  if [ -z "$DRV_CUDA" ]; then
    say "WARNING: nvidia-smi not found; installing the default (CUDA 12.4) torch build"
    TORCH_INDEX=""
  elif ver_ge "$DRV_CUDA" 12.4; then
    TORCH_INDEX=""     # PyPI's torch 2.6.0 is the CUDA 12.4 build: runs on any >=12.4 driver
    say "driver supports CUDA $DRV_CUDA -> torch 2.6.0 from PyPI (CUDA 12.4 build)"
  elif ver_ge "$DRV_CUDA" 11.8; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    say "driver supports CUDA $DRV_CUDA -> torch 2.6.0 cu118 build"
  else
    die "driver only supports CUDA $DRV_CUDA; torch 2.6 needs >= 11.8 (update the driver)"
  fi
fi

# ------------------------------------------------------------ disk space ---- #
mkdir -p "$(dirname "$VENV")"
TMPD="${TMPDIR:-/tmp}"
VFREE="$(gb_free "$(dirname "$VENV")")"
TFREE="$(gb_free "$TMPD")"
if [ "$(dev_of "$(dirname "$VENV")")" = "$(dev_of "$TMPD")" ]; then
  NEED="$(awk -v a=$NEED_VENV_GB -v b=$NEED_TMP_GB 'BEGIN{print a+b}')"
  say "free space: ${VFREE} GB (venv and \$TMPDIR share a disk; need ~${NEED} GB)"
  SHORT="$(awk -v f="$VFREE" -v n="$NEED" 'BEGIN{print (f<n)?1:0}')"
else
  say "free space: ${VFREE} GB for the venv (need ~${NEED_VENV_GB}), ${TFREE} GB in $TMPD (need ~${NEED_TMP_GB})"
  SHORT="$(awk -v f="$VFREE" -v t="$TFREE" -v a=$NEED_VENV_GB -v b=$NEED_TMP_GB 'BEGIN{print (f<a||t<b)?1:0}')"
fi

if [ "$SHORT" = "1" ] && [ "${FORCE:-0}" != "1" ]; then
  echo
  say "NOT ENOUGH SPACE -- nothing was downloaded. What you could reclaim:"
  UL="$("$PY" -c 'import site;print(site.getusersitepackages())' 2>/dev/null)"
  echo "  pip download cache (always safe to delete):"
  size_of "${PIP_CACHE_DIR:-$HOME/.cache/pip}" | sed 's/^/      /'
  echo "  torch stack installed with --user in ~/.local (the CUDA-13 build this GPU can't use):"
  size_of "$UL"/torch "$UL"/nvidia "$UL"/triton "$UL"/torchvision 2>/dev/null | sed 's/^/      /'
  for v in .venv venv env; do
    if [ -d "$v" ] && [ "$v" != "$VENV" ]; then
      echo "  old venv in this repo:"; size_of "$v" | sed 's/^/      /'
    fi
  done
  echo
  echo "  Commands:"
  echo "      rm -rf ${PIP_CACHE_DIR:-~/.cache/pip}"
  echo "      rm -rf $UL/{torch,torchvision,triton,nvidia} $UL/torch-*.dist-info $UL/torchvision-*.dist-info $UL/triton-*.dist-info $UL/nvidia_*"
  echo "          ^ only if nothing else of yours uses that torch. If this home directory is"
  echo "            shared with other machines (e.g. an NFS home on a cluster), they would lose it."
  echo "      rm -rf .venv            # the broken venv, if you don't need it"
  echo
  echo "  Or put the environment and temp files on a bigger disk (check: df -h):"
  echo "      VENV=/some/big/disk/phoenix-venv TMPDIR=/some/big/disk/tmp bash scripts/setup_env.sh"
  echo "  Do NOT delete ~/.cache/huggingface -- the 14 GB LLaVA weights are already there."
  exit 1
fi

# ------------------------------------------------------------------ venv ---- #
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import sys;sys.exit(0 if sys.prefix!=sys.base_prefix else 1)' 2>/dev/null; then
  say "reusing existing venv $VENV"
else
  rm -rf "$VENV"
  say "creating venv $VENV (isolated: no system or ~/.local packages)"
  if ! "$PY" -m venv "$VENV" 2>/tmp/phoenix-venv-err.$$ || [ "${PHX_TEST_NO_ENSUREPIP:-0}" = "1" ]; then
    rm -rf "$VENV"
    say "venv module can't bootstrap pip here (python3-venv missing); bootstrapping pip from PyPI instead"
    "$PY" -m venv --without-pip "$VENV" || die "cannot create a venv at all: $(cat /tmp/phoenix-venv-err.$$)"
    # Bootstrap pip from its own wheel on PyPI (the same host the torch install
    # needs anyway): a pip wheel is runnable as `python pip.whl/pip install pip.whl`.
    # get-pip.py from bootstrap.pypa.io is only the second resort.
    "$VENV/bin/python" - <<'EOF' || die "could not bootstrap pip (PyPI and bootstrap.pypa.io both failed)"
import json, os, subprocess, sys, tempfile, urllib.request
tmp = tempfile.mkdtemp(prefix="phoenix-pip-")
try:
    meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/pip/json", timeout=60))
    whl = next(u for u in meta["urls"] if u["filename"].endswith("-py3-none-any.whl"))
    path = os.path.join(tmp, whl["filename"])
    urllib.request.urlretrieve(whl["url"], path)
    rc = subprocess.call([sys.executable, os.path.join(path, "pip"), "install",
                          "--no-cache-dir", "--quiet", path])
    if rc == 0:
        sys.exit(0)
    print(f"[setup] pip wheel bootstrap exited {rc}; trying get-pip.py", file=sys.stderr)
except Exception as e:
    print(f"[setup] PyPI pip wheel unavailable ({e}); trying get-pip.py", file=sys.stderr)
f = os.path.join(tmp, "get-pip.py")
urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", f)
sys.exit(subprocess.call([sys.executable, f, "--no-cache-dir", "--quiet"]))
EOF
  fi
  rm -f /tmp/phoenix-venv-err.$$
fi
VPY="$VENV/bin/python"
"$VPY" -c 'import site,sys; assert not site.ENABLE_USER_SITE, "user site leaks in"; print("[setup] venv python:", sys.executable)'

export PIP_DISABLE_PIP_VERSION_CHECK=1
"$VPY" -m pip install --no-cache-dir --quiet --upgrade pip

say "installing $TORCH_SPEC (~2.7 GB download, 5.2 GB on disk) ..."
if [ -n "${TORCH_INDEX:-}" ]; then
  "$VPY" -m pip install --no-cache-dir $TORCH_SPEC --index-url "$TORCH_INDEX"
else
  "$VPY" -m pip install --no-cache-dir $TORCH_SPEC
fi
say "installing requirements.txt ..."
"$VPY" -m pip install --no-cache-dir -r requirements.txt

echo
"$VPY" scripts/check_env.py --coco-root "${COCO:-data/coco-mini}" || {
  say "environment built, but check_env still reports a blocker (above)"; exit 1; }

echo
say "done. From now on either activate it:"
echo "      deactivate 2>/dev/null; source $VENV/bin/activate"
echo "  or skip activation entirely (immune to prompt/PATH confusion):"
echo "      PY=$VENV/bin/python bash run_all.sh"
