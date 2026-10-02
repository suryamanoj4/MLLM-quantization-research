"""Environment preflight: catch GPU / driver / install problems before loading 14 GB.

The failure this exists for: a torch wheel built for a newer CUDA than the installed
NVIDIA driver supports. torch then imports fine, prints a warning, reports
`cuda.is_available() == False`, and the first `.to("cuda")` dies deep inside model
loading. Every check here is cheap and runs before any weights are touched.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

# torch 2.6.0 CUDA builds we recommend. cu124 is what PyPI's plain `torch==2.6.0`
# wheel is, so for any driver >= 12.4 no custom index is needed at all.
TORCH26_CUDA_BUILDS = ("11.8", "12.4")
TESTED_TRANSFORMERS = ("4.53.3", "5.17.0")
MIN_VRAM_GB = {"eval": 16.0, "calibrate": 30.0}


@dataclass
class GpuReport:
    torch_version: str = "?"
    torch_file: str = "?"
    torch_cuda_build: str | None = None
    cuda_available: bool = False
    driver_version: str | None = None
    driver_cuda: str | None = None
    gpus: list[tuple[str, float]] = field(default_factory=list)   # (name, GB)
    problems: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _v(s: str | None) -> tuple[int, ...]:
    if not s:
        return ()
    return tuple(int(x) for x in re.findall(r"\d+", s)[:2])


def parse_nvidia_smi(header: str, query: str) -> tuple[str | None, str | None, list]:
    drv = re.search(r"Driver Version:\s*([\d.]+)", header)
    cu = re.search(r"CUDA Version:\s*([\d.]+)", header)
    gpus = []
    for line in query.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            mem = re.findall(r"[\d.]+", parts[1])
            gpus.append((parts[0], float(mem[0]) / 1024 if mem else 0.0))
    return (drv.group(1) if drv else None), (cu.group(1) if cu else None), gpus


def _nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None, []
    try:
        header = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
        query = subprocess.run([exe, "--query-gpu=name,memory.total",
                                "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20).stdout
        return parse_nvidia_smi(header, query)
    except Exception:                                     # noqa: BLE001
        return None, None, []


def suggest_torch_build(driver_cuda: str | None) -> str | None:
    """torch-2.6 CUDA build for this driver: >= 12.4 -> 'cu124' (plain PyPI), else 'cu118'."""
    if not driver_cuda:
        return None
    ok = [b for b in TORCH26_CUDA_BUILDS if _v(b) <= _v(driver_cuda)]
    return f"cu{ok[-1].replace('.', '')}" if ok else None


def gpu_report() -> GpuReport:
    import warnings
    r = GpuReport()
    r.driver_version, r.driver_cuda, r.gpus = _nvidia_smi()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # we explain it ourselves
            import torch
            r.torch_version = torch.__version__
            r.torch_file = os.path.dirname(torch.__file__)
            r.torch_cuda_build = torch.version.cuda
            r.cuda_available = torch.cuda.is_available()
            if r.cuda_available and not r.gpus:
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    r.gpus.append((p.name, p.total_memory / 1024 ** 3))
    except ImportError:
        r.problems.append("torch is not installed in this interpreter")
        r.fixes.append("bash scripts/setup_env.sh")
        return r

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv and not r.torch_file.startswith(sys.prefix):
        r.notes.append(f"torch is loaded from {r.torch_file}, OUTSIDE the active venv "
                       f"({sys.prefix}) -- usually a `pip install --user` copy in "
                       "~/.local leaking in via include-system-site-packages. "
                       "Installing torch into the venv takes precedence over it.")

    if not r.cuda_available:
        if r.driver_cuda and r.torch_cuda_build and _v(r.torch_cuda_build) > _v(r.driver_cuda):
            want = suggest_torch_build(r.driver_cuda)
            r.problems.append(
                f"torch {r.torch_version} is built for CUDA {r.torch_cuda_build}, but the "
                f"NVIDIA driver ({r.driver_version}) only supports up to CUDA "
                f"{r.driver_cuda}. torch silently falls back to CPU.")
            if want:
                r.fixes.append(_torch_install_cmd(want))
        elif not r.gpus:
            r.problems.append("no NVIDIA GPU visible (nvidia-smi missing or no devices)")
        elif r.torch_cuda_build is None:
            r.problems.append(f"torch {r.torch_version} is a CPU-only build")
            r.fixes.append(_torch_install_cmd(suggest_torch_build(r.driver_cuda) or "cu124"))
        else:
            r.problems.append("torch sees no usable CUDA device "
                              "(check CUDA_VISIBLE_DEVICES and `nvidia-smi`)")
    return r


def _torch_install_cmd(build: str) -> str:
    base = "pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0"
    if build == "cu124":
        return base                                  # PyPI default build
    return f"{base} --index-url https://download.pytorch.org/whl/{build}"


# --------------------------------------------------------------------------- #
# interpreter / venv sanity
# --------------------------------------------------------------------------- #
def python_env_problems() -> tuple[list[str], list[str]]:
    """(problems, notes) about *which* python is running and whether pip can install."""
    import sysconfig
    problems, notes = [], []
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    ve = os.environ.get("VIRTUAL_ENV")
    if ve and os.path.realpath(ve) != os.path.realpath(sys.prefix):
        exists = os.path.exists(os.path.join(ve, "bin", "python"))
        problems.append(
            f"your shell says a venv is active (VIRTUAL_ENV={ve}) but this is "
            f"{sys.executable}, not that venv's python"
            + ("" if exists else f" -- {ve}/bin/python does not even exist (moved or "
               "half-built venv)")
            + ". So `python` and `pip` are the system ones.")
    if not in_venv:
        marker = os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")
        if os.path.exists(marker):
            notes.append("system python is externally managed (PEP 668): plain `pip "
                         "install` will refuse. Use `bash scripts/setup_env.sh`, which "
                         "builds an isolated venv.")
    return problems, notes


def hf_model_cached(model_id: str) -> tuple[bool, float]:
    """Is the model already in the HF cache? Returns (cached, GB on disk)."""
    home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache",
                                                  "huggingface"))
    hub = os.environ.get("HF_HUB_CACHE", os.path.join(home, "hub"))
    d = os.path.join(hub, "models--" + model_id.replace("/", "--"))
    if not os.path.isdir(d):
        return False, 0.0
    total, weights = 0, 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            fp = os.path.join(root, f)
            try:
                sz = os.stat(fp).st_size          # follows snapshot symlinks to blobs
            except OSError:
                continue
            if "blobs" in root:
                total += sz
            if f.endswith((".safetensors", ".bin")) and "snapshots" in root:
                weights += 1
    return weights > 0, total / 1e9


def format_report(r: GpuReport) -> str:
    lines = [f"  torch        {r.torch_version}  (CUDA build {r.torch_cuda_build or 'none'})",
             f"  loaded from  {r.torch_file}",
             f"  driver       {r.driver_version or '?'}  (supports CUDA "
             f"{r.driver_cuda or '?'})"]
    for name, gb in r.gpus:
        lines.append(f"  gpu          {name}, {gb:.0f} GB")
    lines.append(f"  cuda usable  {'yes' if r.cuda_available else 'NO'}")
    for n in r.notes:
        lines.append(f"  note         {n}")
    for p in r.problems:
        lines.append(f"  PROBLEM      {p}")
    if r.fixes:
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        lines.append("  fix:")
        if not in_venv or os.environ.get("VIRTUAL_ENV", sys.prefix) != sys.prefix:
            lines.append("      bash scripts/setup_env.sh        # builds an isolated venv "
                         "with the right torch")
        else:
            for f in r.fixes:
                lines.append(f"      {f}")
            lines.append("      pip install --no-cache-dir -r requirements.txt")
        lines.append("      python scripts/check_env.py      # re-check")
    return "\n".join(lines)


class GpuUnavailable(RuntimeError):
    pass


def require_cuda(device: str) -> None:
    """Raise a readable error *before* loading weights if `device` can't be used."""
    if not str(device).startswith("cuda"):
        return
    r = gpu_report()
    if not r.ok:
        raise GpuUnavailable(
            "a CUDA device was requested but cannot be used:\n" + format_report(r)
            + "\n\n(to deliberately run on CPU, pass --device cpu -- expect hours per "
            "evaluation cell for a 7B model)")
