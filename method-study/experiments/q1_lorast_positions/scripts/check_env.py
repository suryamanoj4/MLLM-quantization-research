#!/usr/bin/env python
"""Preflight check -- run this first on any new machine. Takes ~2 s, loads no model.

    python scripts/check_env.py
    python scripts/check_env.py --coco-root data/coco-mini

Exits non-zero if something would make the pipeline fail later, and prints the exact
command that fixes it.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phoenix.env import (MIN_VRAM_GB, TESTED_TRANSFORMERS, format_report, gpu_report,
                         hf_model_cached, python_env_problems)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco-root", default=None)
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--allow-cpu", action="store_true",
                   help="don't fail when no GPU is usable (smoke tests only)")
    a = p.parse_args()

    blockers, warnings = [], []
    print(f"[python] {sys.version.split()[0]}  ({sys.executable})")
    probs, notes = python_env_problems()
    for n in notes:
        print(f"  note  {n}")
    for pr in probs:
        print(f"  PROBLEM  {pr}")
        blockers.append("the running python is not your venv -- run: bash scripts/setup_env.sh")

    r = gpu_report()
    print("[gpu]\n" + format_report(r))
    if not r.ok and not a.allow_cpu:
        blockers.append("no usable CUDA device (see fix above)")
    if r.cuda_available and r.gpus:
        gb = max(g for _, g in r.gpus)
        if gb < MIN_VRAM_GB["eval"]:
            blockers.append(f"largest GPU has {gb:.0f} GB; LLaVA-1.5-7B fp16 needs "
                            f"~{MIN_VRAM_GB['eval']:.0f} GB")
        elif gb < MIN_VRAM_GB["calibrate"]:
            warnings.append(f"{gb:.0f} GB is enough for evaluation but LoRAS calibration "
                            f"keeps FP16 + quantized weights (~{MIN_VRAM_GB['calibrate']:.0f} GB)")

    try:
        import transformers
        v = transformers.__version__
        tested = v in TESTED_TRANSFORMERS
        print(f"[transformers] {v}  from {Path(transformers.__file__).parent}"
              f"{'' if tested else '   (untested version; tested: ' + ', '.join(TESTED_TRANSFORMERS) + ')'}")
        if not tested:
            warnings.append(f"transformers {v} is untested; `pip install -r requirements.txt` "
                            "pins 4.53.3")
    except ImportError:
        blockers.append("transformers not installed: pip install -r requirements.txt")

    for mod, why in (("tqdm", "progress bars"), ("ijson", "streaming annotation parse"),
                     ("PIL", "image loading"), ("matplotlib", "figures")):
        try:
            m = __import__(mod)
            extra = f" (backend {m.backend})" if mod == "ijson" else ""
            print(f"[deps] {mod} ok{extra}")
        except ImportError:
            (blockers if mod == "PIL" else warnings).append(f"{mod} missing ({why})")

    hf = Path.home() / ".cache" / "huggingface"
    import os
    hf = Path(os.environ.get("HF_HOME", hf))
    probe = hf if hf.exists() else Path.home()
    free = shutil.disk_usage(probe).free / 1e9
    cached, gb = hf_model_cached(a.model_id)
    print(f"[disk] {free:.0f} GB free at {probe}; {a.model_id} "
          + (f"already cached ({gb:.1f} GB)" if cached else "NOT cached (needs ~14 GB)"))
    if not cached and free < 15:
        blockers.append(f"{a.model_id} must be downloaded (~14 GB) but only "
                        f"{free:.0f} GB is free for the HF cache (set HF_HOME to a bigger disk)")

    if a.coco_root:
        root = Path(a.coco_root)
        ok = (root / "annotations").exists()
        sp = (root / "splits.json").exists()
        print(f"[data] {root}: annotations {'ok' if ok else 'MISSING'}, "
              f"splits.json {'ok' if sp else 'absent'}")
        if not ok:
            blockers.append(f"{root} has no annotations/ -- run scripts/fetch_data.py "
                            f"--root {root}")

    print()
    for w in warnings:
        print(f"[warn] {w}")
    for b in blockers:
        print(f"[BLOCKER] {b}")
    print("[check_env]", "PASS" if not blockers else "FAIL")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
