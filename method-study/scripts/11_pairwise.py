#!/usr/bin/env python
"""Paired comparisons between any two ladder rungs (not only against fp16).

Reuses the ladder's report (same metrics, same image-clustered bootstrap) with a
chosen reference, on the per-question / per-caption data a ladder run saved. CPU only.

    # is matched noise worse than RTN?  does multimodal calibration beat wiki calibration?
    python scripts/11_pairwise.py --ladder runs/gate2/ladder.json --ref w3a16 \\
        --rungs "w3a16:method=noise,w3a16:wtok=text,w3a16:method=gptq:calib=wiki"
    python scripts/11_pairwise.py --ladder runs/gate2/ladder.json --ref w3a16:method=gptq:calib=wiki \\
        --rungs "w3a16:method=gptq"
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ladder", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--rungs", default="", help="comma list; empty = all others")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("ladder", ROOT / "scripts" / "00_precision_ladder.py")
    lad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lad)
    data = json.loads(Path(a.ladder).read_text())
    table = data["table"]
    rungs = [r.strip() for r in a.rungs.split(",") if r.strip()] or \
        [r for r in table if r != a.ref]
    missing = [r for r in [a.ref] + rungs if r not in table]
    if missing:
        raise SystemExit(f"not in {a.ladder}: {missing}\navailable: {list(table)}")
    sub = {r: table[r] for r in [a.ref] + rungs}
    splits = data["config"]["pope_splits"].split(",")
    out = Path(a.out) if a.out else Path(a.ladder).parent / f"pairwise__{a.ref.replace(':', '_').replace('=', '')}"
    out.mkdir(parents=True, exist_ok=True)
    lad.report(sub, list(sub), splits, a.n_boot, out)


if __name__ == "__main__":
    main()
