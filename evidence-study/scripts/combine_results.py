"""Build the combined ablation (ablation.json, summary.md, figures) from the
per-rung results on disk, without loading any model.

Each rung runs in its own process, because process exit is the only thing that
reliably returns a rung's GPU memory: the GPTQ path keeps a reference that
gc.collect() cannot reach, so loading rungs back to back in one process runs
out of memory. Each run's own ablation.json therefore covers only its rung.
This mirrors the tail of cli.main over every rung present, in ladder order.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/combine_results.py --root .
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from experiments.analysis import export as export_mod  # noqa: E402
from experiments.analysis import metrics as metrics_mod  # noqa: E402
from experiments.analysis import plots as plots_mod  # noqa: E402

LADDER = ["fp16", "w8a8", "w4a16", "w4a8", "w4a4"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--results", default="results")
    args = p.parse_args()
    out = pathlib.Path(args.root) / args.results

    reports, extra, binned = [], {}, {}
    for v in LADDER:
        rp, cp = out / v / "report.json", out / v / "chair_captions.jsonl"
        if not rp.exists():
            print(f"[combine] {v}: no report.json, skipped")
            continue
        reports.append(json.loads(rp.read_text()))
        records = [json.loads(l) for l in cp.read_text().splitlines() if l.strip()] if cp.exists() else []
        rows = metrics_mod.mention_level_table(records)
        attn = [r["attention"] for r in rows if r["attention"] is not None]
        grounded = [r["grounded"] for r in rows if r["attention"] is not None]
        extra[v] = (attn, [1.0 if g else 0.0 for g in grounded])
        binned[v] = metrics_mod.binned_curve(attn, grounded)
        print(f"[combine] {v}: report + {len(records)} CHAIR captions, {len(attn)} object mentions")

    if not reports:
        sys.exit("[combine] no rung has a report.json yet")
    ablation = {"cells": reports}
    export_mod.write_ablation(reports, out)
    export_mod.write_summary_md(ablation, out)
    plots_mod.make_all(ablation, extra, binned, out / "figures")
    print(f"[combine] {len(reports)} rungs -> {out}/ablation.json, summary.md, figures/")


if __name__ == "__main__":
    main()
