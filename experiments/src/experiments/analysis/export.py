from __future__ import annotations

import json
import pathlib


def save_json(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def save_jsonl(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


def cell_report(cell: dict) -> dict:
    return {
        "variant": cell["variant"],
        "pope": cell["pope"],
        "chair": cell["chair"],
        "attention": cell["attention"],
        "dkl": cell["dkl"],
    }


def write_cell(cell: dict, out_dir: pathlib.Path) -> None:
    vdir = out_dir / cell["variant"]
    save_json(cell_report(cell), vdir / "report.json")
    for split, rs in cell["pope_records"].items():
        save_jsonl(rs, vdir / f"pope_{split}.jsonl")
        if cell["prior"]:
            save_jsonl(cell["prior"][split], vdir / f"pope_{split}_prior.jsonl")
    save_jsonl(cell["chair_records"], vdir / "chair_captions.jsonl")


def write_ablation(reports: list[dict], out_dir: pathlib.Path) -> None:
    save_json({"cells": reports}, out_dir / "ablation.json")


def write_summary_md(ablation: dict, out_dir: pathlib.Path) -> None:
    lines = ["# Ablation Summary", ""]
    for cell in ablation["cells"]:
        lines.append(f"## {cell['variant']}")
        for split, m in cell["pope"].items():
            lines.append(
                f"- POPE-{split}: F1 = {m['f1']:.3f}, acc = {m['accuracy']:.3f}, yes-ratio = {m['yes_ratio']:.3f}"
            )
        c = cell["chair"]
        lines.append(f"- CHAIR: s = {c['chair_s']:.3f}, i = {c['chair_i']:.3f}")
        a = cell["attention"]
        lines.append(f"- Attention: mean mass = {a['chair']['mean_mass']:.4f}, entropy = {a['chair']['mean_entropy']:.3f}")
        if cell.get("dkl_mean"):
            lines.append(f"- DKL mean = {cell['dkl_mean']:.4f}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))