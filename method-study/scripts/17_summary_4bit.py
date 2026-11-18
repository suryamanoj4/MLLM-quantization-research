#!/usr/bin/env python
"""Write runs_4bit/SUMMARY.md from the 4-bit study's outputs, optionally publish them.

    python scripts/17_summary_4bit.py --runs runs_4bit
    python scripts/17_summary_4bit.py --runs runs_4bit \\
        --publish ../obsidian-docs/results/4bit-latency-$(date +%F)

SUMMARY.md: versions and GPU, GPU-hours per rung, the full ladder table per model
(paired deltas vs FP16 with unadjusted * and Benjamini-Hochberg + markers), the latency
tables, the figures, and every deviation recorded in runs_4bit/DECISIONS.md.

--publish copies only small files (ladder_report.json per model, latency/*.json and
*.md, figures, SUMMARY.md); anything over 5 MB is refused, and ladder.json files are
never copied. DECISIONS.md is a local working file and is never published.
"""
import argparse
import json
import shutil
from pathlib import Path

MODELS = {"llava": "LLaVA-1.5-7B (float16)", "qwen": "Qwen2.5-VL-7B-Instruct (bfloat16)"}
MAX_BYTES = 5 * 2 ** 20


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def d(cells: dict, k: str, pct: bool = False) -> str:
    c = cells.get(k)
    if not isinstance(c, dict):
        return "-"
    v = 100 * c["delta"] if pct else c["delta"]
    return (f"{v:+.1f}" if pct else f"{v:+.3f}") + ("\\*" if c.get("significant") else "") \
        + ("+" if c.get("significant_fdr") else "")


def ladder_section(name: str, lad: dict, rep: dict) -> list[str]:
    table, summ = lad["table"], rep["summary"]
    fdr = rep.get("fdr", {})
    out = [f"### {MODELS.get(name, name)}", "",
           f"Paired deltas vs `{rep['reference']}` (image-clustered bootstrap, 95% CI). "
           f"`*` = CI excludes 0 (unadjusted); `+` = Benjamini-Hochberg q < "
           f"{fdr.get('alpha', 0.05)} over {fdr.get('n_tests', '?')} tests. "
           "POPE pooled over 3 splits; coverage and CHAIR_i over the caption set.", "",
           "| rung | backend | bits | POPE F1 | dF1 | dRecall | dYes | CHAIR_i | dCHAIR_i | "
           "coverage | dCov | dLen | direction | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rung, row in table.items():
        s = summ.get(rung, {})
        cells = s.get("vs_reference", {})
        f1s = [row[f"pope_{sp}_f1"] for sp in row.get("splits", []) if f"pope_{sp}_f1" in row]
        f1 = sum(f1s) / len(f1s) if f1s else float("nan")
        eb = row.get("eff_wbits")
        dl = (cells.get("length") or {}).get("delta")
        out.append(f"| `{rung}` | {s.get('backend', row.get('backend', '?'))} | "
                   f"{f'{eb:.2f}' if eb is not None else '-'} | {f1:.3f} | {d(cells, 'pope_f1')} | "
                   f"{d(cells, 'pope_recall')} | {d(cells, 'pope_yes')} | {row['CHAIR_i']:.3f} | "
                   f"{d(cells, 'chair_i')} | {row['coverage']:.3f} | {d(cells, 'coverage')} | "
                   f"{f'{dl:+.1f}' if dl is not None else '-'} | "
                   f"{s.get('direction', '-')} | {s.get('verdict', '-')} |")
    return out + [""]


def gpu_hours(lads: dict, lats: dict) -> list[str]:
    out = ["| model | stage | rung / row | GPU-hours | note |", "|---|---|---|---|---|"]
    tot = 0.0
    for m, lad in lads.items():
        for rung, row in lad["table"].items():
            h = (row.get("wall_min") or 0) / 60
            reused = row.get("reused_from")
            if not reused:
                tot += h
            out.append(f"| {m} | ladder | `{rung}` | {h:.2f} | "
                       f"{'reused from ' + reused if reused else ''} |")
    for m, lat in lats.items():
        for name, row in lat["rows"].items():
            h = (row.get("wall_min") or 0) / 60
            tot += h
            out.append(f"| {m} | latency | `{name}` | {h:.2f} | |")
    out.append(f"| **total** | | | **{tot:.1f}** | measurement time only; model loads, "
               "calibration and PTQ are inside each ladder rung's wall time |")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs_4bit")
    ap.add_argument("--publish", default="")
    a = ap.parse_args()
    runs = Path(a.runs)
    lads, reps, lats = {}, {}, {}
    for m in MODELS:
        lad = load(runs / m / "gate4bit" / "ladder.json")
        rep = load(runs / m / "gate4bit" / "ladder_report.json")
        if lad and rep:
            lads[m], reps[m] = lad, rep
        lat = load(runs / "latency" / f"{m}.json")
        if lat:
            lats[m] = lat

    L = ["# 4-bit settings and real-kernel latency: summary", ""]
    L += ["## Environment", ""]
    vs = {**{f"ladder/{m}": v.get("versions", {}) for m, v in lads.items()},
          **{f"latency/{m}": v.get("versions", {}) for m, v in lats.items()}}
    if vs:
        keys = ["gpu", "gpu_mem_gb", "driver", "cuda", "torch", "transformers", "bitsandbytes",
                "autoawq", "gptqmodel", "accelerate", "nvidia-ml-py", "python"]
        L += ["| | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
        for k in keys:
            vals = []
            for v in vs.values():
                x = v.get(k)
                vals.append(f"{x:.0f}" if isinstance(x, float) else str(x))
            L.append(f"| {k} | " + " | ".join(vals) + " |")
    else:
        L.append("_no outputs found under " + str(runs) + "_")
    L += ["", "## GPU-hours per rung", ""] + gpu_hours(lads, lats)
    L += ["", "## Precision ladder (gate4bit)", "",
          "Rows tagged `sim` are simulated (fake-quant, dequantized FP16 matmuls): their quality "
          "is meaningful, their speed is not. `bnb` and `packed` rows run real 4-bit kernels. "
          "Bits = storage bits per weight incl. scales / zeros.", ""]
    for m in lads:
        L += ladder_section(m, lads[m], reps[m])
    L += ["## Latency (real kernels only)", ""]
    for m in lats:
        md = runs / "latency" / f"{m}.md"
        L += [md.read_text() if md.exists() else f"_{md} missing_", ""]
        for n in lats[m].get("notes", []):
            L.append(f"- note ({m}): {n}")
        L.append("")
    figs = sorted((runs / "figures").glob("*.png"))
    L += ["## Figures", ""] + [f"![{f.stem}](figures/{f.name})" for f in figs] + [""]
    dec = runs / "DECISIONS.md"
    L += ["## Deviations from the plan and decisions", "",
          (dec.read_text().split("\n", 1)[1].strip() if dec.exists()
           else "_runs_4bit/DECISIONS.md missing_"), ""]
    out = runs / "SUMMARY.md"
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)

    if a.publish:
        dst = Path(a.publish)
        dst.mkdir(parents=True, exist_ok=True)
        files = [(runs / m / "gate4bit" / "ladder_report.json", dst / f"{m}_ladder_report.json")
                 for m in reps]
        files += [(p, dst / "latency" / p.name) for p in sorted((runs / "latency").glob("*.json"))
                  + sorted((runs / "latency").glob("*.md"))]
        files += [(p, dst / "figures" / p.name) for p in figs]
        files += [(out, dst / "SUMMARY.md")]
        for src, to in files:
            if src.name == "ladder.json":
                continue
            if src.stat().st_size > MAX_BYTES:
                print(f"[publish] skipped {src} ({src.stat().st_size / 2**20:.1f} MB > 5 MB)")
                continue
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, to)
            print(f"[publish] {src} -> {to}")


if __name__ == "__main__":
    main()
