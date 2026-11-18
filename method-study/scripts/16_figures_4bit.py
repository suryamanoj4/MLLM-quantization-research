#!/usr/bin/env python
"""Figures for the 4-bit study (style of 06_figures.py), into runs_4bit/figures/.

    python scripts/16_figures_4bit.py --runs runs_4bit --out runs_4bit/figures

  fig_a_<model>_bits.png   coverage and CHAIR_i vs effective weight bits, every rung
                           (two panels, one measure each; FP16 as a reference line;
                           error bars = FP16 + the paired 95% CI of the change)
  fig_b_cost.png           coverage vs J/caption and vs weights GB for the real-kernel
                           rows (fp16 / bnb / packed), without (filled) and with
                           (hollow) LoRAS-T; one row of panels per model

Inputs: <runs>/<model>/gate4bit/ladder.json (+ ladder_report.json) and
<runs>/latency/<model>.json. Missing inputs are skipped with a note.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots in fixed order (validated: scripts/validate_palette.js, light surface;
# three slots are below 3:1 contrast, so every marker has a dark edge and key points are
# direct-labelled; the full numbers are in SUMMARY.md)
FAMILY_COLOR = {"RTN": "#2a78d6", "GPTQ": "#eb6834", "AWQ": "#1baf7a", "FP4": "#eda100",
                "W4A8/W4A4": "#e87ba4", "NF4": "#008300"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#d8d8d8"
MODELS = ("llava", "qwen")


def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, color=INK); ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, fontsize=11, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def base_spec(rung: str) -> str:
    return rung.partition("+lorast=")[0]


def has_lorast(rung: str) -> bool:
    return "+lorast=" in rung


def family(rung: str) -> str:
    s = base_spec(rung)
    if s == "fp16":
        return "FP16"
    if "nf4" in s:
        return "NF4"
    if "wfmt=nvfp4" in s or "wfmt=mxfp4" in s:
        return "FP4"
    m = re.fullmatch(r"w\d+a(\d+)", s.split(":")[0])
    if m and int(m.group(1)) < 16:
        return "W4A8/W4A4"
    if "backend=packed" in s:
        return "AWQ" if "awq" in s.lower() else "GPTQ"
    if "method=gptq" in s:
        return "GPTQ"
    if "method=awq" in s:
        return "AWQ"
    return "RTN"


def short(rung: str) -> str:
    s = base_spec(rung)
    if "backend=packed" in s:
        t = "AWQ (official)" if "awq" in s.lower() else s.split("ckpt=")[-1].split("/")[-1]
    elif s == "bnb-nf4":
        t = "bnb-nf4"
    else:
        t = (s.replace("method=", "").replace(":calib=wiki", "").replace(":grid=ref", "")
             .replace("targets=language+projector+vision", "+proj+vis")
             .replace("targets=language+projector", "+proj").replace("wfmt=", "")
             .replace("afmt=", "a-"))
    return t + (" +LoRAS-T" if has_lorast(rung) else "")


def latency_key(rung: str) -> str:
    """ladder rung -> 15_latency.py row name."""
    s = base_spec(rung)
    if "backend=packed" in s:
        name = re.sub(r"[^A-Za-z0-9.+-]+", "_", s.split("ckpt=")[-1]).strip("_")
    else:
        name = s
    return name + ("+lorast" if has_lorast(rung) else "")


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fig_bits(model: str, lad: dict, rep: dict | None, out: Path) -> Path | None:
    table = lad["table"]
    ref = table.get("fp16")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, key, cell, label in ((axes[0], "coverage", "coverage", "object coverage"),
                                 (axes[1], "CHAIR_i", "chair_i", "CHAIR_i")):
        if ref is not None:
            ax.axhline(ref[key], color=INK, lw=1.2, ls="--", zorder=1)
            ax.annotate("FP16", (1.0, ref[key]), xycoords=("axes fraction", "data"),
                        xytext=(-4, 4), textcoords="offset points", ha="right",
                        fontsize=8, color=INK)
        seen = set()
        for rung, row in table.items():
            if rung == "fp16" or has_lorast(rung) or row.get("eff_wbits") is None:
                continue
            fam = family(rung)
            real = row.get("backend") in ("bnb", "packed")
            # dodge families horizontally: many rungs share one bit width (e.g. gs=128)
            fams = list(FAMILY_COLOR)
            dx = 0.018 * ((fams.index(fam) if fam in fams else 0) - (len(fams) - 1) / 2)
            x, y = row["eff_wbits"] + dx, row[key]
            yerr = None
            vs = (rep or {}).get("summary", {}).get(rung, {}).get("vs_reference", {}).get(cell)
            if vs and ref is not None:
                lo, hi = ref[key] + vs["lo"], ref[key] + vs["hi"]
                yerr = [[max(y - lo, 0)], [max(hi - y, 0)]]
            ax.errorbar([x], [y], yerr=yerr, fmt="s" if real else "o", ms=8 if real else 7,
                        color=FAMILY_COLOR.get(fam, MUTED), mec=INK, mew=0.8, elinewidth=0.8,
                        ecolor=FAMILY_COLOR.get(fam, MUTED), alpha=0.95, capsize=0, zorder=3,
                        label=None if (fam, real) in seen else fam + (" (real kernels)" if real else ""))
            seen.add((fam, real))
            if real or rung.startswith("w3a16"):
                ax.annotate(short(rung), (x, y), textcoords="offset points",
                            xytext=(5, -10) if fam == "GPTQ" else (5, 3),
                            fontsize=7, color=MUTED)
        _style(ax, "effective weight bits (incl. scales / zeros)", label,
               f"{model}: {label} vs bits")
    h, l = axes[0].get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: list(FAMILY_COLOR).index(l[i].split(" (")[0])
                   if l[i].split(" (")[0] in FAMILY_COLOR else 99)
    axes[0].legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=8,
                   loc="lower right")
    fig.text(0.01, -0.02, "circles = simulated (backend=sim), squares = real kernels; "
             "bars: FP16 + paired 95% CI of the change (image-clustered bootstrap)",
             fontsize=7.5, color=MUTED)
    p = out / f"fig_a_{model}_bits.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def cost_points(lad: dict, lat: dict) -> list[dict]:
    pts = []
    for rung, row in lad["table"].items():
        if row.get("backend") not in ("fp16", "bnb", "packed"):
            continue
        lr = lat["rows"].get(latency_key(rung))
        if lr is None:
            continue
        pts.append({"rung": rung, "coverage": row["coverage"], "lorast": has_lorast(rung),
                    "j": (lr.get("j_caption") or {}).get("median"),
                    "gb": lr.get("weights_gb"), "fam": family(rung),
                    "base": latency_key(base_spec(rung))})
    return pts


def fig_cost(runs: Path, out: Path) -> Path | None:
    data = {}
    for m in MODELS:
        lad = load(runs / m / "gate4bit" / "ladder.json")
        lat = load(runs / "latency" / f"{m}.json")
        if lad and lat:
            data[m] = cost_points(lad, lat)
    data = {m: v for m, v in data.items() if v}
    if not data:
        return None
    fig, axes = plt.subplots(len(data), 2, figsize=(11, 4.2 * len(data)),
                             constrained_layout=True, squeeze=False)
    colors = {"FP16": INK, "NF4": FAMILY_COLOR["NF4"], "AWQ": FAMILY_COLOR["AWQ"],
              "GPTQ": FAMILY_COLOR["GPTQ"]}
    for r, (m, pts) in enumerate(data.items()):
        for c, (key, xl) in enumerate((("j", "energy per caption (J, median)"),
                                       ("gb", "weights (GB)"))):
            ax = axes[r][c]
            by_base = {}
            for p in pts:
                if p[key] is None:
                    continue
                col = colors.get(p["fam"], MUTED)
                ax.plot([p[key]], [p["coverage"]], "o", ms=9, mec=col if p["lorast"] else INK,
                        mfc="white" if p["lorast"] else col, mew=1.6 if p["lorast"] else 0.8,
                        zorder=3, label=("+LoRAS-T" if p["lorast"] else p["fam"]))
                by_base.setdefault(p["base"], []).append(p)
                ax.annotate(short(p["rung"]), (p[key], p["coverage"]), xytext=(6, 3),
                            textcoords="offset points", fontsize=7, color=MUTED)
            for grp in by_base.values():           # connect each row to its +LoRAS-T twin
                if len(grp) == 2 and all(g[key] is not None for g in grp):
                    a, b = sorted(grp, key=lambda g: g["lorast"])
                    ax.annotate("", (b[key], b["coverage"]), (a[key], a["coverage"]),
                                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1), zorder=2)
            _style(ax, xl, "object coverage", f"{m}: coverage vs {'energy' if key == 'j' else 'memory'}")
            h, l = ax.get_legend_handles_labels()
            uniq = {k: v for k, v in zip(l, h) if k != "+LoRAS-T"}
            from matplotlib.lines import Line2D
            uniq["+LoRAS-T (hollow)"] = Line2D([], [], ls="", marker="o", ms=9, mfc="white",
                                              mec=MUTED, mew=1.6)
            ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8)
    fig.text(0.01, -0.01, "real-kernel rows only; filled = without LoRAS-T, hollow = with "
             "(arrow). The LoRAS-T correctors were fitted on simulated W3 RTN.",
             fontsize=7.5, color=MUTED)
    p = out / "fig_b_cost.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs_4bit")
    ap.add_argument("--out", default="runs_4bit/figures")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for m in MODELS:
        lad = load(runs / m / "gate4bit" / "ladder.json")
        if lad is None:
            print(f"[skip] no {runs / m / 'gate4bit' / 'ladder.json'}")
            continue
        made.append(fig_bits(m, lad, load(runs / m / "gate4bit" / "ladder_report.json"), out))
    p = fig_cost(runs, out)
    if p is None:
        print("[skip] fig_b: needs both a ladder and a latency file for some model")
    else:
        made.append(p)
    for p in made:
        if p:
            print("wrote", p)


if __name__ == "__main__":
    main()
