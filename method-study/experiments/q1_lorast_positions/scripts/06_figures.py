#!/usr/bin/env python
"""Turn the run artefacts into the four figures the writeup needs.

    python scripts/06_figures.py --runs runs --out figures

  fig1_drift.png       layer-depth drift, FP16 vs quantized, at visual positions
  fig2_recovery.png    LoRAS rank-vs-recovery, with the full-rank linear ceiling
  fig3_ladder.png      POPE F1 / CHAIR across the precision ladder
  fig4_acab_sweep.png  CHAIR_i vs object coverage as alpha rises (the trade-off)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"a": "#3b6ea5", "b": "#c9622e", "c": "#4f8a63", "d": "#8a5fa8", "grid": "#d8d8d8"}


def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_drift(runs: Path, out: Path):
    files = sorted(runs.glob("drift__*/drift.json"))
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, f in enumerate(files):
        d = json.loads(f.read_text())
        tag = d.get("precision", f.parent.name)
        ax = axes[0]
        ax.plot([r["layer"] for r in d["drift"]["hidden"]],
                [r["cos"] for r in d["drift"]["hidden"]], marker="o", ms=3,
                label=tag, color=list(C.values())[i % 4])
        ax = axes[1]
        ax.plot([r["layer"] for r in d["drift"]["k"]],
                [r["rel_mse"] for r in d["drift"]["k"]], marker="o", ms=3,
                label=f"{tag} K", color=list(C.values())[i % 4])
        ax.plot([r["layer"] for r in d["drift"]["v"]],
                [r["rel_mse"] for r in d["drift"]["v"]], marker="s", ms=3, ls="--",
                label=f"{tag} V", color=list(C.values())[i % 4], alpha=0.6)
    _style(axes[0], "decoder layer", "cosine(FP16, quant)", "residual stream drift")
    _style(axes[1], "decoder layer", "relative MSE", "K / V projection error")
    axes[0].legend(frameon=False, fontsize=8); axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "fig1_drift.png", dpi=180)
    plt.close(fig)


def fig_recovery(runs: Path, out: Path):
    files = sorted(runs.glob("loras__*/loras_diagnostics.json"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    d = json.loads(files[0].read_text())
    for i, (k, v) in enumerate(sorted(d["diagnostics"].items())):
        if "curve" not in v or not k.endswith("|k"):
            continue
        c = v["curve"]
        ax.plot(range(1, len(c) + 1), c, lw=1.2, alpha=0.85,
                label=f"L{k.split('|')[0]}")
    ax.set_xscale("log")
    _style(ax, "rank r", "fraction of K error removed",
           "LoRAS rank vs recovery (flat tail = the linear ceiling)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(out / "fig2_recovery.png", dpi=180)
    plt.close(fig)


def fig_ladder(runs: Path, out: Path):
    f = runs / "ladder" / "ladder.json"
    if not f.exists():
        return
    t = json.loads(f.read_text())["table"]
    order = [p for p in ["fp16", "w8a8", "w4a16", "w4a8", "w4a4", "w2a4"] if p in t]
    fig, ax1 = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(order, [t[p].get("pope_adversarial_f1") for p in order], marker="o",
             color=C["a"], label="POPE-adv F1")
    ax1.plot(order, [t[p].get("pope_random_f1") for p in order], marker="s",
             color=C["c"], ls="--", label="POPE-rand F1")
    ax2.plot(order, [t[p].get("CHAIR_i") for p in order], marker="^",
             color=C["b"], label="CHAIR_i")
    _style(ax1, "precision", "POPE F1", "where the model actually breaks")
    ax2.set_ylabel("CHAIR_i")
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)
    fig.savefig(out / "fig3_ladder.png", dpi=180)
    plt.close(fig)


def fig_sweep(runs: Path, out: Path):
    files = sorted(runs.glob("acab_sweep__*/sweep.json"))
    if not files:
        return
    rows = json.loads(files[0].read_text())["rows"]
    taus = sorted({r["tau_pct"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, q in enumerate(taus):
        rs = sorted([r for r in rows if r["tau_pct"] == q], key=lambda r: r["alpha"])
        col = list(C.values())[i % 4]
        axes[0].plot([r["alpha"] for r in rs], [r["CHAIR_i"] for r in rs],
                     marker="o", color=col, label=f"tau=p{q:.0f}")
        axes[0].plot([r["alpha"] for r in rs], [r["coverage"] for r in rs],
                     marker="s", ls="--", color=col, alpha=0.6)
        axes[1].plot([r["coverage"] for r in rs], [r["CHAIR_i"] for r in rs],
                     marker="o", color=col, label=f"tau=p{q:.0f}")
        for r in rs:
            axes[1].annotate(f"{r['alpha']:g}", (r["coverage"], r["CHAIR_i"]),
                             fontsize=6, xytext=(3, 3), textcoords="offset points")
    _style(axes[0], "alpha", "CHAIR_i (solid) / coverage (dashed)",
           "raising alpha: hallucination vs how much is still described")
    _style(axes[1], "object coverage", "CHAIR_i", "the actual operating curve")
    axes[0].legend(frameon=False, fontsize=8); axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "fig4_acab_sweep.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_drift(runs, out); fig_recovery(runs, out)
    fig_ladder(runs, out); fig_sweep(runs, out)
    print("figures written to", out)
    for f in sorted(out.glob("*.png")):
        print("  ", f)


if __name__ == "__main__":
    main()
