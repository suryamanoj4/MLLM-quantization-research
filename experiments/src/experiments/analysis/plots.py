from __future__ import annotations

import os
import pathlib

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"fp16": "#2e8b57", "w8": "#e6a817", "w4": "#c0392b"}
SPLITS = ("random", "popular", "adversarial")


def figure_f1(ablation: dict, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    variants = [c["variant"] for c in ablation["cells"]]
    x = np.arange(len(variants))
    width = 0.25
    for i, split in enumerate(SPLITS):
        vals = [c["pope"][split]["f1"] for c in ablation["cells"]]
        axes[0].bar(x + i * width, vals, width, label=split)
    axes[0].set_xticks(x + width, variants)
    axes[0].set_ylabel("POPE F1")
    axes[0].set_title("H1: hallucination vs precision")
    axes[0].legend()

    chair_s = [c["chair"]["chair_s"] for c in ablation["cells"]]
    chair_i = [c["chair"]["chair_i"] for c in ablation["cells"]]
    axes[1].bar(x - width / 2, chair_s, width, label="CHAIR_s")
    axes[1].bar(x + width / 2, chair_i, width, label="CHAIR_i")
    axes[1].set_xticks(x, variants)
    axes[1].set_ylabel("rate (lower better)")
    axes[1].set_title("CHAIR")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "f1_hallucination_vs_precision.png", dpi=150)
    plt.close(fig)


def figure_f2_decay(ablation: dict, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cell in ablation["cells"]:
        profile = cell["attention"]["chair"].get("decile_profile")
        if not profile:
            continue
        ax.plot(range(1, len(profile) + 1), profile, marker="o", label=cell["variant"], color=PALETTE.get(cell["variant"]))
    ax.set_xlabel("decile of generation")
    ax.set_ylabel("visual attention mass")
    ax.set_title("H4: fallback timeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "f2_fallback_timeline.png", dpi=150)
    plt.close(fig)


def figure_f3_scatter(ablation: dict, cells_extra: dict, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, data in cells_extra.items():
        attn, grounded = data
        ax.scatter(attn, grounded, s=8, alpha=0.4, label=variant, color=PALETTE.get(variant, None))
        order = np.argsort(attn)
        xs = np.asarray(attn)[order]
        slope, intercept = np.polyfit(attn, grounded, 1)
        ax.plot(xs, slope * xs + intercept, color=PALETTE.get(variant, None))
    ax.set_xlabel("mention-level visual attention mass")
    ax.set_ylabel("grounded")
    ax.set_title("H3: attention vs grounding")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "f3_attention_hallucination.png", dpi=150)
    plt.close(fig)


def figure_f4_binned(ablation: dict, cells_binned: dict, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, bins in cells_binned.items():
        xs = [b["attention"] for b in bins]
        ys = [b["hallucination_rate"] for b in bins]
        ax.plot(xs, ys, marker="o", label=variant, color=PALETTE.get(variant, None))
    ax.set_xlabel("attention decile (mean mass)")
    ax.set_ylabel("hallucination rate")
    ax.set_title("binned grounding curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "f4_binned_grounding.png", dpi=150)
    plt.close(fig)


def figure_f5_entropy(ablation: dict, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cell in ablation["cells"]:
        e = cell["attention"]["chair"]["mean_entropy"]
        ax.axvline(e, label=f"{cell['variant']} ({e:.2f})", color=PALETTE.get(cell["variant"]), linestyle="--")
    ax.set_xlabel("mean attention entropy")
    ax.set_title("H2: attention diffusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "f5_attention_entropy.png", dpi=150)
    plt.close(fig)


def make_all(ablation: dict, cells_extra: dict, cells_binned: dict, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    figure_f1(ablation, out)
    figure_f2_decay(ablation, out)
    figure_f3_scatter(ablation, cells_extra, out)
    figure_f4_binned(ablation, cells_binned, out)
    figure_f5_entropy(ablation, out)