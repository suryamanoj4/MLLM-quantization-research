from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from experiments.analysis import metrics as m
from experiments.analysis import plots
from experiments.data import coco
from experiments.data import datasets as ds
from experiments.experiment import chair_utils as cu

FAIL = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        FAIL.append(name)


# --- chair_utils ---
mentions = cu.extract_mentions(
    "A man and a woman sit on a bench next to a hot dog stand, "
    "while a dog plays with a frisbee near a parked car."
)
check("chair mentions set", mentions == {"person", "bench", "hot dog", "dog", "frisbee", "car"})
check("80 classes", len(cu.COCO_CLASSES) == 80)
spans = cu.extract_mentions_with_spans("the red bus with a zebra")
check("spans", [(c, s, e) for c, s, e in spans] == [("bus", 8, 11), ("zebra", 19, 24)])

# --- metrics ---
recs = [
    {"label": "yes", "answer": "yes", "correct": True},
    {"label": "yes", "answer": "no", "correct": False},
    {"label": "no", "answer": "no", "correct": True},
    {"label": "no", "answer": "yes", "correct": False},
    {"label": "yes", "answer": "yes", "correct": True},
]
pm = m.pope_metrics(recs)
check("pope acc", abs(pm["accuracy"] - 0.6) < 1e-9)
check("pope f1", abs(pm["f1"] - 2 / 3) < 1e-9)
check("pope yes_ratio", abs(pm["yes_ratio"] - 0.6) < 1e-9)

chair = m.chair_metrics(
    [
        {"n_mentions": 4, "n_hallucinated": 1},
        {"n_mentions": 5, "n_hallucinated": 0},
    ]
)
check("chair_s", abs(chair["chair_s"] - 0.5) < 1e-9)
check("chair_i", abs(chair["chair_i"] - 1 / 9) < 1e-9)

rng = __import__("random").Random(0)
attn = [0.05 + 0.4 * rng.random() for _ in range(120)]
grounded = [a > 0.3 for a in attn]
pb = m.point_biserial(attn, grounded, iters=200)
check("point-biserial positive", pb["r"] > 0.3)
check("binned curve", len(m.binned_curve(attn, grounded, 10)) == 10)
bins = m.binned_curve(attn, grounded, 10)
check("binned monotone", bins[0]["hallucination_rate"] >= bins[-1]["hallucination_rate"])

# --- datasets resamplers ---
qs = [{"image": f"i{j % 3}.jpg", "text": f"q{j}", "label": "yes"} for j in range(18)]
out = ds.resample_pope(qs, n_images=1, seed=7)
check("resample keeps blocks", len(out) == 6 and len({q["image"] for q in out}) == 1)

# --- coco ---
inst = {1: {"person"}, 2: {"dog", "person"}, 3: {"car"}}
fns = coco.sample_val2014_filenames(inst, 2, seed=0)
check("coco filename format", all(f.startswith("COCO_val2014_") for f in fns))
check("coco roundtrip", all(coco.filename_to_image_id(f) == int(f.split("_")[-1].split(".")[0]) for f in fns))

# --- plots ---
ablation = {
    "cells": [
        {
            "variant": "fp16",
            "pope": {s: {"f1": 0.87 - 0.01 * i} for i, s in enumerate(("random", "popular", "adversarial"))},
            "chair": {"chair_s": 0.12, "chair_i": 0.05},
            "attention": {
                "chair": {"mean_mass": 0.25, "mean_entropy": 4.1, "decile_profile": [0.3] * 10},
                "pope": {},
            },
        },
        {
            "variant": "w8",
            "pope": {s: {"f1": 0.84 - 0.01 * i} for i, s in enumerate(("random", "popular", "adversarial"))},
            "chair": {"chair_s": 0.20, "chair_i": 0.09},
            "attention": {
                "chair": {"mean_mass": 0.19, "mean_entropy": 4.6, "decile_profile": [0.24] * 10},
                "pope": {},
            },
        },
        {
            "variant": "w4",
            "pope": {s: {"f1": 0.78 - 0.01 * i} for i, s in enumerate(("random", "popular", "adversarial"))},
            "chair": {"chair_s": 0.32, "chair_i": 0.16},
            "attention": {
                "chair": {"mean_mass": 0.12, "mean_entropy": 5.2, "decile_profile": [0.15] * 10},
                "pope": {},
            },
        },
    ]
}
with tempfile.TemporaryDirectory() as td:
    out_dir = pathlib.Path(td) / "figures"
    plots.make_all(
        ablation,
        {"fp16": ([0.3] * 50 + [0.1] * 50, [1.0] * 50 + [0.0] * 50)},
        {"fp16": [{"attention": 0.3, "hallucination_rate": 0.2, "n": 10} for _ in range(10)]},
        out_dir,
    )
    files = sorted(p.name for p in out_dir.glob("*.png"))
    check("figures written", files == [
        "f1_hallucination_vs_precision.png",
        "f2_fallback_timeline.png",
        "f3_attention_hallucination.png",
        "f4_binned_grounding.png",
        "f5_attention_entropy.png",
    ])

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")