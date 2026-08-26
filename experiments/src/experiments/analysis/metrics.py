from __future__ import annotations

import random

import numpy as np
from scipy import stats


def pope_metrics(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}
    correct = sum(1 for r in records if r["correct"])
    yes_true = sum(1 for r in records if r["label"] == "yes")
    pred_yes = sum(1 for r in records if r["answer"] == "yes")
    tp = sum(1 for r in records if r["label"] == "yes" and r["answer"] == "yes")
    fp = sum(1 for r in records if r["label"] == "no" and r["answer"] == "yes")
    fn = sum(1 for r in records if r["label"] == "yes" and r["answer"] == "no")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n,
        "accuracy": correct / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": pred_yes / n if n else 0.0,
        "label_yes_rate": yes_true / n if n else 0.0,
    }


def chair_metrics(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}
    captions_with_hallu = sum(1 for r in records if r["n_hallucinated"] > 0)
    total_mentions = sum(r["n_mentions"] for r in records)
    total_hallu = sum(r["n_hallucinated"] for r in records)
    return {
        "n_captions": n,
        "chair_s": captions_with_hallu / n,
        "chair_i": total_hallu / total_mentions if total_mentions else 0.0,
        "total_mentions": total_mentions,
        "total_hallucinated": total_hallu,
        "mentions_per_caption": total_mentions / n,
    }


def attention_summary_from_records(pope_records: dict[str, list[dict]], chair_records: list[dict]) -> dict:
    def agg(records: list[dict]) -> dict:
        masses = [r["attention"]["mean_mass"] for r in records if r.get("attention")]
        ents = [r["attention"]["mean_entropy"] for r in records if r.get("attention")]
        drifts = [r["attention"]["drift_rate"] for r in records if r.get("attention")]
        profiles = [r["attention"]["decile_profile"] for r in records if r.get("attention") and r["attention"].get("decile_profile")]
        n_bins = max((len(p) for p in profiles), default=0)
        profile = []
        for i in range(n_bins):
            vals = [p[i] for p in profiles if i < len(p)]
            profile.append(float(np.mean(vals)) if vals else 0.0)
        return {
            "mean_mass": float(np.mean(masses)) if masses else 0.0,
            "mean_entropy": float(np.mean(ents)) if ents else 0.0,
            "mean_drift": float(np.mean(drifts)) if drifts else 0.0,
            "decile_profile": profile,
            "n": len(masses),
        }

    return {
        "pope": {s: agg(rs) for s, rs in pope_records.items()},
        "chair": agg(chair_records),
    }


def mention_level_table(chair_records: list[dict]) -> list[dict]:
    rows = []
    for r in chair_records:
        for m in r["mentions"]:
            rows.append(
                {
                    "image": r["image"],
                    "class": m["class"],
                    "grounded": m["grounded"],
                    "attention": m["attention"],
                }
            )
    return rows


def point_biserial(attention: list[float], grounded: list[bool], iters: int = 2000, seed: int = 42) -> dict:
    x = np.asarray(attention, dtype=float)
    y = np.asarray([1.0 if g else 0.0 for g in grounded])
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": len(x)}
    r, p = stats.pointbiserialr(x, y)
    rng = random.Random(seed)
    boots = []
    idx = list(range(len(x)))
    for _ in range(iters):
        sample = rng.choices(idx, k=len(idx))
        boots.append(stats.pointbiserialr(x[sample], y[sample])[0])
    boots = np.sort(boots)
    return {
        "r": float(r),
        "p": float(p),
        "ci_low": float(boots[int(0.025 * iters)]),
        "ci_high": float(boots[int(0.975 * iters)]),
        "n": len(x),
    }


def binned_curve(attention: list[float], grounded: list[bool], n_bins: int = 10) -> list[dict]:
    pairs = sorted(zip(attention, grounded), key=lambda p: p[0])
    n = len(pairs)
    if n < n_bins:
        return []
    bins = []
    for i in range(n_bins):
        lo = i * n // n_bins
        hi = (i + 1) * n // n_bins
        chunk = pairs[lo:hi]
        attn = sum(p[0] for p in chunk) / len(chunk)
        hallu = sum(0 if p[1] else 1 for p in chunk) / len(chunk)
        bins.append({"attention": attn, "hallucination_rate": hallu, "n": len(chunk)})
    return bins


def logistic_slope(attention: list[float], grounded: list[bool]) -> dict:
    x = np.asarray(attention, dtype=float)
    y = np.asarray([1.0 if g else 0.0 for g in grounded])
    if len(np.unique(y)) < 2 or len(x) < 8:
        return {"slope": 0.0, "n": len(x)}
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "n": len(x)}


def kl_summary(per_sample: list[float]) -> dict:
    if not per_sample:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(np.mean(per_sample)), "std": float(np.std(per_sample)), "n": len(per_sample)}