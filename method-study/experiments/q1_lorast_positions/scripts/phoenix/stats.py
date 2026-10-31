"""Paired, image-clustered bootstrap for comparing two configurations.

Every configuration in this project is scored on the *same* questions and the same
images, so differences should be tested paired: an unpaired +-1.96*SE band treats the
two runs as independent samples and is several times too wide. Questions are also
not independent -- POPE asks 6 questions per image and one bad image moves all 6 --
so we resample *images* (clusters), not questions.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Sequence


def pope_f1(preds: Sequence[str | None], labels: Sequence[str]) -> float:
    tp = fp = fn = 0
    for p, l in zip(preds, labels):
        p = p or "no"
        if p == "yes" and l == "yes":
            tp += 1
        elif p == "yes":
            fp += 1
        elif l == "yes":
            fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def pope_acc(preds, labels) -> float:
    return sum((p or "no") == l for p, l in zip(preds, labels)) / max(len(labels), 1)


def pope_stat(preds, labels, which: str) -> float:
    """f1 | acc | precision | recall | yes (unparseable answers count as 'no')."""
    if which == "f1":
        return pope_f1(preds, labels)
    if which == "acc":
        return pope_acc(preds, labels)
    p = [(x or "no") for x in preds]
    if which == "yes":
        return sum(x == "yes" for x in p) / max(len(p), 1)
    tp = sum(x == "yes" and l == "yes" for x, l in zip(p, labels))
    if which == "precision":
        return tp / max(sum(x == "yes" for x in p), 1)
    if which == "recall":
        return tp / max(sum(l == "yes" for l in labels), 1)
    raise KeyError(which)


def auroc(scores: Sequence[float], labels: Sequence[str]) -> float:
    """P(score of a random 'yes' item > score of a random 'no' item), ties = 1/2.

    Threshold-free: a model that only says "yes" more often keeps its AUROC; a model
    that separates present from absent objects less well loses it."""
    pos = [s for s, l in zip(scores, labels) if l == "yes"]
    neg = [s for s, l in zip(scores, labels) if l != "yes"]
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i, n = 0.0, 0, len(allv)
    while i < n:                                      # average ranks over ties
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2
        rank_sum += r * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def compare_auroc(scores_a, scores_b, labels, image_ids, n_boot: int = 1000,
                  seed: int = 0) -> dict:
    def m(sc):
        return lambda idx: auroc([sc[i] for i in idx], [labels[i] for i in idx])
    return paired_bootstrap(m(scores_a), m(scores_b), image_ids, n_boot, seed)


def _clusters(image_ids: Sequence[int]) -> dict[int, list[int]]:
    c: dict[int, list[int]] = defaultdict(list)
    for i, iid in enumerate(image_ids):
        c[iid].append(i)
    return c


def paired_bootstrap(metric: Callable[[list[int]], float],
                     base_metric: Callable[[list[int]], float],
                     image_ids: Sequence[int], n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """CI for metric(A) - metric(B) over the same items, resampling images.

    `metric(idx)` / `base_metric(idx)` evaluate each configuration on the item
    indices `idx` (with repeats).
    """
    clusters = list(_clusters(image_ids).values())
    rng = random.Random(seed)
    all_idx = [i for c in clusters for i in c]
    delta = metric(all_idx) - base_metric(all_idx)
    samples = []
    for _ in range(n_boot):
        idx = [i for _ in range(len(clusters)) for i in rng.choice(clusters)]
        samples.append(metric(idx) - base_metric(idx))
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot) - 1]
    # two-sided p-value for "no difference"
    p = 2 * min(sum(s <= 0 for s in samples), sum(s >= 0 for s in samples)) / n_boot
    return {"delta": delta, "lo": lo, "hi": hi, "p": min(1.0, p),
            "significant": not (lo <= 0.0 <= hi), "n_clusters": len(clusters)}


def compare_pope(preds_a, preds_b, labels, image_ids, metric: str = "f1",
                 n_boot: int = 2000, seed: int = 0) -> dict:
    def m(p):
        return lambda idx: pope_stat([p[i] for i in idx], [labels[i] for i in idx], metric)
    return paired_bootstrap(m(preds_a), m(preds_b), image_ids, n_boot, seed)


def compare_chair(per_a: Sequence[tuple[int, int]], per_b: Sequence[tuple[int, int]],
                  image_ids: Sequence[int], which: str = "CHAIR_i",
                  n_boot: int = 2000, seed: int = 0) -> dict:
    """per_x[k] = (objects mentioned, objects hallucinated) for caption k."""
    def m(per):
        if which == "CHAIR_s":
            return lambda idx: sum(per[i][1] > 0 for i in idx) / max(len(idx), 1)
        return lambda idx: (sum(per[i][1] for i in idx)
                            / max(sum(per[i][0] for i in idx), 1))
    return paired_bootstrap(m(per_a), m(per_b), image_ids, n_boot, seed)


def compare_ratio(per_a: Sequence[tuple[float, float]], per_b: Sequence[tuple[float, float]],
                  image_ids: Sequence[int], n_boot: int = 2000, seed: int = 0) -> dict:
    """sum(num)/sum(den) per configuration, e.g. coverage = objects hit / objects present."""
    def m(per):
        return lambda idx: sum(per[i][0] for i in idx) / max(sum(per[i][1] for i in idx), 1e-9)
    return paired_bootstrap(m(per_a), m(per_b), image_ids, n_boot, seed)


def compare_mean(xs_a: Sequence[float], xs_b: Sequence[float], image_ids: Sequence[int],
                 n_boot: int = 2000, seed: int = 0) -> dict:
    def m(xs):
        return lambda idx: sum(xs[i] for i in idx) / max(len(idx), 1)
    return paired_bootstrap(m(xs_a), m(xs_b), image_ids, n_boot, seed)


def fmt_delta(r: dict, pct: bool = False) -> str:
    f = (lambda x: f"{100*x:+.1f}") if pct else (lambda x: f"{x:+.3f}")
    star = "*" if r["significant"] else " "
    return f"{f(r['delta'])} [{f(r['lo'])},{f(r['hi'])}]{star}"
