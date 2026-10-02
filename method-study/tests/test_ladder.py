#!/usr/bin/env python
"""Offline tests for scripts/00_precision_ladder.py and phoenix/stats.py.

1. The whole ladder script runs end to end on a synthetic COCO with a tiny model
   standing in for LLaVA (no GPU, no download), including --resume and the report.
2. The verdict logic is checked on hand-built rows whose right answer is known:
   a collapsed rung must be BROKEN, a significantly worse but fluent rung must be
   'degraded', and a statistically indistinguishable rung must be 'ok'.
3. The paired bootstrap is checked for calibration and power.

    python tests/test_ladder.py
"""
from __future__ import annotations

import importlib.util
import io
import json
import random
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import torch
from PIL import Image

from phoenix.chair import COCO_80
from phoenix.stats import compare_chair, compare_pope

OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, cond, detail=""):
    print(f"[{OK if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def load_ladder_module():
    spec = importlib.util.spec_from_file_location("ladder", ROOT / "scripts" / "00_precision_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_coco(root: Path, n_images: int = 12):
    rng = random.Random(0)
    cats = [{"id": i + 1, "name": n, "supercategory": "x"} for i, n in enumerate(COCO_80)]
    ids = [1000 + 7 * k for k in range(n_images)]
    cal = [5000 + 7 * k for k in range(6)]               # calibration pool (PTQ rungs)
    imgs, anns, caps, gt = [], [], [], {}
    for iid in ids + cal:
        imgs.append({"id": iid, "file_name": f"COCO_val2014_{iid:012d}.jpg"})
        objs = rng.sample(range(1, 81), 3)
        gt[iid] = [COCO_80[o - 1] for o in objs]
        for o in objs:
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": o,
                         "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0})
        caps.append({"id": len(caps) + 1, "image_id": iid, "caption": "a photo of a dog"})
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "instances_val2014.json").write_text(json.dumps(
        {"images": imgs, "annotations": anns, "categories": cats}))
    (root / "annotations" / "captions_val2014.json").write_text(json.dumps(
        {"images": imgs, "annotations": caps}))
    (root / "val2014").mkdir()
    for iid in ids + cal:
        Image.new("RGB", (32, 32), (iid % 255, 40, 90)).save(
            root / "val2014" / f"COCO_val2014_{iid:012d}.jpg")
    (root / "pope").mkdir()
    for split in ("random", "popular", "adversarial"):
        rows = []
        for iid in ids:
            for o in gt[iid]:
                rows.append({"image": f"COCO_val2014_{iid:012d}.jpg",
                             "text": f"Is there a {o} in the image?", "label": "yes"})
            for o in [c for c in COCO_80 if c not in gt[iid]][:3]:
                rows.append({"image": f"COCO_val2014_{iid:012d}.jpg",
                             "text": f"Is there a {o} in the image?", "label": "no"})
        (root / "pope" / f"coco_pope_{split}.json").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
    (root / "splits.json").write_text(json.dumps(
        {"subset": "val2014", "eval_ids": ids, "calib_ids": cal, "meta": {}}))
    return ids


def fake_loader():
    from test_smoke import StubProcFull, tiny_model
    calls = {"n": 0}

    def load_model(model_id, dtype=None, device="cpu", **kw):
        calls["n"] += 1
        torch.manual_seed(0)                      # same weights for every rung
        lm = tiny_model()
        lm.processor = StubProcFull()
        return lm
    return load_model, calls


# =========================================================================== #
def test_end_to_end():
    print("\n== ladder script end to end (synthetic COCO, tiny model) ==")
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / "coco"
        build_coco(root)
        mod = load_ladder_module()
        mod.load_model, calls = fake_loader()
        out = tmp / "ladder"
        argv = ["00_precision_ladder.py", "--coco-root", str(root), "--out", str(out),
                "--ladder", "fp16,w8a8,w2a4:clip=0.99,w4a16:method=gptq:ncal=4,w4a4:method=rot,"
                "w3a16:method=noise,w3a16:wtok=image",
                "--n-pope-images", "6",
                "--n-chair-images", "4", "--max-new-tokens", "6", "--device", "cpu",
                "--dtype", "float32", "--n-boot", "200"]
        sys.argv = argv
        buf = io.StringIO()
        with redirect_stdout(buf):
            mod.main()
        log = buf.getvalue()
        data = json.loads((out / "ladder.json").read_text())
        check("all rungs run and are saved", list(data["table"]) == [
            "fp16", "w8a8", "w2a4:clip=0.99", "w4a16:method=gptq:ncal=4", "w4a4:method=rot",
            "w3a16:method=noise", "w3a16:wtok=image"]
              and data["complete"], f"{list(data['table'])}")
        check("P(yes) saved per question and AUROC compared",
              len(data["table"]["w8a8"]["pope"]["random"]["p_yes"]) == 36
              and "pope_auroc" in json.loads((out / "ladder_report.json").read_text())
              ["summary"]["w8a8"]["vs_reference"])
        spec = importlib.util.spec_from_file_location("obj", ROOT / "scripts" / "10_object_analysis.py")
        obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        sys.argv = ["10_object_analysis.py", "--coco-root", str(root), "--ladder",
                    str(out / "ladder.json"), "--n-boot", "50"]
        with redirect_stdout(io.StringIO()) as b:
            obj.main()
        oa = json.loads((out / "object_analysis.json").read_text())
        check("object analysis runs on a ladder's captions",
              set(oa["rungs"]) == set(data["table"]) - {"fp16"}, b.getvalue()[-120:])
        g = data["table"]["w4a16:method=gptq:ncal=4"]
        check("a PTQ rung records its calibration", g["ptq"].get("method") == "gptq"
              and g["ptq"]["gptq"]["mean_loss_ratio_vs_rtn"] <= 1.0001, f"{g['ptq']}")
        check("blind (no-image) perplexity and coverage are recorded per rung",
              "ppl" in g["blind"] and len(g["per_caption_cov"]) == 4)
        check("report classifies a direction per rung",
              all("direction" in v for v in json.loads((out / "ladder_report.json")
                                                         .read_text())["summary"].values()))
        row = data["table"]["w8a8"]
        check("per-question POPE vectors saved for paired tests",
              len(row["pope"]["adversarial"]["preds"]) == 36
              and len(row["pope"]["adversarial"]["image_ids"]) == 36)
        check("per-caption CHAIR counts saved", len(row["per_caption"]) == 4)
        check("spec options reach the quantizer",
              abs(data["table"]["w2a4:clip=0.99"]["quant"]["a_clip"] - 0.99) < 1e-9)
        rep = json.loads((out / "ladder_report.json").read_text())
        check("report written with a verdict per rung",
              set(rep["summary"]) == set(data["table"]), f"ref={rep['reference']}")
        check("report prints the paired-comparison table", "paired comparison vs 'fp16'" in log)

        n_before = calls["n"]
        sys.argv = ["00_precision_ladder.py", "--coco-root", str(root), "--out", str(out),
                    "--ladder", "fp16,w8a8,w2a4:clip=0.99,w4a16:method=gptq:ncal=4,"
                    "w4a4:method=rot,w3a16:method=noise,w3a16:wtok=image,w4a6",
                    "--n-pope-images", "6",
                    "--n-chair-images", "4", "--max-new-tokens", "6", "--device", "cpu",
                    "--dtype", "float32", "--n-boot", "200", "--resume"]
        with redirect_stdout(io.StringIO()):
            mod.main()
        data2 = json.loads((out / "ladder.json").read_text())
        check("--resume runs only the missing rung", calls["n"] - n_before == 1
              and "w4a6" in data2["table"], f"loaded {calls['n'] - n_before} model(s)")

        sys.argv = argv[:-1] + ["200", "--ladder", "fp16,w4b8"]
        try:
            with redirect_stdout(io.StringIO()):
                mod.main()
            check("a typo'd rung fails before any model loads", False)
        except KeyError:
            check("a typo'd rung fails before any model loads", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _rows(n_img=60, per_img=6, err_ref=0.12, err_b=0.12, n_cap=40, hall_ref=0.2,
          hall_b=0.2, cover=0.75, length=90.0, unparsed=0.0, seed=0):
    """Build a (ref, b) pair of ladder rows with controlled error rates."""
    rng = random.Random(seed)
    ids = [i for i in range(n_img) for _ in range(per_img)]
    labels = ["yes" if k % 2 else "no" for k in range(len(ids))]

    def preds(err):
        out = []
        for l in labels:
            if rng.random() < unparsed:
                out.append(None)
            elif rng.random() < err:
                out.append("no" if l == "yes" else "yes")
            else:
                out.append(l)
        return out

    def row(err, hall, cov, ln, unp):
        pr = preds(err)
        cap_ids = list(range(n_cap))
        per = [(5, sum(rng.random() < hall for _ in range(5))) for _ in cap_ids]
        return {"splits": ["adversarial", "random"],
                "pope": {s: {"preds": pr, "labels": labels, "image_ids": ids}
                         for s in ("adversarial", "random")},
                **{f"pope_{s}_unparsed": unp for s in ("adversarial", "random")},
                "captions": [{"image_id": i, "caption": "x"} for i in cap_ids],
                "per_caption": per, "coverage": cov, "avg_len": ln, "rep4": 0.01}
    return row(err_ref, hall_ref, cover, length, 0.0), row(err_b, hall_b, cover, length, unparsed)


def test_verdicts():
    print("\n== verdict logic on rows with a known answer ==")
    mod = load_ladder_module()
    out = Path(tempfile.mkdtemp())
    try:
        ref, same = _rows(err_ref=0.12, err_b=0.12, seed=1)
        _, worse = _rows(err_ref=0.12, err_b=0.30, hall_b=0.35, seed=2)
        _, broken = _rows(err_ref=0.12, err_b=0.45, cover=0.02, length=20.0,
                          unparsed=0.6, seed=3)
        table = {"fp16": ref, "w4a8": same, "w4a5": worse, "w4a4": broken}
        with redirect_stdout(io.StringIO()):
            mod.report(table, list(table), ["adversarial", "random"], 500, out)
        rep = json.loads((out / "ladder_report.json").read_text())["summary"]
        check("statistically identical rung -> ok", rep["w4a8"]["verdict"].startswith("ok"),
              rep["w4a8"]["verdict"])
        check("worse but fluent rung -> degraded (the fallback regime)",
              rep["w4a5"]["verdict"].startswith("degraded"), rep["w4a5"]["verdict"][:40])
        check("collapsed rung -> BROKEN, not 'low hallucination'",
              rep["w4a4"]["verdict"].startswith("BROKEN"), rep["w4a4"]["verdict"][:70])
        cands = json.loads((out / "ladder_report.json").read_text())["candidates"]
        check("only the degraded rung is proposed as PREC", cands == ["w4a5"], f"{cands}")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _directional(kind, n_img=80, per_img=6, p=0.25, seed=0):
    """A rung that errs only one way: 'fallback' = false yes, 'omission' = misses."""
    rng = random.Random(seed)
    ids = [i for i in range(n_img) for _ in range(per_img)]
    labels = ["yes" if k % 2 else "no" for k in range(len(ids))]
    base = []
    for l in labels:
        base.append(("no" if l == "yes" else "yes") if rng.random() < 0.08 else l)
    out = []
    for b, l in zip(base, labels):
        if kind == "fallback" and l == "no" and rng.random() < p:
            out.append("yes")
        elif kind == "omission" and l == "yes" and rng.random() < p:
            out.append("no")
        else:
            out.append(b)
    def row(pr):
        return {"splits": ["random"], "pope": {"random": {"preds": pr, "labels": labels,
                                                          "image_ids": ids}},
                "pope_random_unparsed": 0.0,
                "captions": [{"image_id": i, "caption": "a b c"} for i in range(20)],
                "per_caption": [(5, 1)] * 20, "per_caption_cov": [(3, 4)] * 20,
                "coverage": 0.75, "avg_len": 3.0, "rep4": 0.0, "blind": {"ppl": 16.0}}
    return row(base), row(out)


def test_direction():
    print("\n== failure-direction classification ==")
    mod = load_ladder_module()
    out = Path(tempfile.mkdtemp())
    try:
        ref, fb = _directional("fallback")
        _, om = _directional("omission")
        _, same = _directional("none")
        with redirect_stdout(io.StringIO()):
            mod.report({"fp16": ref, "fb": fb, "om": om, "same": same},
                       ["fp16", "fb", "om", "same"], ["random"], 500, out)
        rep = json.loads((out / "ladder_report.json").read_text())["summary"]
        check("more false 'yes' -> fallback", rep["fb"]["direction"] == "fallback",
              rep["fb"]["direction"])
        check("more misses -> omission", rep["om"]["direction"] == "omission",
              rep["om"]["direction"])
        check("no change -> none", rep["same"]["direction"] == "none", rep["same"]["direction"])
        check("LM intact is reported as grounding-specific",
              "grounding-specific" in rep["om"]["verdict"], rep["om"]["verdict"])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_bootstrap():
    print("\n== paired bootstrap calibration and power ==")
    false_pos = 0
    for s in range(40):
        a, b = _rows(err_ref=0.15, err_b=0.15, seed=100 + s)
        r = compare_pope(b["pope"]["adversarial"]["preds"], a["pope"]["adversarial"]["preds"],
                         a["pope"]["adversarial"]["labels"], a["pope"]["adversarial"]["image_ids"],
                         "f1", n_boot=400, seed=s)
        false_pos += r["significant"]
    check("~5% false positives when there is no difference", false_pos <= 5,
          f"{false_pos}/40 significant under the null")

    a, b = _rows(err_ref=0.12, err_b=0.20, seed=7)
    r = compare_pope(b["pope"]["adversarial"]["preds"], a["pope"]["adversarial"]["preds"],
                     a["pope"]["adversarial"]["labels"], a["pope"]["adversarial"]["image_ids"],
                     "f1", n_boot=1000)
    check("detects an 8-point accuracy drop on 360 paired questions",
          r["significant"] and r["delta"] < 0, f"dF1={r['delta']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]")

    per_a = [(5, 1)] * 40
    per_b = [(5, 1)] * 20 + [(5, 3)] * 20
    rc = compare_chair(per_b, per_a, list(range(40)), "CHAIR_i", n_boot=500)
    check("CHAIR_i difference and CI are computed on mention counts",
          abs(rc["delta"] - 0.2) < 1e-9 and rc["significant"], f"dCHAIR_i={rc['delta']:+.3f}")


if __name__ == "__main__":
    test_end_to_end()
    test_direction()
    test_verdicts()
    test_bootstrap()
    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        sys.exit(1)
    print("all ladder tests passed")
