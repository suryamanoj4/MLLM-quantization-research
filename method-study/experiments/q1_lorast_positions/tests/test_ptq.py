#!/usr/bin/env python
"""Tests for the published-PTQ harness (phoenix/ptq) on a tiny LLaVA.

What has to hold for the study to mean anything:
  * the transforms (SmoothQuant scaling, rotation, AWQ scaling) are exact in FP
  * AWQ and GPTQ can only improve on RTN on their own objective
  * rotation / SmoothQuant actually remove the damage an outlier channel does at A4
  * GPTQ / AWQ weights reach the forward pass (not silently re-quantized by RTN)
  * NF4 lands on the bitsandbytes code book
  * text calibration really has no image tokens and matches the mm token budget

    python tests/test_ptq.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

from test_fixes import IMG, Proc
from phoenix.model import LoadedModel
from phoenix.ptq import apply_ptq
from phoenix.ptq import calib as calib_mod
from phoenix.ptq.gptq import gptq_quantize
from phoenix.ptq.transforms import random_rotation
from phoenix.quant import (NF4_CODE, FakeQuantLinear, QuantConfig, iter_fq, quant_mode,
                           quantize_model, quantize_weight, quantize_weight_nf4)

OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, cond, detail=""):
    print(f"[{OK if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def tiny(seed=0, outlier=True):
    torch.manual_seed(seed)
    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, image_size=28, patch_size=14, projection_dim=32)
    tc = LlamaConfig(hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                     num_attention_heads=4, num_key_value_heads=4, vocab_size=64,
                     max_position_embeddings=256)
    cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=IMG,
                      vision_feature_layer=-2, vision_feature_select_strategy="default")
    m = LlavaForConditionalGeneration(cfg).eval().requires_grad_(False)
    lm = LoadedModel(model=m, processor=Proc(), image_token_id=IMG,
                     device=torch.device("cpu"), dtype=torch.float32)
    if outlier:   # the LLM pathology every A4 method exists to fix: a few huge channels
        for layer in lm.layers:
            layer.input_layernorm.weight[[3, 17]] = 40.0
            layer.post_attention_layernorm.weight[[5]] = 30.0
    return lm


def seqs(lm, n=6, seed=0):
    out = []
    for k in range(n):
        img = Image.fromarray(np.random.RandomState(seed + k).randint(0, 255, (40, 40, 3),
                                                                     dtype=np.uint8))
        out.append(lm.processor(images=[img], text=[f"calib {seed} {k}"]))
    return out


def logits(lm, batch):
    return lm.model(**batch, use_cache=False).logits.float()


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def quantized_error(spec, base_logits, test, **kw):
    lm = tiny()
    cfg = QuantConfig.from_ladder(spec, group_size=32, **kw)
    info = apply_ptq(lm, cfg, seqs=seqs(lm))
    quantize_model(lm.model, cfg, verbose=False)
    return rel(logits(lm, test), base_logits), lm, info


# =========================================================================== #
def test_specs():
    print("\n== spec parsing ==")
    c = QuantConfig.from_ladder("w3a16:method=awq:calib=text:ncal=32")
    check("method / calib / ncal parse", (c.w_bits, c.method, c.calib, c.n_calib)
          == (3, "awq", "text", 32))
    c = QuantConfig.from_ladder("nf4", group_size=128)
    check("'nf4' = W4A16, NF4 format, bnb block size 64 by default",
          (c.w_bits, c.a_bits, c.w_format, c.group_size) == (4, 16, "nf4", 64))
    c = QuantConfig.from_ladder("w4a4:method=rot+gptq:actorder=1")
    check("rot+gptq with act-order", c.method == "rot+gptq" and c.gptq_actorder)
    bad = 0
    for s in ("w4a16:method=foo", "w4a16:calib=bar", "w3a16:wfmt=nf4",
              "nf4:method=gptq", "w16a16:method=gptq", "w4a8:backend=bnb"):
        try:
            QuantConfig.from_ladder(s)
        except KeyError:
            bad += 1
    check("invalid combinations are refused before any model loads", bad == 6, f"{bad}/6")
    check("rtn fp16 is still the identity; a method is not",
          QuantConfig.from_ladder("fp16").is_identity
          and not QuantConfig.from_ladder("w16a16:method=rot").is_identity)


def test_nf4():
    print("\n== NF4 ==")
    torch.manual_seed(0)
    w = torch.randn(16, 128)
    q = quantize_weight_nf4(w, 64)
    blocks = w.reshape(-1, 64)
    am = blocks.abs().amax(1, keepdim=True)
    codes = torch.tensor(NF4_CODE)
    normed = (q.reshape(-1, 64) / am)
    on_code = (normed[..., None] - codes).abs().amin(-1).max()
    check("every value is (code book entry) x (block absmax)", float(on_code) < 1e-6)
    check("idempotent", torch.allclose(quantize_weight_nf4(q, 64), q, atol=1e-6))
    nearest = codes[(blocks / am)[..., None].sub(codes).abs().argmin(-1)]
    check("nearest-code rounding", torch.allclose(normed, nearest, atol=1e-6))


def test_transforms_exact():
    print("\n== transforms preserve the FP function ==")
    test = seqs(tiny(), 2, seed=100)[0]
    base = logits(tiny(), test)
    for method in ("sq", "rot"):
        lm = tiny()
        cfg = QuantConfig.from_ladder(f"w16a16:method={method}")
        apply_ptq(lm, cfg, seqs=seqs(lm))
        check(f"{method}: FP logits unchanged", rel(logits(lm, test), base) < 1e-4,
              f"rel err {rel(logits(lm, test), base):.2e}")
    for n in (64, 96, 11008 // 172):
        r = random_rotation(n, 0, "cpu")
        check(f"rotation {n}x{n} is orthogonal",
              torch.allclose(r @ r.t(), torch.eye(n), atol=1e-4))
    # AWQ: quantize, then switch quant off -> must be the original model
    lm = tiny()
    cfg = QuantConfig.from_ladder("w3a16:method=awq", group_size=32)
    apply_ptq(lm, cfg, seqs=seqs(lm))
    quantize_model(lm.model, cfg, verbose=False)
    with quant_mode(lm.model, False):
        e = rel(logits(lm, test), base)
    check("awq: scales folded exactly (quant off == original)", e < 1e-4, f"{e:.2e}")
    check("awq: every wrapped Linear carries the searched weight",
          all(m.prequantized for m in iter_fq(lm.model)))


def test_gptq_math():
    print("\n== GPTQ on one Linear ==")
    torch.manual_seed(0)
    X = torch.randn(4096, 128) @ (torch.randn(128, 128) * 0.3 + torch.eye(128))
    X[:, 7] *= 20
    W = torch.randn(64, 128)
    H = X.t() @ X / X.shape[0]
    R = quantize_weight(W, 3, 32, False)
    err = lambda Q: float(((X @ (W - Q).t()) ** 2).mean())
    for actorder in (False, True):
        Q = gptq_quantize(W, H, 3, 32, False, actorder)
        check(f"output error below RTN's (actorder={actorder})", err(Q) < 0.8 * err(R),
              f"gptq {err(Q):.3f} vs rtn {err(R):.3f}")
        levels = max(int(torch.unique(Q[r, g:g + 32]).numel())
                     for r in range(64) for g in range(0, 128, 32))
        check(f"at most 2^3 levels per (row, group) (actorder={actorder})", levels <= 8,
              f"max {levels}")


def test_end_to_end():
    print("\n== end to end on the tiny LLaVA (outlier channels injected) ==")
    test = seqs(tiny(), 2, seed=100)[0]
    base = logits(tiny(), test)
    res = {}
    for spec in ("w3a16", "w3a16:method=awq", "w3a16:method=gptq",
                 "w8a4", "w8a4:method=sq", "w16a4", "w16a4:method=rot",
                 "w4a4", "w4a4:method=rot", "w4a4:method=rot+gptq"):
        e, lm, info = quantized_error(spec, base, test)
        res[spec] = e
        for k in ("awq", "gptq"):
            if k in info:
                check(f"{spec}: {k} objective <= RTN's", info[k]["mean_loss_ratio_vs_rtn"] <= 1.0001,
                      f"ratio {info[k]['mean_loss_ratio_vs_rtn']:.3f}")
    print("   logit rel. error:", {k: round(v, 3) for k, v in res.items()})
    check("gptq beats rtn at W3", res["w3a16:method=gptq"] < res["w3a16"])
    check("awq does not lose to rtn at W3", res["w3a16:method=awq"] <= res["w3a16"] * 1.05)
    # a 64-wide toy can only spread an outlier by sqrt(64) = 8x, so expect modest gains
    # here; the real check is scripts/09_validate_ptq.py on the 7B model
    check("SmoothQuant cuts the A4 outlier damage", res["w8a4:method=sq"] < 0.85 * res["w8a4"])
    check("rotation cuts the A4 outlier damage", res["w16a4:method=rot"] < 0.85 * res["w16a4"])
    check("rotation does not hurt at W4A4", res["w4a4:method=rot"] <= res["w4a4"] * 1.02)
    check("rot+gptq <= rot", res["w4a4:method=rot+gptq"] <= res["w4a4:method=rot"] * 1.05)

    # FP mode of a rotated + GPTQ model is still the original
    lm = tiny()
    cfg = QuantConfig.from_ladder("w4a4:method=rot+gptq", group_size=32)
    apply_ptq(lm, cfg, seqs=seqs(lm))
    quantize_model(lm.model, cfg, verbose=False)
    with quant_mode(lm.model, False):
        e = rel(logits(lm, test), base)
    check("rot+gptq: quant off == original model", e < 1e-4, f"{e:.2e}")
    fq = list(iter_fq(lm.model))
    check("rotation reaches FakeQuantLinear", all(m.in_rot is not None for m in fq))
    n_mat = len({m.in_rot.data_ptr() for m in fq})
    check("one shared rotation matrix per width", n_mat == 2, f"{n_mat} distinct")


def _fake_coco(root: Path, n=24):
    from phoenix.fetch import write_splits
    (root / "annotations").mkdir(parents=True)
    (root / "val2014").mkdir()
    ids = [100 + i for i in range(n)]
    images = [{"id": i, "file_name": f"COCO_val2014_{i:012d}.jpg"} for i in ids]
    cats = [{"id": 1, "name": "dog"}, {"id": 2, "name": "car"}]
    anns = [{"id": k, "image_id": i, "category_id": 1 + k % 2} for k, i in enumerate(ids)]
    caps = [{"id": 10 * k + j, "image_id": i, "caption": f"a dog near a car number {k} {j}"}
            for k, i in enumerate(ids) for j in range(5)]
    (root / "annotations" / "instances_val2014.json").write_text(json.dumps(
        {"images": images, "annotations": anns, "categories": cats}))
    (root / "annotations" / "captions_val2014.json").write_text(json.dumps(
        {"images": images, "annotations": caps}))
    rs = np.random.RandomState(0)
    for i in ids:
        Image.fromarray(rs.randint(0, 255, (40, 40, 3), dtype=np.uint8)).save(
            root / "val2014" / f"COCO_val2014_{i:012d}.jpg")
    write_splits(root, ids[: n // 2], ids[n // 2:], "val2014")
    return ids[: n // 2], ids[n // 2:]


def test_calibration_data():
    print("\n== calibration sources ==")
    tmp = Path(tempfile.mkdtemp(prefix="phoenix-ptq-"))
    try:
        eval_ids, calib_ids = _fake_coco(tmp / "coco")
        lm = tiny()
        mm = calib_mod.build(lm, tmp / "coco", "mm", 4)
        tx = calib_mod.build(lm, tmp / "coco", "text", 4)
        check("mm sequences contain image tokens",
              all(bool((s["input_ids"] == IMG).any()) for s in mm))
        check("text sequences contain none",
              not any(bool((s["input_ids"] == IMG).any()) for s in tx))
        L = lambda ss: [int(s["input_ids"].shape[1]) for s in ss]
        check("text sequences match the mm token budget",
              abs(sum(L(tx)) - sum(L(mm))) <= len(mm), f"{L(tx)} vs {L(mm)}")
        from phoenix.data import build_calibration_set
        used = {s.image_id for s in build_calibration_set(tmp / "coco", "train2014", 4,
                                                          seed=11, fallback_subset="val2014")}
        check("calibration images come from the calib pool only",
              used <= set(calib_ids) and not used & set(eval_ids))
        cfg = QuantConfig.from_ladder("w3a16:method=gptq:calib=text:ncal=4", group_size=32)
        info = apply_ptq(lm, cfg, coco_root=tmp / "coco")
        check("gptq runs end to end from text calibration", info.get("calib") == "text")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_controls():
    print("\n== Gate 2 controls: matched noise, weight-path attribution, AUROC ==")
    from phoenix.loras import steer_mask
    from phoenix.quant import matched_noise
    from phoenix.stats import auroc
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    cfg = QuantConfig.from_ladder("w3a16:method=noise", group_size=64)
    n = matched_noise(w, cfg, "x")
    r = quantize_weight(w, 3, 64, False)
    e_n, e_r = float((n - w).pow(2).sum()), float((r - w).pow(2).sum())
    check("noise carries RTN's error energy", abs(e_n / e_r - 1) < 0.1, f"ratio {e_n / e_r:.3f}")
    corr = float(torch.nn.functional.cosine_similarity((n - w).flatten(), (r - w).flatten(), 0))
    check("...in a direction unrelated to RTN's error", abs(corr) < 0.1, f"cos {corr:+.3f}")
    check("noise is reproducible", torch.equal(n, matched_noise(w, cfg, "x"))
          and not torch.equal(n, matched_noise(w, cfg, "y")))

    test = seqs(tiny(), 1, seed=100)[0]
    mask = test["input_ids"] == IMG
    out = {}
    for spec in ("fp16", "w2a16", "w2a16:wtok=image", "w2a16:wtok=text"):
        lm = tiny()
        c = QuantConfig.from_ladder(spec, group_size=32)
        quantize_model(lm.model, c, verbose=False)
        from phoenix.quant import release_fp_weights
        release_fp_weights(lm.model)               # must not break the wtok rungs
        with steer_mask(mask):
            out[spec] = logits(lm, test)
    full = rel(out["w2a16"], out["fp16"])
    img, txt = rel(out["w2a16:wtok=image"], out["fp16"]), rel(out["w2a16:wtok=text"], out["fp16"])
    check("wtok=image and wtok=text each perturb less than quantizing everywhere",
          0 < img < full and 0 < txt < full, f"image {img:.3f} text {txt:.3f} all {full:.3f}")
    pre = int(mask[0].nonzero()[0])
    early = rel(out["w2a16:wtok=image"][:, :pre], out["fp16"][:, :pre])
    check("wtok=image leaves positions before the image exactly FP16", early < 1e-6, f"{early:.1e}")

    check("AUROC: perfect, chance, inverted",
          auroc([0.9, 0.8, 0.1, 0.2], ["yes", "yes", "no", "no"]) == 1.0
          and auroc([0.5] * 4, ["yes", "no", "yes", "no"]) == 0.5
          and auroc([0.1, 0.9], ["yes", "no"]) == 0.0)
    rng = np.random.RandomState(0)
    lab = ["yes" if k % 2 else "no" for k in range(400)]
    sc = [float(rng.rand() + (0.3 if l == "yes" else 0)) for l in lab]
    shifted = [x ** 0.3 for x in sc]          # monotone: more yes, same ranking
    check("AUROC ignores a pure yes-bias shift", abs(auroc(shifted, lab) - auroc(sc, lab)) < 1e-12)


def main():
    test_specs()
    test_nf4()
    test_transforms_exact()
    test_gptq_math()
    test_end_to_end()
    test_calibration_data()
    test_controls()
    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        sys.exit(1)
    print("all ptq tests passed")


if __name__ == "__main__":
    main()
