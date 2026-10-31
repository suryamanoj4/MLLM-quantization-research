#!/usr/bin/env python
"""CPU tests for the LoRAS-T placement ablation (positions="all") on a tiny random Llama.

    python tests/test_lorast_positions.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from phoenix import loras_text as LT
from phoenix.loras import steer_enabled, steer_mask
from phoenix.quant import QuantConfig, quant_mode, quantize_model
from test_smoke import IMAGE_TOKEN_ID, fake_batch, tiny_model

torch.manual_seed(0)
_failures = []


def check(name, cond, detail=""):
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def fresh():
    torch.manual_seed(0)
    lm = tiny_model(n_layers=4, d=64)
    quantize_model(lm.model, QuantConfig(w_bits=2, a_bits=16, group_size=16), verbose=False)
    return lm


@torch.no_grad()
def layer0_out(lm, b, steer):
    cap = {}
    h = lm.layers[0].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).clone()))
    img = b["input_ids"] == IMAGE_TOKEN_ID
    with quant_mode(lm.model, True), steer_mask(img), steer_enabled(steer):
        lm.model(**b, use_cache=False)
    h.remove()
    return cap["h"]


@torch.no_grad()
def img_err(lm, batches, steer):
    """Relative last-layer error vs FP at IMAGE positions."""
    num = den = 0.0
    for b in batches:
        img = b["input_ids"] == IMAGE_TOKEN_ID
        with steer_mask(img), steer_enabled(False), quant_mode(lm.model, False):
            hf = lm.model(**b, use_cache=False, output_hidden_states=True).hidden_states[-1][img]
        with steer_mask(img), steer_enabled(steer), quant_mode(lm.model, True):
            hq = lm.model(**b, use_cache=False, output_hidden_states=True).hidden_states[-1][img]
        num += float((hq - hf).pow(2).sum()); den += float(hf.pow(2).sum())
    return num / den


def test_scripts_end_to_end():
    """The README workflow: calibrate text + all, then both as ladder rungs (--resume)."""
    print("\n== README workflow: 12_calibrate_lorast x2 -> 00_precision_ladder ==")
    import importlib.util
    import io
    import json
    import shutil
    from contextlib import redirect_stdout
    from test_ladder import build_coco, fake_loader
    from phoenix.quant import QuantConfig as QC

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    root_dir = Path(__file__).resolve().parents[1]
    tmp = Path(tempfile.mkdtemp())
    try:
        coco = tmp / "coco"
        build_coco(coco)
        loader, _ = fake_loader()
        cal = _load(root_dir / "scripts" / "12_calibrate_lorast.py", "cal_pos")

        def load_quantized(model_id, spec, coco_root=None, keep_fp=False, **kw):
            lm = loader(model_id)
            cfg = QC.from_ladder(spec)
            quantize_model(lm.model, cfg, verbose=False)
            return lm, cfg, {"method": cfg.method}
        cal.load_quantized = load_quantized
        pts = {}
        for pos in ("text", "all"):
            pts[pos] = tmp / "lt" / f"w2_{pos}.pt"
            sys.argv = ["12", "--coco-root", str(coco), "--precision", "w2a16", "--n-calib", "8",
                        "--max-new-tokens", "6", "--rank", "8", "--device", "cpu",
                        "--dtype", "float32", "--qa-per-image", "2", "--positions", pos,
                        "--out", str(pts[pos])]
            with redirect_stdout(io.StringIO()):
                cal.main()
        metas = {k: json.loads(v.with_suffix(".json").read_text())["meta"] for k, v in pts.items()}
        check("--positions recorded in each corrector file",
              metas["text"]["positions"] == "text" and metas["all"]["positions"] == "all")
        check("equal parameter count (matched capacity)",
              metas["text"]["params"] == metas["all"]["params"], str(metas["all"]["params"]))

        lad = _load(root_dir / "scripts" / "00_precision_ladder.py", "lad_pos")
        lad.load_model, calls = fake_loader()
        out = tmp / "ladder"
        base = ["00", "--coco-root", str(coco), "--out", str(out), "--n-pope-images", "6",
                "--n-chair-images", "4", "--max-new-tokens", "6", "--device", "cpu",
                "--dtype", "float32", "--n-boot", "100"]
        sys.argv = base + ["--ladder", "fp16,w2a16"]
        with redirect_stdout(io.StringIO()):
            lad.main()
        rungs = [f"w2a16+lorast={pts['text']}", f"w2a16+lorast={pts['all']}"]
        sys.argv = base + ["--ladder", "fp16", "--ladder-extra", ",".join(rungs), "--resume"]
        with redirect_stdout(io.StringIO()):
            lad.main()
        t = json.loads((out / "ladder.json").read_text())["table"]
        check("both LoRAS-T rungs evaluated next to fp16 / w2a16",
              set(t) == {"fp16", "w2a16", *rungs})
        check("text-only and all-positions rungs give different outputs",
              [c["caption"] for c in t[rungs[0]]["captions"]]
              != [c["caption"] for c in t[rungs[1]]["captions"]]
              or t[rungs[0]]["pope"] != t[rungs[1]]["pope"])
        rep = json.loads((out / "ladder_report.json").read_text())
        check("ladder report has paired deltas for both rungs",
              all(r in json.dumps(rep) for r in rungs))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    train = [fake_batch(B=1, seed=s) for s in range(60)]
    held = [fake_batch(B=1, seed=1000 + s) for s in range(10)]
    b = held[0]
    img = b["input_ids"] == IMAGE_TOKEN_ID

    print("\n== text positions (LoRAS-T, unchanged behaviour) ==")
    lm_t = fresh()
    ct, dt = LT.calibrate(lm_t, train, list(range(4)), rank=16, ridge=1e-3, verbose=False)
    check("default positions is 'text'", all(c.positions == "text" for c in ct.values()))
    a, c = layer0_out(lm_t, b, False), layer0_out(lm_t, b, True)
    check("text corrector leaves image positions untouched",
          torch.equal(a[img], c[img]) and not torch.equal(a[~img], c[~img]))
    tok_t = dt[0]["n_tokens"]

    print("\n== all positions (ablation) ==")
    lm_a = fresh()
    e0 = img_err(lm_a, held, steer=False)
    ca, da = LT.calibrate(lm_a, train, list(range(4)), rank=16, ridge=1e-3, verbose=False,
                          positions="all")
    check("every corrector is an all-positions corrector",
          all(c.positions == "all" for c in ca.values()))
    a, c = layer0_out(lm_a, b, False), layer0_out(lm_a, b, True)
    check("all-positions corrector changes image AND text positions",
          not torch.equal(a[img], c[img]) and not torch.equal(a[~img], c[~img]))
    check("fit uses the image tokens too (more tokens than text-only)",
          da[0]["n_tokens"] > tok_t, f"{tok_t} -> {da[0]['n_tokens']}")
    e1 = img_err(lm_a, held, steer=True)
    check("held-out error at image positions drops", e1 < e0, f"{e0:.4f} -> {e1:.4f}")
    check("same parameter count as the text corrector at equal rank",
          sum(x.A.numel() + x.B.numel() + x.bias.numel() for x in ca.values())
          == sum(x.A.numel() + x.B.numel() + x.bias.numel() for x in ct.values()))

    print("\n== lazy FP16 targets == eager ==")
    lm_e = fresh()
    ce, _ = LT.calibrate(lm_e, train[:20], [0, 1], rank=8, ridge=1e-3, verbose=False,
                         positions="all", lazy_fp=False)
    LT.detach_hidden(lm_e.layers)
    lm_l = fresh()
    cl, _ = LT.calibrate(lm_l, train[:20], [0, 1], rank=8, ridge=1e-3, verbose=False,
                         positions="all", lazy_fp=True)
    # eager caches FP16 targets in half precision on the CPU, lazy keeps them exact:
    # equal up to that rounding (~1e-4 relative)
    rel = max(float((ce[k].A @ ce[k].B - cl[k].A @ cl[k].B).norm() / (ce[k].A @ ce[k].B).norm())
              for k in ce)
    check("lazy and eager FP16 targets give the same correctors", rel < 5e-3, f"max rel diff {rel:.1e}")

    print("\n== save / load ==")
    with tempfile.TemporaryDirectory() as d:
        LT.save(Path(d) / "a.pt", ca, {"positions": "all"})
        c2, meta = LT.load(Path(d) / "a.pt", "cpu", torch.float32)
        check("positions survive save/load", all(x.positions == "all" for x in c2.values()))
        # a file written before the ablation existed (no 'positions' key) loads as text
        blob = torch.load(Path(d) / "a.pt", weights_only=False)
        for t in blob["state"].values():
            t.pop("positions")
        torch.save(blob, Path(d) / "old.pt")
        c3, _ = LT.load(Path(d) / "old.pt", "cpu", torch.float32)
        check("old correctors (no positions key) load as text-only",
              all(x.positions == "text" for x in c3.values()))
    try:
        LT.HiddenCorrector(ca[0].A, ca[0].B, ca[0].bias, 1.0, positions="image")
        check("bad positions value is rejected", False)
    except ValueError:
        check("bad positions value is rejected", True)

    test_scripts_end_to_end()

    print(f"\n{'all LoRAS-T position tests passed' if not _failures else f'{len(_failures)} FAILED: {_failures}'}")
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()
