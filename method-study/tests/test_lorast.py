#!/usr/bin/env python
"""CPU tests for LoRAS-T (text-side residual correctors) on a tiny random Llama.

    python tests/test_lorast.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

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


@torch.no_grad()
def final_err(lm, batches, steer: bool):
    """Relative error of the last hidden state at text positions, and logit KL."""
    num = den = kl = n = 0.0
    for b in batches:
        txt = b["input_ids"] != IMAGE_TOKEN_ID
        img = ~txt
        with steer_mask(img), steer_enabled(False), quant_mode(lm.model, False):
            o = lm.model(**b, use_cache=False, output_hidden_states=True)
        hf, lf = o.hidden_states[-1][txt], o.logits[txt].log_softmax(-1)
        with steer_mask(img), steer_enabled(steer), quant_mode(lm.model, True):
            o = lm.model(**b, use_cache=False, output_hidden_states=True)
        hq, lq = o.hidden_states[-1][txt], o.logits[txt].log_softmax(-1)
        num += float((hq - hf).pow(2).sum()); den += float(hf.pow(2).sum())
        kl += float(F.kl_div(lq, lf, log_target=True, reduction="sum")); n += lf.shape[0]
    return num / den, kl / n


def test_calibrate():
    print("\n== LoRAS-T calibration ==")
    lm = tiny_model(n_layers=4, d=64)
    quantize_model(lm.model, QuantConfig(w_bits=2, a_bits=16, group_size=16), verbose=False)
    train = [fake_batch(B=1, seed=s) for s in range(60)]
    held = [fake_batch(B=1, seed=1000 + s) for s in range(10)]
    e0, kl0 = final_err(lm, held, steer=False)
    corr, diag = LT.calibrate(lm, train, list(range(4)), rank=16, ridge=1e-3,
                              layers_per_pass=1, val_every=5, verbose=False)
    e1, kl1 = final_err(lm, held, steer=True)
    check("one corrector per layer", sorted(corr) == [0, 1, 2, 3])
    check("held-out final-layer error at text positions drops", e1 < e0,
          f"{e0:.4f} -> {e1:.4f}")
    check("held-out next-token KL to FP drops", kl1 < kl0, f"{kl0:.4f} -> {kl1:.4f}")
    vals = [d["val_red"] for d in diag.values()]
    check("validation reduction positive at every layer", all(v > 0 for v in vals),
          f"{min(vals):.3f}..{max(vals):.3f}")

    # steer disabled -> exactly the plain quantized model
    b = held[0]
    img = b["input_ids"] == IMAGE_TOKEN_ID
    with torch.no_grad(), quant_mode(lm.model, True), steer_mask(img):
        with steer_enabled(False):
            a = lm.model(**b, use_cache=False).logits
        LT.detach_hidden(lm.layers)
        c = lm.model(**b, use_cache=False).logits
        LT.attach_hidden(lm.layers, corr)
    check("steer_enabled(False) == no correctors", torch.equal(a, c))

    # image positions are left alone in prefill (first corrected layer)
    cap = {}
    h = lm.layers[0].register_forward_hook(
        lambda m, i, o: cap.__setitem__(len(cap), (o[0] if isinstance(o, tuple) else o).clone()))
    with torch.no_grad(), quant_mode(lm.model, True), steer_mask(img):
        with steer_enabled(False):
            lm.model(**b, use_cache=False)
        with steer_enabled(True):
            lm.model(**b, use_cache=False)
    h.remove()
    check("prefill: image positions untouched, text positions corrected",
          torch.equal(cap[0][img], cap[1][img]) and not torch.equal(cap[0][~img], cap[1][~img]))

    # decode steps are corrected too (mask shape does not match a 1-token step)
    with torch.no_grad(), quant_mode(lm.model, True):
        with steer_mask(img), steer_enabled(True):
            out = lm.model(**b, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1)[:, None]
        am = torch.cat([b["attention_mask"], torch.ones(1, 1, dtype=torch.long)], 1)
        steps = {}
        for on in (False, True):
            with steer_enabled(on):
                pkv = out.past_key_values
                if hasattr(pkv, "crop"):
                    pkv.crop(b["input_ids"].shape[1])
                steps[on] = lm.model(input_ids=nxt, attention_mask=am,
                                     past_key_values=pkv, use_cache=True).logits
    check("decode steps are corrected", not torch.allclose(steps[False], steps[True]))

    # massive-norm tokens bypass the corrector
    c0 = corr[0]
    hbig = torch.randn(1, 3, 64) * 1e4
    with steer_enabled(True):
        check("norm gate leaves massive activations alone", torch.equal(c0(hbig, None), hbig))

    with tempfile.TemporaryDirectory() as d:
        LT.save(Path(d) / "c.pt", corr, {"precision": "w2a16", "rank": 16})
        c2, meta = LT.load(Path(d) / "c.pt", "cpu", torch.float32)
        check("save/load round trip", meta["precision"] == "w2a16"
              and all(torch.equal(c2[k].A, corr[k].A) and c2[k].max_norm == corr[k].max_norm
                      for k in corr))
    LT.set_scale(lm.layers, 0.0)
    e_off, _ = final_err(lm, held[:3], steer=True)
    e_ref, _ = final_err(lm, held[:3], steer=False)
    check("scale 0 == off", abs(e_off - e_ref) < 1e-9)
    LT.detach_hidden(lm.layers)


def test_rung_parse():
    print("\n== ladder rung syntax ==")
    spec = importlib.util.spec_from_file_location(
        "lad", Path(__file__).resolve().parents[1] / "scripts" / "00_precision_ladder.py")
    lad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lad)
    check("plain rung", lad.split_rung("w3a16:method=gptq") == ("w3a16:method=gptq", None))
    check("lorast rung", lad.split_rung("w3a16+lorast=runs/x.pt") == ("w3a16", ("runs/x.pt", 1.0)))
    check("lorast rung with scale",
          lad.split_rung("w3a16:method=gptq:calib=wiki+lorast=r/y.pt@0.5")
          == ("w3a16:method=gptq:calib=wiki", ("r/y.pt", 0.5)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scripts_end_to_end():
    print("\n== 12_calibrate_lorast -> ladder rung (synthetic COCO, tiny model) ==")
    import io
    import json
    import shutil
    from contextlib import redirect_stdout
    from test_ladder import build_coco, fake_loader
    root_dir = Path(__file__).resolve().parents[1]
    tmp = Path(tempfile.mkdtemp())
    try:
        coco = tmp / "coco"
        build_coco(coco)
        loader, _ = fake_loader()
        cal = _load(root_dir / "scripts" / "12_calibrate_lorast.py", "cal")

        def load_quantized(model_id, spec, coco_root=None, keep_fp=False, **kw):
            lm = loader(model_id)
            cfg = QuantConfig.from_ladder(spec)
            quantize_model(lm.model, cfg, verbose=False)
            return lm, cfg, {"method": cfg.method}
        cal.load_quantized = load_quantized
        pt = tmp / "lt" / "w2.pt"
        sys.argv = ["12", "--coco-root", str(coco), "--precision", "w2a16", "--n-calib", "8",
                    "--max-new-tokens", "6", "--rank", "8", "--device", "cpu",
                    "--dtype", "float32", "--qa-per-image", "2", "--out", str(pt)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            cal.main()
        meta = json.loads(pt.with_suffix(".json").read_text())["meta"]
        check("calibration script writes correctors + diagnostics",
              pt.exists() and meta["precision"] == "w2a16" and len(meta["layers"]) == 4,
              buf.getvalue().strip().splitlines()[-1][:110])
        check("yes/no calibration questions used and checked",
              meta["n_qa_batches"] > 0 and "qa" in meta["kl_before"] and "qa" in meta["kl_after"],
              f"|dP(yes)| {meta['kl_before']['qa']['abs']:.4f} -> {meta['kl_after']['qa']['abs']:.4f}")
        check("held-out KL reported before/after", "kl" in meta["kl_before"]
              and "kl" in meta["kl_after"],
              f"{meta['kl_before']['kl']:.4f} -> {meta['kl_after']['kl']:.4f}")

        lad = _load(root_dir / "scripts" / "00_precision_ladder.py", "lad2")
        lad.load_model, calls = fake_loader()
        out = tmp / "ladder"
        base = ["00", "--coco-root", str(coco), "--out", str(out), "--n-pope-images", "6",
                "--n-chair-images", "4", "--max-new-tokens", "6", "--device", "cpu",
                "--dtype", "float32", "--n-boot", "100"]
        sys.argv = base + ["--ladder", "fp16,w2a16"]
        with redirect_stdout(io.StringIO()):
            lad.main()
        n0 = calls["n"]
        rung = f"w2a16+lorast={pt}"
        sys.argv = base + ["--ladder", "fp16", "--ladder-extra", rung, "--resume"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            lad.main()
        t = json.loads((out / "ladder.json").read_text())["table"]
        check("ladder runs only the LoRAS-T rung on --resume",
              calls["n"] - n0 == 1 and set(t) == {"fp16", "w2a16", rung})
        check("LoRAS-T rung records its correctors", t[rung]["ptq"]["lorast"]["rank"] == 8)
        check("LoRAS-T changes the captions",
              [c["caption"] for c in t[rung]["captions"]]
              != [c["caption"] for c in t["w2a16"]["captions"]])
        sys.argv = base[:-1] + ["100", "--n-chair-images", "5", "--ladder", "fp16",
                                "--ladder-extra", rung, "--resume"]
        try:
            with redirect_stdout(io.StringIO()):
                lad.main()
            check("mismatched --resume with extra rungs refuses to start fresh", False)
        except SystemExit:
            check("mismatched --resume with extra rungs refuses to start fresh", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_calibrate()
    test_rung_parse()
    test_scripts_end_to_end()
    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        sys.exit(1)
    print("all LoRAS-T tests passed")
