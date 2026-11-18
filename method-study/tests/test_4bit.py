#!/usr/bin/env python
"""CPU tests for the 4-bit / latency additions (branch exp/4bit-latency).

    python tests/test_4bit.py

  * grid=ref (GPTQ reference grid) for RTN and GPTQ; the default grid is unchanged
  * GPTQ Cholesky fallback: damp x10 up to 3 retries, then raise
  * NVFP4 / MXFP4 weight and activation formats against hand-computed values
  * backend=packed: spec validation, nothing wrapped, clear errors for the sim-only
    experiments, LoRAS-T hooks + steer_mask on non-nn.Linear (packed-style) layers
  * the gate4bit presets, Benjamini-Hochberg FDR, and the ladder end to end with
    sim / bnb / packed rows tagged
  * the fused LoRAS-T corrector path matches the default path (atol 1e-3 in fp16)
  * the latency harness: forced-length decode ignores EOS and reproduces greedy
    generate(), TPOT excludes the prefill, weight bytes, summaries, and
    scripts/15_latency.py end to end (tiny LLaVA, CPU clocks) incl. --resume
  * 16_figures_4bit.py and 17_summary_4bit.py on those outputs (+ --publish size rule)
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))   # phoenix/ lives in scripts/
sys.path.insert(0, str(ROOT / "tests"))

import torch

from phoenix.ptq.gptq import damped_inverse_cholesky, gptq_quantize
from phoenix.quant import (FakeQuantLinear, QuantConfig, effective_wbits, fp4_qdq,
                           quantize_weight, round_e2m1, round_e4m3)

OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, cond, detail=""):
    print(f"[{OK if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def _old_rtn(w, bits, gs, sym):
    """The RTN quantizer exactly as it was before the grid option (reference copy)."""
    out_f, in_f = w.shape
    wg = w.reshape(out_f, in_f // gs, gs).float()
    if sym:
        qmax = 2 ** (bits - 1) - 1
        scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        deq = torch.round(wg / scale).clamp(-qmax - 1, qmax) * scale
    else:
        qmax = 2 ** bits - 1
        mx, mn = wg.amax(dim=-1, keepdim=True), wg.amin(dim=-1, keepdim=True)
        scale = ((mx - mn) / qmax).clamp(min=1e-8)
        zero = torch.round(-mn / scale)
        deq = ((torch.round(wg / scale) + zero).clamp(0, qmax) - zero) * scale
    return deq.reshape(out_f, in_f).to(w.dtype)


def test_grid():
    print("\n== grid=ours|ref ==")
    c = QuantConfig.from_ladder("w4a16:method=gptq:calib=wiki:grid=ref")
    check("spec key grid=ref parses", c.w_grid == "ref" and c.method == "gptq")
    check("default grid is ours", QuantConfig.from_ladder("w4a16").w_grid == "ours")
    bad = 0
    for s in ("w4a16:grid=foo", "w4a16:method=awq:grid=ref", "nf4:grid=ref"):
        try:
            QuantConfig.from_ladder(s)
        except KeyError:
            bad += 1
    check("grid=foo / awq+ref / nf4+ref are refused", bad == 3, f"{bad}/3")

    torch.manual_seed(0)
    w = torch.randn(8, 256, dtype=torch.float16)
    same = all(torch.equal(quantize_weight(w, b, gs, sym), _old_rtn(w, b, gs, sym))
               for b in (3, 4) for gs in (32, 128) for sym in (False, True))
    check("grid=ours is bit-identical to the old RTN quantizer", same)

    # hand-computed: one all-positive group [1, 2, 3, 4] at 2 bits
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    ours = quantize_weight(x, 2, 4, False, grid="ours")
    ref = quantize_weight(x, 2, 4, False, grid="ref")
    # ours: [min,max]=[1,4], scale 1, exact.  ref: [0,4], scale 4/3, zero 0,
    # round([0.75, 1.5, 2.25, 3.0]) = [1, 2, 2, 3] (half to even) -> x 4/3
    check("ours on [1,2,3,4] @2b is exact", torch.allclose(ours, x))
    check("ref on [1,2,3,4] @2b = [4/3, 8/3, 8/3, 4]",
          torch.allclose(ref, torch.tensor([[4 / 3, 8 / 3, 8 / 3, 4.0]]), atol=1e-6), str(ref))
    # mixed-sign group: min(0,min)/max(0,max) equals min/max, so the grids agree
    y = torch.tensor([[-1.0, 0.2, 0.9, 2.0]])
    check("ref == ours when the group straddles 0",
          torch.equal(quantize_weight(y, 3, 4, False, grid="ref"),
                      quantize_weight(y, 3, 4, False, grid="ours")))
    # reference symmetric: scale = 2*absmax/(2^b-1), zero 2^(b-1): 4 bits, absmax 7.5 -> scale 1
    # round([-7.5, -3.2, 0.4, 7.4]) = [-8, -3, 0, 7] (half to even); +8 in [0, 15]
    z = torch.tensor([[-7.5, -3.2, 0.4, 7.4]])
    check("ref symmetric: levels (q-8)*2*absmax/15",
          torch.allclose(quantize_weight(z, 4, 4, True, grid="ref"),
                         torch.tensor([[-8.0, -3.0, 0.0, 7.0]]), atol=1e-6),
          str(quantize_weight(z, 4, 4, True, grid="ref")))

    # GPTQ with H = I has nothing to propagate: it must equal RTN on the same grid
    W = torch.randn(16, 128)
    H = torch.eye(128)
    for grid in ("ours", "ref"):
        for sym in (False, True):
            q = gptq_quantize(W, H, 4, 32, sym=sym, grid=grid)
            r = quantize_weight(W, 4, 32, sym, grid=grid)
            check(f"GPTQ(H=I) == RTN on grid={grid} sym={sym}",
                  torch.allclose(q, r, atol=1e-5), f"{(q - r).abs().max():.1e}")
    # and grid=ref GPTQ does not lose to grid=ref RTN on its own objective
    X = torch.randn(512, 128) @ torch.randn(128, 128) * 0.3
    H = X.t() @ X / 512
    q = gptq_quantize(W, H, 3, 32, grid="ref")
    r = quantize_weight(W, 3, 32, False, grid="ref")
    loss = lambda Q: float(((Q - W) @ H * (Q - W)).sum())
    check("GPTQ grid=ref beats RTN grid=ref on the layer-wise loss",
          loss(q) < loss(r), f"{loss(q):.2f} vs {loss(r):.2f}")


def _indefinite(n=64, neg=-0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    lam = torch.ones(n)
    lam[-1] = neg
    return Q @ torch.diag(lam) @ Q.t()


def test_cholesky_fallback():
    print("\n== GPTQ Cholesky fallback ==")
    info = {}
    damped_inverse_cholesky(torch.eye(8) * 2, 0.01, info=info)
    check("PD Hessian: no retry", info == {"damp": 0.01, "retries": 0}, str(info))
    # eigenvalue -0.05, mean diag ~0.98: damp 0.01 fails, 0.1 (one retry) succeeds
    info = {}
    H = _indefinite(neg=-0.05)
    damped_inverse_cholesky(H, 0.01, info=info)
    check("indefinite Hessian: succeeds after one x10 retry",
          info.get("retries") == 1 and abs(info["damp"] - 0.1) < 1e-12, str(info))
    W = torch.randn(4, 64)
    q = gptq_quantize(W, H, 4, 32, info=info)
    check("gptq_quantize goes through with the fallback", bool(torch.isfinite(q).all()))
    # eigenvalue -20, mean diag ~0.67: even damp 10 (3 retries) is not enough -> raise
    raised = False
    try:
        damped_inverse_cholesky(_indefinite(neg=-20.0), 0.01)
    except RuntimeError as e:
        raised = "after 3 retries" in str(e)
    check("hopeless Hessian raises after 3 retries", raised)
    raised = False
    try:
        damped_inverse_cholesky(torch.full((4, 4), float("nan")), 0.01)
    except RuntimeError:
        raised = True
    check("NaN Hessian raises", raised)


def test_fp4():
    print("\n== NVFP4 / MXFP4 ==")
    # E2M1 grid, ties to the even code, saturation
    x = torch.tensor([0.2, 0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 7.0, -0.75, -2.6])
    want = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -1.0, -3.0])
    check("E2M1 rounding (RNE ties, saturate at 6)", torch.equal(round_e2m1(x), want),
          str(round_e2m1(x).tolist()))
    e = round_e4m3(torch.tensor([448.0, 500.0, 164.27, 224.0, 0.3, 17.0]))
    # 164.27 in [128,256) has step 16 -> 160; 0.3 in [0.25,0.5) step 1/32 -> 0.3125;
    # 17 in [16,32) step 2 -> 16 or 18: tie -> even mantissa (16)
    check("E4M3 rounding (saturate at 448)",
          torch.equal(e, torch.tensor([448.0, 448.0, 160.0, 224.0, 0.3125, 16.0])), str(e.tolist()))

    # NVFP4: two blocks of 16 in one row. Tensor amax 6 -> s = 6/(6*448) = 1/448.
    # block 0: amax 6 -> E4M3(448) = 448 -> block scale 1: plain E2M1 rounding
    # block 1: amax 2.2 -> 2.2/6*448 = 164.27 -> E4M3 160 -> scale 160/448 = 0.35714
    #          2.2 / 0.35714 = 6.16 -> 6 -> 2.142857 ; 1.0 / 0.35714 = 2.8 -> 3 -> 1.071429
    w = torch.zeros(1, 32)
    w[0, :4] = torch.tensor([0.3, 1.0, 5.0, -6.0])
    w[0, 16:18] = torch.tensor([2.2, 1.0])
    q = fp4_qdq(w, "nvfp4")
    sc = 160 / 448
    exp = torch.zeros(1, 32)
    exp[0, :4] = torch.tensor([0.5, 1.0, 4.0, -6.0])
    exp[0, 16:18] = torch.tensor([6 * sc, 3 * sc])
    check("NVFP4 block scales: FP8-E4M3 x per-tensor FP32", torch.allclose(q, exp, atol=1e-6),
          str(q[0, 16:18].tolist()))
    # same tensor scaled by 1000: identical up to the factor (global scale absorbs it)
    check("NVFP4 is scale-equivariant through the FP32 tensor scale",
          torch.allclose(fp4_qdq(w * 1000, "nvfp4"), q * 1000, rtol=1e-5))
    # per-row global scale (activations): row 1 = 0.01 x row 0 -> same codes
    two = torch.cat([w, w * 0.01])
    qr = fp4_qdq(two, "nvfp4", global_scale="row")
    check("NVFP4 per-token FP32 scale: rows do not interact",
          torch.allclose(qr[1], qr[0] * 0.01, rtol=1e-5) and torch.allclose(qr[0], q[0]))

    # MXFP4: one block of 32. amax 7 -> floor(log2 7) = 2 -> scale 2^(2-2) = 1
    m = torch.zeros(1, 32)
    m[0, :4] = torch.tensor([0.3, 1.0, 5.0, -7.0])
    qm = fp4_qdq(m, "mxfp4")
    check("MXFP4 scale 2^(floor(log2 amax)-2), E2M1 saturates",
          torch.allclose(qm[0, :4], torch.tensor([0.5, 1.0, 4.0, -6.0])), str(qm[0, :4].tolist()))
    # amax 0.9 -> floor(log2) = -1 -> scale 1/8: 0.9*8 = 7.2 -> 6 -> 0.75; 0.2*8 = 1.6 -> 1.5
    m2 = torch.zeros(1, 32)
    m2[0, :2] = torch.tensor([0.9, 0.2])
    check("MXFP4 power-of-two scale below 1", torch.allclose(
        fp4_qdq(m2, "mxfp4")[0, :2], torch.tensor([0.75, 0.1875])))
    # amax exactly a power of two: 4 -> floor(log2 4) = 2 -> scale 1 -> 4 is exact
    m3 = torch.zeros(1, 32); m3[0, 0] = 4.0; m3[0, 1] = -2.9
    check("MXFP4 exact power of two", torch.allclose(fp4_qdq(m3, "mxfp4")[0, :2],
                                                     torch.tensor([4.0, -3.0])))
    check("all-zero block stays zero", torch.equal(fp4_qdq(torch.zeros(2, 64), "nvfp4"),
                                                   torch.zeros(2, 64)))
    # padding: a last dim that is not a multiple of the block
    odd = torch.randn(3, 40)
    check("non-multiple last dim keeps its shape", fp4_qdq(odd, "mxfp4").shape == odd.shape)

    # specs, fixed blocks, wiring into FakeQuantLinear
    c = QuantConfig.from_ladder("w4a16:wfmt=nvfp4", group_size=128)
    check("wfmt=nvfp4 -> block 16 regardless of --group-size", c.group_size == 16)
    c = QuantConfig.from_ladder("w4a16:wfmt=mxfp4", group_size=128)
    check("wfmt=mxfp4 -> block 32", c.group_size == 32)
    c = QuantConfig.from_ladder("w4a4:wfmt=nvfp4:afmt=nvfp4")
    check("w4a4 nvfp4 weights + activations parse", (c.w_format, c.a_format) == ("nvfp4", "nvfp4"))
    bad = 0
    for s_ in ("w3a16:wfmt=nvfp4", "w4a16:wfmt=nvfp4:gs=64", "w4a16:wfmt=mxfp4:method=gptq",
               "w4a8:afmt=nvfp4", "w4a4:afmt=fp8", "w4a4:afmt=mxfp4:clip=0.99"):
        try:
            QuantConfig.from_ladder(s_)
        except KeyError:
            bad += 1
    check("invalid FP4 combinations are refused", bad == 6, f"{bad}/6")
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 8, bias=False)
    fq = FakeQuantLinear(lin, QuantConfig.from_ladder("w4a4:wfmt=nvfp4:afmt=nvfp4"))
    check("FakeQuantLinear holds NVFP4 weights",
          torch.equal(fq.weight_q, fp4_qdq(lin.weight.data, "nvfp4")))
    xa = torch.randn(2, 5, 64)
    y = fq(xa)
    yref = torch.nn.functional.linear(fp4_qdq(xa, "nvfp4", "row"), fq.weight_q)
    check("FakeQuantLinear quantizes activations per token to NVFP4",
          torch.allclose(y, yref, atol=1e-6))
    eb = {s_: effective_wbits(QuantConfig.from_ladder(s_)) for s_ in
          ("fp16", "w4a16:gs=128", "w4a16:wsym=1:gs=128", "nf4", "w4a16:wfmt=nvfp4",
           "w4a16:wfmt=mxfp4", "w3a16:gs=128")}
    check("effective bits", eb == {"fp16": 16.0, "w4a16:gs=128": 4 + 20 / 128,
                                   "w4a16:wsym=1:gs=128": 4.125, "nf4": 4.5,
                                   "w4a16:wfmt=nvfp4": 4.5, "w4a16:wfmt=mxfp4": 4.25,
                                   "w3a16:gs=128": 3 + 19 / 128}, str(eb))


class PackedLinear(torch.nn.Module):
    """Stands in for a packed-kernel layer (AWQ WQLinear / bnb Linear4bit are not
    nn.Linear): quantize_model must leave it alone, LoRAS-T must still hook around it."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.register_buffer("qweight", lin.weight.data.clone())
        self.bias = lin.bias

    def forward(self, x):
        return torch.nn.functional.linear(x, self.qweight, self.bias)


def packify(lm):
    for layer in lm.layers:
        for mod in list(layer.modules()):
            for n, ch in list(mod.named_children()):
                if isinstance(ch, torch.nn.Linear):
                    setattr(mod, n, PackedLinear(ch))
    return lm


def tiny_corrector_file(path, d=64, n_layers=4, rank=4, seed=0):
    from phoenix import loras_text as LT
    g = torch.Generator().manual_seed(seed)
    corr = {li: LT.HiddenCorrector(0.05 * torch.randn(d, rank, generator=g),
                                   0.05 * torch.randn(rank, d, generator=g),
                                   0.01 * torch.randn(d, generator=g), max_norm=1e9)
            for li in range(n_layers)}
    LT.save(path, corr, {"precision": "w3a16", "rank": rank, "params": 0})
    return corr


def test_packed():
    print("\n== backend=packed ==")
    from phoenix import loras_text as LT
    from phoenix.loras import steer_mask
    from phoenix.ptq import load_quantized
    from phoenix.quant import iter_fq, quantize_model
    spec = "w4a16:backend=packed:ckpt=Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    c = QuantConfig.from_ladder(spec)
    check("packed spec parses (ckpt with a slash)",
          (c.backend, c.ckpt) == ("packed", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"))
    check("row tags: fp16 / sim / bnb / packed",
          [QuantConfig.from_ladder(x).row_backend for x in ("fp16", "w4a16", "bnb-nf4", spec)]
          == ["fp16", "sim", "bnb", "packed"])
    check("only fp16 / bnb / packed rows may be timed",
          [QuantConfig.from_ladder(x).real_kernels for x in ("fp16", "w4a16", "bnb-nf4", spec)]
          == [True, False, True, True])
    errs = {}
    for s_ in ("w4a16:backend=packed", spec + ":wtok=text", "w4a4:backend=packed:ckpt=x:abits_image=8",
               "w4a16:backend=packed:ckpt=x:abits_text=8", "w4a16:backend=packed:ckpt=x:method=gptq",
               "w4a16:ckpt=x"):
        try:
            QuantConfig.from_ladder(s_)
            errs[s_] = None
        except KeyError as e:
            errs[s_] = str(e)
    check("packed refuses: no ckpt, wtok=, abits_image/text, method=, ckpt without packed",
          all(errs.values()), str([k for k, v in errs.items() if not v]))
    check("the wtok error says why", "no FP16 weights" in (errs[spec + ":wtok=text"] or ""))

    from test_smoke import fake_batch, tiny_model
    torch.manual_seed(0)
    lm = packify(tiny_model())
    with redirect_stdout(io.StringIO()):
        wrapped = quantize_model(lm.model, c)
    check("quantize_model wraps nothing for packed", wrapped == [] and not list(iter_fq(lm.model)))
    raised = ""
    try:
        load_quantized("llava-hf/llava-1.5-7b-hf", spec, keep_fp=True)
    except ValueError as e:
        raised = str(e)
    check("calibrating against a packed checkpoint is refused", "FP16 shadow" in raised, raised[:60])

    # LoRAS-T forward hooks + steer_mask on the packed-style model
    tmp = Path(tempfile.mkdtemp())
    try:
        corr = tiny_corrector_file(tmp / "c.pt")
        loaded, _ = LT.load(tmp / "c.pt", lm.device, lm.dtype)
        b = fake_batch()
        ids = b["input_ids"]
        img = ids == lm.image_token_id
        with steer_mask(img):
            base = lm.model(input_ids=ids, output_hidden_states=True).hidden_states
        LT.attach_hidden(lm.layers, loaded)
        with steer_mask(img):
            out = lm.model(input_ids=ids, output_hidden_states=True).hidden_states
        # layer-0 output: correction at text positions only, image positions untouched
        h0 = base[1]
        A, B_, bias = corr[0].A, corr[0].B, corr[0].bias
        exp_text = h0[~img] + (h0[~img] @ A) @ B_ + bias
        check("LoRAS-T hook runs on packed-style layers: text positions corrected",
              torch.allclose(out[1][~img], exp_text, atol=1e-5))
        check("... image positions untouched under steer_mask",
              torch.allclose(out[1][img], h0[img], atol=1e-6))
        LT.detach_hidden(lm.layers)
    finally:
        shutil.rmtree(tmp)


def test_fdr_and_presets():
    print("\n== BH-FDR and presets ==")
    from phoenix.stats import benjamini_hochberg
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    # sorted .005,.01,.03,.04 -> x4/1,4/2,4/3,4/4 = .02,.02,.04,.04 (monotone from the top)
    check("BH q-values, hand-computed", all(abs(a - b) < 1e-12 for a, b in
                                            zip(q, [0.02, 0.04, 0.04, 0.02])), str(q))
    check("BH caps at 1 and keeps order", benjamini_hochberg([0.9, 0.8]) == [0.9, 0.9]
          and benjamini_hochberg([1.0, 1.0, 0.5]) == [1.0, 1.0, 1.0] and benjamini_hochberg([]) == [])
    import importlib.util
    sp = importlib.util.spec_from_file_location("ladder", ROOT / "scripts" / "00_precision_ladder.py")
    lad = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(lad)
    g = lad.PRESETS["gate4bit"]
    ok = True
    for r in lad.PRESETS["gate4bit-qwen"]:
        try:
            QuantConfig.from_ladder(r, group_size=128)
        except KeyError as e:
            ok = False
            print("   bad rung", r, e)
    check("every gate4bit(-qwen) rung parses", ok)
    gs = sorted(QuantConfig.from_ladder(r, group_size=128).group_size for r in g
                if r.startswith("w4a16:method=gptq") and "targets" not in r and "wsym" not in r)
    check("gate4bit: GPTQ at gs 32/64/128 on the reference grid", gs == [32, 64, 128]
          and all(QuantConfig.from_ladder(r).w_grid == "ref" for r in g if "gptq:calib=wiki:grid" in r))
    check("gate4bit spans 16 -> 4 -> 3 bits with the Gate-2 anchors",
          g[0] == "fp16" and "w3a16" in g and "w3a16:method=gptq:calib=wiki" in g
          and "w4a4:method=rot+gptq:calib=wiki" in g and "bnb-nf4" in g)
    check("gate4bit-qwen = gate4bit + the official AWQ checkpoint",
          lad.PRESETS["gate4bit-qwen"][:-1] == g
          and QuantConfig.from_ladder(lad.PRESETS["gate4bit-qwen"][-1]).backend == "packed")
    # add_fdr on hand-built cells
    cells = {"a": {"pope_f1": {"p": 0.001, "significant": True, "delta": -0.1},
                   "chair_i": {"p": 0.04, "significant": True, "delta": 0.01},
                   "blind_ppl_ratio": 1.0},
             "b": {"coverage": {"p": 0.5, "significant": False, "delta": 0.0}}}
    n = lad.add_fdr(cells)
    check("add_fdr: family = table metrics only, q annotated",
          n == 3 and cells["a"]["pope_f1"]["significant_fdr"]
          and not cells["a"]["chair_i"]["significant_fdr"]     # 0.04*3/2 = 0.06
          and abs(cells["a"]["chair_i"]["q_bh"] - 0.06) < 1e-12)


def test_ladder_backends():
    print("\n== ladder end to end: sim / bnb / packed rows ==")
    from test_ladder import build_coco, fake_loader, load_ladder_module
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / "coco"
        build_coco(root)
        mod = load_ladder_module()
        base_loader, calls = fake_loader()
        seen = []

        def loader(model_id, dtype=None, device="cpu", **kw):
            seen.append((model_id, kw))
            lm = base_loader(model_id, dtype, device, **kw)
            return packify(lm) if kw.get("packed") else lm
        mod.load_model = loader
        tiny_corrector_file(tmp / "c.pt")
        packed = "w4a16:backend=packed:ckpt=org/fake-awq"
        out = tmp / "ladder"
        sys.argv = ["00_precision_ladder.py", "--coco-root", str(root), "--out", str(out),
                    "--ladder", f"fp16,w4a16:wfmt=nvfp4,w4a4:wfmt=nvfp4:afmt=nvfp4,"
                    f"w4a16:method=gptq:ncal=4:grid=ref,bnb-nf4,{packed},{packed}+lorast={tmp / 'c.pt'}",
                    "--n-pope-images", "6", "--n-chair-images", "4", "--max-new-tokens", "6",
                    "--device", "cpu", "--dtype", "float32", "--n-boot", "200",
                    "--n-blind-captions", "4"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            mod.main()
        log = buf.getvalue()
        data = json.loads((out / "ladder.json").read_text())
        rep = json.loads((out / "ladder_report.json").read_text())
        tags = {k: v["backend"] for k, v in data["table"].items()}
        check("every row tagged fp16 | sim | bnb | packed", tags == {
            "fp16": "fp16", "w4a16:wfmt=nvfp4": "sim", "w4a4:wfmt=nvfp4:afmt=nvfp4": "sim",
            "w4a16:method=gptq:ncal=4:grid=ref": "sim", "bnb-nf4": "bnb", packed: "packed",
            f"{packed}+lorast={tmp / 'c.pt'}": "packed"}, str(tags))
        pk = [kw for mid, kw in seen if mid == "org/fake-awq"]
        check("packed rung loads ckpt as-is with the base model's processor",
              len(pk) == 2 and all(k.get("packed") and k.get("processor_id") ==
                                   "llava-hf/llava-1.5-7b-hf" for k in pk))
        check("packed rung skips PTQ and wraps nothing",
              "packed checkpoint org/fake-awq -- runs its own kernels" in log
              and "packed checkpoint" in data["table"][packed]["ptq"]["method"])
        check("LoRAS-T attaches to a packed rung", "lorast" in
              data["table"][f"{packed}+lorast={tmp / 'c.pt'}"]["ptq"])
        check("versions logged in ladder.json and ladder_report.json",
              "torch" in data["versions"] and "bitsandbytes" in rep["versions"]
              and "driver" in rep["versions"])
        check("effective bits recorded", data["table"]["w4a16:wfmt=nvfp4"]["eff_wbits"] == 4.5
              and data["table"]["fp16"]["eff_wbits"] == 16.0)
        qs = [c.get("q_bh") for s in rep["summary"].values()
              for k, c in s["vs_reference"].items() if k in rep["fdr"]["metrics"]]
        check("report carries BH q-values next to the unadjusted CIs",
              qs and all(q is not None for q in qs) and rep["fdr"]["n_tests"] == len(qs))
        check("table prints both markers' legend", "Benjamini-Hochberg q <" in log
              and "packed" in log.split("paired comparison")[1])
        # reuse: copy identical rungs from another ladder instead of re-running
        n0 = calls["n"]
        out2 = tmp / "ladder2"
        sys.argv = ["00_precision_ladder.py", "--coco-root", str(root), "--out", str(out2),
                    "--ladder", "fp16,w4a16:wfmt=nvfp4,w4a16:wfmt=mxfp4",
                    "--reuse-from", str(out / "ladder.json"),
                    "--n-pope-images", "6", "--n-chair-images", "4", "--max-new-tokens", "6",
                    "--device", "cpu", "--dtype", "float32", "--n-boot", "200",
                    "--n-blind-captions", "4"]
        with redirect_stdout(io.StringIO()):
            mod.main()
        d2 = json.loads((out2 / "ladder.json").read_text())
        check("--reuse-from copies identical rungs and runs only the new one",
              calls["n"] - n0 == 1 and d2["table"]["fp16"].get("reused_from")
              and "reused_from" not in d2["table"]["w4a16:wfmt=mxfp4"])
    finally:
        shutil.rmtree(tmp)


def test_fused_corrector():
    print("\n== fused LoRAS-T path ==")
    from phoenix import loras_text as LT
    from phoenix.loras import steer_enabled, steer_mask
    g = torch.Generator().manual_seed(1)
    d, r = 256, 16
    A = (0.05 * torch.randn(d, r, generator=g)).half()
    B = (0.05 * torch.randn(r, d, generator=g)).half()
    bias = (0.01 * torch.randn(d, generator=g)).half()
    h = torch.randn(3, 40, d, generator=g).half()
    h[0, 0] *= 200                                   # a massive (BOS-like) token
    img = torch.zeros(3, 40, dtype=torch.bool)
    img[:, 5:25] = True
    norm = float(torch.randn(3, 40, d, generator=g).norm(dim=-1).max()) * 1.25
    for positions in ("text", "all"):
        ref = LT.HiddenCorrector(A, B, bias, max_norm=norm, positions=positions)
        fus = LT.HiddenCorrector(A, B, bias, max_norm=norm, positions=positions, fused=True)
        for name, hh, m in (("prefill", h, img), ("decode", h[:, -1:], None)):
            a, b = ref(hh, m), fus(hh, m)
            diff = float((a.float() - b.float()).abs().max())
            check(f"fused == default ({positions}, {name}) in fp16, atol 1e-3",
                  a.dtype == b.dtype == torch.float16 and diff <= 1e-3, f"{diff:.1e}")
        sel = ref._select(h, img)
        out = fus(h, img)
        check(f"fused leaves unselected positions bit-exact ({positions})",
              torch.equal(out[~sel], h[~sel]) and bool((~sel[0, 0])))
    # end to end on a tiny model: same greedy tokens, same logits, both paths
    from test_smoke import fake_batch, tiny_model
    from phoenix.acab import generate
    torch.manual_seed(0)
    lm = tiny_model()
    tmp = Path(tempfile.mkdtemp())
    try:
        tiny_corrector_file(tmp / "c.pt")
        b = fake_batch()
        img = b["input_ids"] == lm.image_token_id
        outs = {}
        for fused in (False, True):
            corr, _ = LT.load(tmp / "c.pt", lm.device, lm.dtype, fused=fused)
            LT.attach_hidden(lm.layers, corr)
            with steer_mask(img):
                r = generate(lm, {"input_ids": b["input_ids"]}, None, max_new_tokens=8)
            with steer_mask(img):
                logits = lm.model(input_ids=b["input_ids"]).logits
            outs[fused] = (r["sequences"], logits)
            LT.detach_hidden(lm.layers)
        check("tiny model: fused and default generate the same tokens",
              torch.equal(outs[False][0], outs[True][0]))
        check("tiny model: logits match", torch.allclose(outs[False][1], outs[True][1], atol=1e-4),
              f"{(outs[False][1] - outs[True][1]).abs().max():.1e}")
        corr, _ = LT.load(tmp / "c.pt", lm.device, lm.dtype)
        LT.attach_hidden(lm.layers, corr)
        LT.set_fused(lm.layers, True)
        check("set_fused switches attached correctors", all(c.fused for c in corr.values()))
        with steer_mask(img), steer_enabled(False):
            off = lm.model(input_ids=b["input_ids"]).logits
        LT.detach_hidden(lm.layers)
        with steer_mask(img):
            plain = lm.model(input_ids=b["input_ids"]).logits
        check("steer_enabled(False) still disables the fused path", torch.equal(off, plain))
    finally:
        shutil.rmtree(tmp)


def test_latency_harness():
    print("\n== latency harness ==")
    from PIL import Image
    import numpy as np
    from test_fixes import tiny_llava
    from phoenix import latency as Lt
    from phoenix.acab import generate
    lm = tiny_llava()
    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (40, 40, 3), dtype=np.uint8))
    batch = lm.processor(images=[img], text=["describe"])
    clock = Lt.Clock("cpu")
    f = Lt.forced_decode(lm, batch, 12, clock)
    g = generate(lm, batch, None, max_new_tokens=12)
    n = g["sequences"].shape[1]
    check("forced decode == greedy generate() tokens (same cache path)",
          torch.equal(f["tokens"][:, :n], g["sequences"]), f"{n} tokens compared")
    check("forced decode always returns exactly n tokens", f["tokens"].shape == (1, 12))
    check("TPOT = decode-only time / (n-1); e2e = TTFT + decode (prefill not in TPOT)",
          abs(f["tpot_ms"] * 11 - f["decode_ms"]) < 1e-9 and abs(f["e2e_ms"] - f["ttft_ms"] - f["decode_ms"]) < 1e-9)
    tok = lm.processor.tokenizer
    old = tok.eos_token_id
    try:
        tok.eos_token_id = int(f["tokens"][0, 0])        # EOS = the first generated token
        f2 = Lt.forced_decode(lm, batch, 12, clock)
        nat = Lt.natural_decode(lm, batch, 12, clock)
        check("forced length ignores EOS; natural length stops at it",
              f2["tokens"].shape[1] == 12 and nat["len"] == [0], f"natural len {nat['len']}")
    finally:
        tok.eos_token_id = old
    check("natural_length counts tokens before the first EOS",
          Lt.natural_length(torch.tensor([[5, 6, 2, 2], [5, 6, 7, 8]]), 2) == [2, 4])
    feats = Lt.vision_forward(lm, batch)
    feats = feats[0] if isinstance(feats, (list, tuple)) else feats
    check("vision_forward = encoder + projector output (text hidden size)",
          feats.shape[-1] == 64, str(tuple(feats.shape)))
    wb = Lt.weight_bytes(lm.model)
    manual = sum(t.numel() * t.element_size() for t in
                 {id(t): t for t in list(lm.model.parameters()) + list(lm.model.buffers())}.values())
    check("weight_bytes = parameter + buffer bytes", wb == manual, f"{wb} vs {manual}")
    sm = Lt.summarize([1, 2, 3, 4, 100])
    check("summaries: median and p90", sm["median"] == 3 and abs(sm["p90"] - 61.6) < 1e-9, str(sm))
    check("energy reader degrades to None without NVML/CUDA",
          not Lt.Energy("cpu").ok and Lt.Energy("cpu").read() is None)


def test_latency_script():
    print("\n== scripts/15_latency.py end to end (tiny LLaVA, CPU) ==")
    import importlib.util
    from test_fixes import tiny_llava
    from test_ladder import build_coco
    sp = importlib.util.spec_from_file_location("lat", ROOT / "scripts" / "15_latency.py")
    lat = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(lat)
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / "coco"
        build_coco(root, n_images=12)
        loads = []

        def fake_load(spec, args):
            loads.append(spec)
            lm = tiny_llava()
            cfg = QuantConfig.from_ladder(spec)
            if cfg.backend == "packed":
                packify(lm)
            return lm, cfg, {"method": spec}
        lat.load_row_model = fake_load
        tiny_corrector_file(tmp / "c.pt", d=64, n_layers=4)
        base = ["15_latency.py", "--coco-root", str(root), "--device", "cpu", "--dtype", "float32",
                "--n-warmup", "1", "--n-images", "4", "--repeats", "1", "--max-new-tokens", "5",
                "--batch-sweep", "1,2", "--sweep-repeats", "1", "--out", str(tmp / "lat"),
                "--name", "tiny", "--lorast", str(tmp / "c.pt")]
        sys.argv = base + ["--rows", "fp16,w4a16:backend=packed:ckpt=org/fake-awq"]
        with redirect_stdout(io.StringIO()):
            lat.main()
        d = json.loads((tmp / "lat" / "tiny.json").read_text())
        rows = d["rows"]
        check("rows: each config without, with and fused LoRAS-T", list(rows) == [
            "fp16", "fp16+lorast", "fp16+lorast_fused",
            "org_fake-awq", "org_fake-awq+lorast", "org_fake-awq+lorast_fused"], str(list(rows)))
        r = rows["fp16+lorast"]
        check("row has every metric with median and p90",
              all(r[k]["median"] is not None and r[k]["p90"] is not None for k in
                  ("vision_ms", "prefill_ms", "ttft_ms", "tpot_ms", "e2e_forced_ms",
                   "e2e_natural_ms", "len_natural")))
        check("batch sweep recorded (TPOT and throughput)",
              set(r["sweep"]) == {"1", "2"} and r["sweep"]["2"]["decode_tok_s"]["median"] > 0)
        check("LoRAS-T adds its corrector bytes to the weights",
              rows["fp16+lorast"]["weights_gb"] > rows["fp16"]["weights_gb"])
        check("backend tags fp16 / packed", rows["fp16"]["backend"] == "fp16"
              and rows["org_fake-awq"]["backend"] == "packed")
        check("versions logged", "torch" in d["versions"] and "driver" in d["versions"])
        md = (tmp / "lat" / "tiny.md").read_text()
        check("markdown table written", "| fp16+lorast_fused |" in md and "TPOT" in md)
        n0 = len(loads)
        sys.argv = base + ["--rows", "fp16,w4a16:backend=packed:ckpt=org/fake-awq", "--resume"]
        with redirect_stdout(io.StringIO()):
            lat.main()
        check("--resume skips finished rows (no model loaded)", len(loads) == n0)
        sys.argv = base + ["--rows", "w4a16:wfmt=nvfp4"]
        refused = False
        try:
            with redirect_stdout(io.StringIO()):
                lat.main()
        except SystemExit as e:
            refused = "never timed" in str(e)
        check("a simulated rung is refused (never timed)", refused)
    finally:
        shutil.rmtree(tmp)


def test_figures_and_summary():
    print("\n== 16_figures_4bit.py / 17_summary_4bit.py on tiny outputs ==")
    import importlib.util
    import subprocess
    from test_ladder import build_coco, fake_loader, load_ladder_module
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / "coco"
        build_coco(root)
        runs = tmp / "runs_4bit"
        mod = load_ladder_module()
        base_loader, _ = fake_loader()
        mod.load_model = lambda mid, dtype=None, device="cpu", **kw: (
            packify(base_loader(mid, dtype, device)) if kw.get("packed") else base_loader(mid, dtype, device))
        tiny_corrector_file(tmp / "c.pt")
        sys.argv = ["00_precision_ladder.py", "--coco-root", str(root), "--out",
                    str(runs / "llava" / "gate4bit"), "--ladder",
                    f"fp16,w4a16:gs=32,w4a16:wfmt=mxfp4,w3a16,bnb-nf4,bnb-nf4+lorast={tmp / 'c.pt'}",
                    "--n-pope-images", "6", "--n-chair-images", "4", "--max-new-tokens", "6",
                    "--device", "cpu", "--dtype", "float32", "--n-boot", "100",
                    "--n-blind-captions", "2"]
        with redirect_stdout(io.StringIO()):
            mod.main()
        lat = {"rows": {"fp16": {"weights_gb": 14.0, "j_caption": {"median": 50.0}, "wall_min": 1},
                        "bnb-nf4": {"weights_gb": 4.5, "j_caption": {"median": 60.0}, "wall_min": 1},
                        "bnb-nf4+lorast": {"weights_gb": 4.6, "j_caption": {"median": 61.0},
                                           "wall_min": 1}}, "versions": {"gpu": "test"}}
        (runs / "latency").mkdir(parents=True)
        (runs / "latency" / "llava.json").write_text(json.dumps(lat))
        (runs / "latency" / "llava.md").write_text("| row |\n|---|\n| fp16 |\n")
        (runs / "DECISIONS.md").write_text("# t\n\n- a decision\n")
        py = sys.executable
        r = subprocess.run([py, str(ROOT / "scripts" / "16_figures_4bit.py"), "--runs", str(runs),
                            "--out", str(runs / "figures")], capture_output=True, text=True)
        figs = sorted(p.name for p in (runs / "figures").glob("*.png"))
        check("figures (a) per model and (b) written", figs == ["fig_a_llava_bits.png", "fig_b_cost.png"],
              str(figs) + r.stderr[-200:])
        big = runs / "latency" / "huge.json"
        big.write_bytes(b"0" * (5 * 2 ** 20 + 1))
        pub = tmp / "pub"
        r = subprocess.run([py, str(ROOT / "scripts" / "17_summary_4bit.py"), "--runs", str(runs),
                            "--publish", str(pub)], capture_output=True, text=True)
        sm = (runs / "SUMMARY.md").read_text() if (runs / "SUMMARY.md").exists() else ""
        check("SUMMARY.md: environment, GPU-hours, ladder table, latency, figures, deviations",
              all(h in sm for h in ("## Environment", "## GPU-hours per rung", "`w4a16:wfmt=mxfp4`",
                                    "## Latency", "fig_b_cost.png", "- a decision")), r.stderr[-300:])
        check("summary ladder table shows the backend tag and both markers' legend",
              "| `bnb-nf4` | bnb |" in sm and "Benjamini-Hochberg" in sm)
        out = sorted(str(p.relative_to(pub)) for p in pub.rglob("*") if p.is_file())
        check("--publish copies the small files, never ladder.json, DECISIONS.md or files > 5 MB",
              "llava_ladder_report.json" in out and "SUMMARY.md" in out and "DECISIONS.md" not in out
              and "figures/fig_b_cost.png" in out and not any("ladder.json" == Path(o).name for o in out)
              and "latency/huge.json" not in out, str(out))
    finally:
        shutil.rmtree(tmp)


def main():
    test_grid()
    test_cholesky_fallback()
    test_fp4()
    test_packed()
    test_fdr_and_presets()
    test_ladder_backends()
    test_fused_corrector()
    test_latency_harness()
    test_latency_script()
    test_figures_and_summary()
    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        sys.exit(1)
    print("all 4-bit tests passed")


if __name__ == "__main__":
    main()
