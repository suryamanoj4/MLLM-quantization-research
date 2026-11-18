#!/usr/bin/env python
"""CPU tests for the 4-bit / latency additions (branch exp/4bit-latency).

    python tests/test_4bit.py

  * grid=ref (GPTQ reference grid) for RTN and GPTQ; the default grid is unchanged
  * GPTQ Cholesky fallback: damp x10 up to 3 retries, then raise
  * NVFP4 / MXFP4 weight and activation formats against hand-computed values
"""
from __future__ import annotations

import sys
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


def main():
    test_grid()
    test_cholesky_fallback()
    test_fp4()
    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        sys.exit(1)
    print("all 4-bit tests passed")


if __name__ == "__main__":
    main()
