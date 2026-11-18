#!/usr/bin/env python
"""CPU tests for the 4-bit / latency additions (branch exp/4bit-latency).

    python tests/test_4bit.py

  * grid=ref (GPTQ reference grid) for RTN and GPTQ; the default grid is unchanged
  * GPTQ Cholesky fallback: damp x10 up to 3 retries, then raise
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
from phoenix.quant import QuantConfig, quantize_weight

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


def main():
    test_grid()
    test_cholesky_fallback()
    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        sys.exit(1)
    print("all 4-bit tests passed")


if __name__ == "__main__":
    main()
