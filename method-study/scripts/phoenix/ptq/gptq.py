"""GPTQ (Frantar et al., ICLR 2023): second-order, error-compensating weight rounding.

For a Linear with calibration inputs X (rows = tokens), the layer-wise objective is
    min_Q || X W^T - X Q^T ||_F^2  =  sum_rows (w - q)^T H (w - q),   H = X^T X / n.
GPTQ quantizes one input column at a time and pushes each column's rounding error onto
the not-yet-quantized columns along H^{-1}:
    e = (w_j - q_j) / [H^{-1}]_jj ;   w_{k>j} -= e [H^{-1}]_{j,k}
using the upper Cholesky factor of H^{-1} (numerically stable form) and lazy block
updates of 128 columns. Group parameters (scale, zero) are fitted at each group's
first column, on the error-updated weights, exactly as the reference does.

Layers are processed "true-sequentially": q/k/v, then o_proj (whose input statistics
are recomputed with q/k/v already quantized), then gate/up, then down_proj; the next
decoder layer's inputs come from the quantized layer. Inputs are what the quantizer
will see, so after rotation H is computed from the rotated activations (QuaRot does
GPTQ this way).

Grid (spec key grid=): the default "ours" fits the asymmetric grid to min/max of the
group (our RTN grid), so GPTQ vs RTN differs only in the rounding, not in the grid.
grid=ref uses the reference's min(0, min)/max(0, max) grid (and its symmetric form);
RTN accepts the same key, so a GPTQ-vs-RTN comparison can be made on either grid.

If the damped Hessian is not positive definite, the Cholesky is retried with the
dampening multiplied by 10, up to CHOL_RETRIES times, then it raises.
"""
from __future__ import annotations

import torch

from ..quant import int_grid, int_qdq
from .engine import GROUPS, LayerInput, input_hooks, layer_forward, progress, sub

CHOL_RETRIES = 3


def _grid(x: torch.Tensor, bits: int, sym: bool, grid: str = "ours"):
    """x: [rows, k] -> per-row (scale, zero, unsigned)."""
    scale, zero, unsigned = int_grid(x, bits, sym, grid)
    return scale[:, 0], zero[:, 0], unsigned


def _q(w: torch.Tensor, scale, zero, bits: int, unsigned: bool) -> torch.Tensor:
    return int_qdq(w, scale, zero, unsigned, bits)


def damped_inverse_cholesky(H: torch.Tensor, damp: float, retries: int = CHOL_RETRIES,
                            info: dict | None = None) -> torch.Tensor:
    """Upper Cholesky factor of (H + damp*mean(diag H)*I)^-1, retrying with damp x10."""
    cols = H.shape[0]
    eye = torch.eye(cols, device=H.device, dtype=H.dtype)
    mean_diag = torch.mean(torch.diag(H))
    d, last = damp, None
    for attempt in range(retries + 1):
        try:
            L = torch.linalg.cholesky(H + d * mean_diag * eye)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            if not bool(torch.isfinite(Hinv).all()):
                raise RuntimeError("non-finite inverse Hessian factor")
            if info is not None:
                info["damp"], info["retries"] = d, attempt
            if attempt:
                print(f"[gptq] Cholesky needed damp={d:g} ({attempt} retr"
                      f"{'y' if attempt == 1 else 'ies'})", flush=True)
            return Hinv
        except RuntimeError as e:                # torch.linalg.LinAlgError subclasses it
            last = e
            d *= 10
    raise RuntimeError(f"GPTQ Cholesky failed after {retries} retries "
                       f"(damp {damp:g} -> {d / 10:g}): {last}")


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, group_size: int,
                  sym: bool = False, actorder: bool = False, damp: float = 0.01,
                  grid: str = "ours", info: dict | None = None) -> torch.Tensor:
    W = W.float().clone()
    H = H.float().clone()
    rows, cols = W.shape
    gs = group_size if group_size and group_size > 0 and cols % group_size == 0 else cols
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    static = None
    if actorder:
        # static groups: fitted on the original W, in the original column order
        static = [_grid(W[:, g:g + gs], bits, sym, grid) for g in range(0, cols, gs)]
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
        invperm = torch.argsort(perm)

    Hinv = damped_inverse_cholesky(H, damp, info=info)

    bs = gs if gs <= 1024 else 128
    Q = torch.zeros_like(W)
    scale = zero = None
    unsigned = int_grid(W[:1, :1], bits, sym, grid)[2]
    if gs == cols and not actorder:
        scale, zero, _ = _grid(W, bits, sym, grid)
    for i1 in range(0, cols, bs):
        i2 = min(i1 + bs, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if actorder:
                scale, zero, _ = static[int(perm[c]) // gs]
            elif gs < cols and c % gs == 0:
                scale, zero, _ = _grid(W1[:, i:i + gs], bits, sym, grid)
            w = W1[:, i]
            q = _q(w, scale, zero, bits, unsigned)
            Q1[:, i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            E1[:, i] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if actorder:
        Q = Q[:, invperm]
    return Q


@torch.no_grad()
def gptq(lm, inps: list[LayerInput], cfg) -> dict:
    stats = {"layers": []}
    for layer in progress(lm.layers, "[gptq]"):
        originals = {}
        row = {}
        for _, names in GROUPS:
            names = [n for n in names if not any(s in n for s in cfg.skip_names)]
            if not names:
                continue
            first = sub(layer, names[0])
            acc = {"H": None, "n": 0}

            def fn(_, m, x):
                x = x.float()
                acc["H"] = x.t() @ x if acc["H"] is None else acc["H"] + x.t() @ x
                acc["n"] += x.shape[0]

            with input_hooks({names[0]: first}, fn):
                for x in inps:
                    layer_forward(layer, x)
            H = acc["H"] / max(acc["n"], 1)
            for n in names:
                fc = sub(layer, n)
                W = fc.weight.data
                ci: dict = {}
                Q = gptq_quantize(W, H, cfg.w_bits, cfg.group_size, cfg.w_symmetric,
                                  cfg.gptq_actorder, cfg.gptq_damp,
                                  grid=getattr(cfg, "w_grid", "ours"), info=ci)
                if ci.get("retries"):
                    stats.setdefault("chol_retries", []).append(
                        {"linear": n, "damp": ci["damp"], "retries": ci["retries"]})
                # layer-wise proxy loss, GPTQ vs RTN (same grid), on these inputs
                from ..quant import quantize_weight
                R = quantize_weight(W, cfg.w_bits, cfg.group_size, cfg.w_symmetric,
                                    grid=getattr(cfg, "w_grid", "ours")).float()
                dq, dr = Q - W.float(), R - W.float()
                row[n] = (float((dr @ H * dr).sum()), float((dq @ H * dq).sum()))
                originals[n] = W.clone()
                fc.weight.data = Q.to(W.dtype)               # later groups see it quantized
            del H, acc
        inps[:] = [LayerInput(layer_forward(layer, x), x.kwargs, x.image_mask) for x in inps]
        for n, W in originals.items():
            fc = sub(layer, n)
            fc.phoenix_wq = fc.weight.data
            fc.weight.data = W
        stats["layers"].append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ratios = [v[1] / max(v[0], 1e-20) for r in stats["layers"] for v in r.values()]
    stats["mean_loss_ratio_vs_rtn"] = sum(ratios) / max(len(ratios), 1)
    print(f"[gptq] layer-wise loss {stats['mean_loss_ratio_vs_rtn']:.3f}x of RTN's "
          "(mean over Linears, on calibration inputs)", flush=True)
    return stats
