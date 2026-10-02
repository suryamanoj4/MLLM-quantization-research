"""Contribution 2 -- LoRAS: Low-Rank Activation Steering.

The proposal learns W_A, W_B by SGD on an MSE loss (Eq. 4). That objective has a
*closed-form global optimum*: it is reduced-rank ridge regression. Concretely, for

    min_{rank(M) <= r}  || R - X M ||_F^2 ,   R = H_fp - H_q,  X = H_q

the solution is

    M_ols = (Cxx + lambda I)^{-1} Cxr
    G     = M_ols^T Cxx M_ols            (d x d, PSD)
    V_r   = top-r eigenvectors of G
    M_r   = M_ols V_r V_r^T   =>   A = M_ols V_r  (d x r),  B = V_r^T  (r x d)

(Izenman 1975; Eckart-Young applied to the fitted values, not to M itself -- the
distinction matters, truncating the SVD of M_ols directly is *not* optimal.)

Consequences, and why this is worth doing:
  * seconds instead of minutes, no learning rate / epochs / early stopping,
  * exactly reproducible -- no seed variance to report,
  * the eigenvalues of G are an *explained-variance spectrum* of the correctable
    error, so you get the rank-vs-recovery curve for free from one fit. That curve
    is also the honest answer to "how much of the quantization error is even
    linearly recoverable" -- if the r -> d limit only removes 30% of the error, no
    amount of SGD on a rank-16 adapter will do better, and you should know that in
    week 4 rather than week 11.

Everything is computed from streaming sufficient statistics (Cxx, Cxr, means,
||R||^2), so calibration memory is O(d^2) per site and independent of the number of
calibration tokens.

Two steering sites:
  "kv"     -- correct the *outputs* of k_proj / v_proj at visual positions
              (pre-RoPE, so the map is position independent). Because visual tokens
              only ever appear in the prefix, the correction runs once during
              prefill and the KV cache carries it for free: **zero decode-time
              overhead**, unlike the proposal's Eq. 3 which perturbs the residual
              stream at every layer.
  "hidden" -- correct the residual stream at the layer input (closer to Eq. 3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# global steering context: which token positions are visual in the current fwd
# --------------------------------------------------------------------------- #
class SteerContext:
    """Process-global visual mask for the forward pass currently in flight."""
    mask: torch.Tensor | None = None       # [B, T] bool
    enabled: bool = True

    @classmethod
    def set(cls, mask: torch.Tensor | None) -> None:
        cls.mask = mask

    @classmethod
    def clear(cls) -> None:
        cls.mask = None


class steer_mask:
    """`with steer_mask(m): model(...)`"""

    def __init__(self, mask: torch.Tensor | None):
        self.mask = mask
        self._prev = None

    def __enter__(self):
        self._prev = SteerContext.mask
        SteerContext.mask = self.mask
        return self

    def __exit__(self, *a):
        SteerContext.mask = self._prev


class steer_enabled:
    def __init__(self, on: bool):
        self.on = on
        self._prev = None

    def __enter__(self):
        self._prev = SteerContext.enabled
        SteerContext.enabled = self.on
        return self

    def __exit__(self, *a):
        SteerContext.enabled = self._prev


# --------------------------------------------------------------------------- #
# sufficient statistics + closed-form solver
# --------------------------------------------------------------------------- #
@dataclass
class RRRStats:
    d_in: int
    d_out: int
    device: torch.device
    n: int = 0
    _init: bool = field(default=False, repr=False)

    def __post_init__(self):
        dev, f64 = self.device, torch.float64
        self.Sxx = torch.zeros(self.d_in, self.d_in, device=dev, dtype=f64)
        self.Sxr = torch.zeros(self.d_in, self.d_out, device=dev, dtype=f64)
        self.sx = torch.zeros(self.d_in, device=dev, dtype=f64)
        self.sr = torch.zeros(self.d_out, device=dev, dtype=f64)
        self.ssr = torch.zeros((), device=dev, dtype=f64)
        self._init = True

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """x, y: [N, d] quantized and FP16 activations at the same positions."""
        x = x.detach().float()
        r = y.detach().float() - x
        self.Sxx += (x.T @ x).double()
        self.Sxr += (x.T @ r).double()
        self.sx += x.sum(0).double()
        self.sr += r.sum(0).double()
        self.ssr += r.pow(2).sum().double()
        self.n += x.shape[0]

    # -- centred second-moment matrices -------------------------------------- #
    def _centred(self):
        n = max(self.n, 1)
        mx, mr = self.sx / n, self.sr / n
        Cxx = self.Sxx - n * torch.outer(mx, mx)
        Cxr = self.Sxr - n * torch.outer(mx, mr)
        ssr = self.ssr - n * mr.pow(2).sum()
        return Cxx, Cxr, ssr, mx, mr

    @torch.no_grad()
    def solve(self, rank: int, ridge: float = 1e-2, use_bias: bool = True,
              energy: float | None = None) -> dict:
        """Return A [d_in, r], B [r, d_out], bias [d_out] and the recovery spectrum."""
        Cxx, Cxr, ssr, mx, mr = self._centred()
        d = self.d_in
        lam = ridge * (torch.diagonal(Cxx).mean().clamp(min=1e-12))
        Creg = Cxx + lam * torch.eye(d, device=Cxx.device, dtype=Cxx.dtype)

        M_ols = torch.linalg.solve(Creg, Cxr)                 # [d_in, d_out]
        G = M_ols.T @ Cxx @ M_ols                             # [d_out, d_out] PSD
        G = 0.5 * (G + G.T)
        evals, evecs = torch.linalg.eigh(G)                   # ascending
        evals = evals.flip(0).clamp(min=0)
        evecs = evecs.flip(1)

        # explained-error spectrum: SSE(r) = ssr - 2 tr(V_r^T M_ols^T Cxr V_r) + sum_{i<=r} eval_i
        Q = M_ols.T @ Cxr                                     # [d_out, d_out]
        Q = 0.5 * (Q + Q.T)
        cross = torch.einsum("ij,jk,ki->i", evecs.T, Q, evecs)  # diag(V^T Q V)
        # With the optimal bias, uncentred SSE == centred SSE, so we can report the
        # reduction against the *total* error energy ||R||^2 -- "what fraction of the
        # quantization error does LoRAS remove" -- rather than a centred proxy.
        sse_curve = ssr - 2 * torch.cumsum(cross, 0) + torch.cumsum(evals, 0)
        denom = self.ssr.clamp(min=1e-12)
        red_curve = (1.0 - sse_curve / denom).clamp(min=0, max=1)
        # ridge can make the curve very slightly non-monotone; searchsorted needs sorted
        red_curve = torch.cummax(red_curve, dim=0).values

        if energy is not None:
            target = float(energy) * float(red_curve[-1])
            r_sel = int(torch.searchsorted(red_curve, torch.tensor(
                target, device=red_curve.device, dtype=red_curve.dtype)).item()) + 1
            rank = max(1, min(r_sel, self.d_out))
        rank = int(max(1, min(rank, self.d_out)))

        Vr = evecs[:, :rank]                                  # [d_out, r]
        A = (M_ols @ Vr)                                      # [d_in, r]
        B = Vr.T.contiguous()                                 # [r, d_out]
        bias = (mr - mx @ (A @ B)) if use_bias else torch.zeros_like(mr)

        return {
            "A": A, "B": B, "bias": bias, "rank": rank,
            # exact, recomputed from the sufficient statistics for the actual
            # (A, B, bias) triple -- so it is correct with or without the bias term
            "rel_mse_reduction": self.evaluate(A, B, bias),
            "ceiling": float(red_curve[-1]),
            "curve": red_curve.detach().float().cpu(),
            "n_tokens": self.n,
            "ssr": float(ssr),
        }

    @torch.no_grad()
    def evaluate(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> float:
        """Relative MSE reduction of a given corrector on *these* statistics (held-out)."""
        M = (A @ B).double()
        b = bias.double()
        n = max(self.n, 1)
        # ||R - (X M + b)||^2 = ssr - 2 tr(M^T Sxr) - 2 b.sr + tr(M^T Sxx M)
        #                       + 2 b^T M^T sx + n b^T b
        sse = (self.ssr
               - 2 * torch.trace(M.T @ self.Sxr)
               - 2 * (b * self.sr).sum()
               + torch.trace(M.T @ self.Sxx @ M)
               + 2 * (b @ (M.T @ self.sx))
               + n * (b * b).sum())
        return float((1.0 - sse / self.ssr.clamp(min=1e-12)).clamp(min=-10, max=1))

    def free(self):
        for k in ("Sxx", "Sxr", "sx", "sr", "ssr"):
            setattr(self, k, None)


# --------------------------------------------------------------------------- #
# the steering module
# --------------------------------------------------------------------------- #
class LoRASCorrector(nn.Module):
    """y <- y + (y @ A) @ B + bias, applied only at visual token positions."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor,
                 scale: float = 1.0):
        super().__init__()
        self.register_buffer("A", A, persistent=True)
        self.register_buffer("B", B, persistent=True)
        self.register_buffer("bias", bias, persistent=True)
        self.scale = float(scale)

    @property
    def rank(self) -> int:
        return self.A.shape[1]

    def forward(self, y: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None or self.scale == 0.0:
            return y
        if y.shape[:2] != mask.shape:
            return y                      # decode step / shape mismatch -> no-op
        if not bool(mask.any()):
            return y
        yv = y[mask]
        corr = (yv @ self.A) @ self.B + self.bias
        out = y.clone()
        out[mask] = yv + self.scale * corr.to(yv.dtype)
        return out


class LoRASProj(nn.Module):
    """Wraps k_proj / v_proj and applies the corrector to its output."""

    def __init__(self, base: nn.Module, corrector: LoRASCorrector | None = None):
        super().__init__()
        self.base = base
        self.corrector = corrector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.corrector is not None and SteerContext.enabled:
            y = self.corrector(y, SteerContext.mask)
        return y


# --------------------------------------------------------------------------- #
# attach / detach helpers
# --------------------------------------------------------------------------- #
SITES = ("k", "v")


def _attn_of(layer: nn.Module) -> nn.Module:
    for n in ("self_attn", "attention", "attn"):
        m = getattr(layer, n, None)
        if m is not None:
            return m
    raise RuntimeError("no self-attention submodule found on decoder layer")


def wrap_projections(layers: nn.ModuleList, layer_ids: Sequence[int],
                     sites: Sequence[str] = SITES) -> dict[tuple[int, str], LoRASProj]:
    """Insert (initially empty) LoRASProj wrappers so hooks/correctors can be added."""
    handles: dict[tuple[int, str], LoRASProj] = {}
    for li in layer_ids:
        attn = _attn_of(layers[li])
        for s in sites:
            name = f"{s}_proj"
            base = getattr(attn, name)
            if isinstance(base, LoRASProj):
                handles[(li, s)] = base
                continue
            w = LoRASProj(base)
            setattr(attn, name, w)
            handles[(li, s)] = w
    return handles


def unwrap_projections(layers: nn.ModuleList) -> None:
    for layer in layers:
        attn = _attn_of(layer)
        for s in SITES:
            name = f"{s}_proj"
            m = getattr(attn, name, None)
            if isinstance(m, LoRASProj):
                setattr(attn, name, m.base)


def set_loras_scale(model_layers: nn.ModuleList, scale: float) -> None:
    for layer in model_layers:
        attn = _attn_of(layer)
        for s in SITES:
            m = getattr(attn, f"{s}_proj", None)
            if isinstance(m, LoRASProj) and m.corrector is not None:
                m.corrector.scale = scale


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save_loras(path: str, correctors: dict[tuple[int, str], LoRASCorrector], meta: dict) -> None:
    state = {f"{li}|{s}": {"A": c.A.cpu(), "B": c.B.cpu(), "bias": c.bias.cpu()}
             for (li, s), c in correctors.items()}
    torch.save({"state": state, "meta": meta}, path)


def load_loras(path: str, device, dtype) -> tuple[dict[tuple[int, str], LoRASCorrector], dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for key, t in blob["state"].items():
        li, s = key.split("|")
        out[(int(li), s)] = LoRASCorrector(
            t["A"].to(device=device, dtype=dtype),
            t["B"].to(device=device, dtype=dtype),
            t["bias"].to(device=device, dtype=dtype),
        )
    return out, blob["meta"]


def attach_loras(layers: nn.ModuleList,
                 correctors: dict[tuple[int, str], LoRASCorrector]) -> None:
    layer_ids = sorted({li for li, _ in correctors})
    wrap_projections(layers, layer_ids)
    for (li, s), c in correctors.items():
        attn = _attn_of(layers[li])
        proj = getattr(attn, f"{s}_proj")
        assert isinstance(proj, LoRASProj), "call wrap_projections first"
        proj.corrector = c


def loras_param_count(correctors: dict) -> int:
    return sum(c.A.numel() + c.B.numel() + c.bias.numel() for c in correctors.values())


def summarise(results: dict) -> str:
    lines = ["layer site  rank   fit_red  val_red  ceiling"]
    for (li, s), r in sorted(results.items()):
        lines.append(f"{li:5d} {s:>4}  {r['rank']:4d}  {r['rel_mse_reduction']:7.3f} "
                     f" {r.get('val_red', float('nan')):7.3f}  {r['ceiling']:7.3f}")
    return "\n".join(lines)


def cosine_gain(x_q: torch.Tensor, x_fp: torch.Tensor,
                corrector: LoRASCorrector | None = None) -> tuple[float, float]:
    """Mean cosine similarity to FP16 before/after correction."""
    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()
    before = cos(x_q, x_fp)
    if corrector is None:
        return before, before
    corr = (x_q @ corrector.A) @ corrector.B + corrector.bias
    return before, cos(x_q + corr.to(x_q.dtype), x_fp)


def suggest_rank(curve: torch.Tensor, energy: float = 0.9, cap: int = 256) -> int:
    """Smallest rank reaching `energy` of the *achievable* (full-rank) recovery."""
    curve = torch.cummax(curve, dim=0).values
    target = float(energy) * float(curve[-1])
    idx = int(torch.searchsorted(curve, torch.tensor(target, dtype=curve.dtype)).item()) + 1
    return int(max(1, min(idx, cap, len(curve))))


def fmt_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n}B"


def math_check() -> None:  # pragma: no cover - used by tests/docs
    """Sanity: RRR at full rank must equal ridge OLS."""
    torch.manual_seed(0)
    n, d = 2000, 16
    X = torch.randn(n, d, dtype=torch.float64)
    Mt = torch.randn(d, d, dtype=torch.float64) * 0.1
    R = X @ Mt + 0.01 * torch.randn(n, d, dtype=torch.float64)
    st = RRRStats(d, d, torch.device("cpu"))
    st.update(X, X + R)
    full = st.solve(rank=d, ridge=1e-8, use_bias=True)
    assert full["rel_mse_reduction"] > 0.99, full["rel_mse_reduction"]
    assert math.isfinite(full["ceiling"])
