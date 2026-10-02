"""LoRAS calibration driver: sequential, closed-form, memory-bounded.

Sequential (error-propagating) calibration matters. If every layer is fitted
against the *uncorrected* quantized activations, each corrector is solving a
problem that will not exist at inference time once the earlier correctors are
installed -- the same mismatch GPTQ/OmniQuant avoid by calibrating block by block.
Here, when block B is being fitted, every corrector in blocks < B is already live
on the quantized forward pass, while the FP16 targets come from a clean FP16 pass
with all steering disabled.

`layers_per_pass` trades fidelity for wall-clock: 1 = strictly sequential
(n_layers passes over the calibration set), 32 = fully greedy (1 pass).
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch

from .loras import (LoRASCorrector, RRRStats, attach_loras, save_loras,
                    steer_enabled, steer_mask, wrap_projections)
from .probes import kv_capture
from .quant import quant_mode


def _chunks(seq: Sequence[int], size: int) -> list[list[int]]:
    return [list(seq[i:i + size]) for i in range(0, len(seq), size)]


@torch.no_grad()
def calibrate_loras(lm,
                    batch_iter_fn: Callable[[], Iterable[dict]],
                    visual_mask_fn: Callable,
                    layer_ids: Sequence[int],
                    rank: int = 16,
                    ridge: float = 1e-2,
                    sites: Sequence[str] = ("k", "v"),
                    layers_per_pass: int = 4,
                    use_bias: bool = True,
                    energy: float | None = None,
                    val_frac: float = 0.2,
                    stats_device: str | None = None,
                    verbose: bool = True) -> tuple[dict, dict]:
    """Fit LoRAS correctors. Returns ({(layer, site): corrector}, diagnostics)."""
    model = lm.model
    dev = torch.device(stats_device) if stats_device else lm.device
    layer_ids = sorted(layer_ids)
    wrap_projections(lm.layers, layer_ids, sites)

    correctors: dict[tuple[int, str], LoRASCorrector] = {}
    diagnostics: dict[tuple[int, str], dict] = {}

    for bi, block in enumerate(_chunks(layer_ids, layers_per_pass)):
        if verbose:
            print(f"[loras] pass {bi + 1}/{len(_chunks(layer_ids, layers_per_pass))} "
                  f"-> layers {block}")
        fit: dict[tuple[int, str], RRRStats] = {}
        val: dict[tuple[int, str], RRRStats] = {}

        for j, batch in enumerate(batch_iter_fn()):
            mask = visual_mask_fn(batch["input_ids"])
            is_val = (j % max(1, int(round(1 / max(val_frac, 1e-6)))) == 0) and val_frac > 0

            cap_fp = kv_capture(lm.layers, block, sites); cap_fp.set_mask(mask)
            with steer_enabled(False), quant_mode(model, False):
                model(**batch, use_cache=False)
            target = {k: v.clone() for k, v in cap_fp.store.items()}
            cap_fp.remove()

            cap_q = kv_capture(lm.layers, block, sites); cap_q.set_mask(mask)
            with steer_mask(mask), steer_enabled(True), quant_mode(model, True):
                model(**batch, use_cache=False)
            source = dict(cap_q.store)
            cap_q.remove()

            store = val if is_val else fit
            for key, x in source.items():
                y = target[key]
                if key not in store:
                    store[key] = RRRStats(x.shape[-1], y.shape[-1], dev)
                store[key].update(x.to(dev), y.to(dev))
            del target, source
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        new: dict[tuple[int, str], LoRASCorrector] = {}
        for key, st in fit.items():
            sol = st.solve(rank=rank, ridge=ridge, use_bias=use_bias, energy=energy)
            c = LoRASCorrector(sol["A"].to(lm.dtype).to(lm.device),
                               sol["B"].to(lm.dtype).to(lm.device),
                               sol["bias"].to(lm.dtype).to(lm.device))
            new[key] = c
            d = {k: sol[k] for k in ("rank", "rel_mse_reduction", "ceiling", "n_tokens")}
            d["curve"] = sol["curve"][: min(len(sol["curve"]), 512)].tolist()
            if key in val:
                d["val_red"] = val[key].evaluate(sol["A"], sol["B"], sol["bias"])
            diagnostics[key] = d
            st.free()
        for st in val.values():
            st.free()

        attach_loras(lm.layers, new)          # live for the next block
        correctors.update(new)
        if verbose:
            for key in sorted(new):
                d = diagnostics[key]
                print(f"        L{key[0]:02d}{key[1]} rank={d['rank']:3d} "
                      f"fit={d['rel_mse_reduction']:.3f} "
                      f"val={d.get('val_red', float('nan')):.3f} "
                      f"ceiling={d['ceiling']:.3f}")

    return correctors, diagnostics


def save(path, correctors, diagnostics, meta):
    meta = dict(meta)
    meta["diagnostics"] = {f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items() if kk != "curve"}
                           for k, v in diagnostics.items()}
    save_loras(path, correctors, meta)
