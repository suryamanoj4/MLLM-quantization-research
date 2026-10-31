#!/usr/bin/env python
"""Systems metrics: TTFT, decode throughput, peak VRAM, for each intervention.

This is the number the proposal commits to ("<1.5% latency impact"), so measure it
rather than assert it. LoRAS should cost ~0 at decode time by construction -- the
correction is applied to visual K/V during prefill only and the KV cache carries it
-- so any LoRAS decode overhead you see is a bug in the masking, and this script is
how you catch it.
"""
from _common import (acab_cfg_from_args, add_acab_args, add_loras_args,
                     base_parser, build_model, maybe_attach_loras, run_dir)

from phoenix.data import build_caption_set
from phoenix.evaluate import acab_session, measure_latency
from phoenix.utils import save_json


def main():
    p = base_parser(__doc__)
    add_loras_args(p)
    add_acab_args(p)
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    samples = build_caption_set(args.coco_root, args.eval_subset, n_images=32,
                                seed=args.seed)
    out = {}

    for name, use_loras, acab_mode, gate in [
        ("baseline", False, "off", "prev"),
        ("loras", True, "off", "prev"),
        ("acab_add_prev", False, "add", "prev"),
        ("acab_add_twopass", False, "add", "two_pass"),
        ("acab_mult_eager", False, "mult", "prev"),
        ("loras+acab", True, "add", "prev"),
    ]:
        if use_loras and not args.loras:
            continue
        lm, _ = build_model(args, keep_fp=False)
        if use_loras:
            maybe_attach_loras(lm, args)
        args.acab, args.gate = acab_mode, gate
        cfg = acab_cfg_from_args(args)
        with acab_session(lm, cfg) as ctrl:
            m = measure_latency(lm, samples, ctrl, n_trials=args.n_trials,
                                max_new_tokens=args.max_new_tokens)
        out[name] = m
        print(f"{name:18s} TTFT={m['ttft_ms_median']:7.1f}ms  "
              f"decode={m['decode_tok_per_s_median']:6.2f} tok/s  "
              f"VRAM={m['peak_vram_gb']:.2f}GB", flush=True)
        del lm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if "baseline" in out:
        b = out["baseline"]["decode_tok_per_s_median"]
        for k, v in out.items():
            v["decode_overhead_pct"] = 100 * (b / max(v["decode_tok_per_s_median"], 1e-9) - 1)
            print(f"{k:18s} decode overhead vs baseline: {v['decode_overhead_pct']:+.2f}%")

    d = run_dir(args, "latency")
    save_json({"config": vars(args), "results": out}, d / "latency.json")
    print(f"\nwritten to {d/'latency.json'}")


if __name__ == "__main__":
    main()
