#!/usr/bin/env python
"""Real-kernel latency, memory and energy: fp16, bnb-nf4 and (Qwen) the official AWQ
checkpoint, each without and with LoRAS-T. 05_latency.py is left as it was.

    python scripts/15_latency.py --coco-root data/coco-mini --model-id llava-hf/llava-1.5-7b-hf \\
        --out runs_4bit/latency --name llava --resume
    python scripts/15_latency.py --coco-root data/coco-mini --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --dtype bfloat16 --out runs_4bit/latency --name qwen --resume

Protocol (see phoenix/latency.py for what each number is):
  * batch 1 except the batch sweep; one fixed image resolution: LLaVA's processor
    always crops to 336x336 (576 visual tokens); Qwen images are resized to
    --qwen-side x --qwen-side and min_pixels = max_pixels = side^2 (default 672 ->
    576 visual tokens, the same count as LLaVA);
  * --n-warmup 10 untimed runs, then --n-images 32 distinct images x --repeats 5;
  * TPOT is decode-only (forced length, EOS ignored); e2e at forced length
    (--max-new-tokens 128) and at natural length (the eval path, stops at EOS);
  * weights GB = parameter + buffer bytes; peak = torch.cuda.max_memory_allocated;
  * energy from NVML's total-energy counter: J/caption (natural) and J/token (forced);
  * batch sweep (--batch-sweep 1,4,16): TPOT and decode / end-to-end tokens per second.

LoRAS-T rows reuse an existing corrector (runs/lorast*/ or runs_qwen/lorast*/, w3a16
RTN, text positions, the model's hidden size); --lorast PATH forces one. Each LoRAS-T
row is timed twice: the default corrector path and the fused one (--no-fused skips
it). The correctors were fitted on the simulated W3 rung: on these rows only their
cost is measured, not their effect on quality.

Only rows whose backend is fp16 / bnb / packed can be passed (--rows): simulated rungs
run dequantized FP16 matmuls, so their timings would mean nothing.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from phoenix import latency as L
from phoenix import loras_text as LT
from phoenix.data import build_caption_set, open_image
from phoenix.model import prepare_batch
from phoenix.ptq import load_quantized
from phoenix.quant import QuantConfig, precision_slug
from phoenix.utils import resolve_dtype, save_json, set_seed, versions

QWEN_AWQ = "w4a16:backend=packed:ckpt=Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
CORRECTOR_GLOBS = ("runs/lorast*/*.pt", "runs_qwen/lorast*/*.pt", "runs_qwen/**/lorast*/*.pt",
                   "runs_4bit/lorast*/*.pt", "runs*/q1_*/**/*.pt")


def default_rows(model_id: str) -> list[str]:
    rows = ["fp16", "bnb-nf4"]
    if "qwen" in model_id.lower():
        rows.append(QWEN_AWQ)
    return rows


def find_corrector(hidden: int, explicit: str = "") -> tuple[str | None, str]:
    """A w3a16-RTN text-position LoRAS-T file for this hidden size, preferring +QA."""
    if explicit:
        return explicit, "given with --lorast"
    cands = []
    for g in CORRECTOR_GLOBS:
        for p in sorted(Path(".").glob(g)):
            try:
                blob = torch.load(p, map_location="cpu", weights_only=False)
                meta, state = blob.get("meta", {}), blob.get("state", {})
                first = next(iter(state.values()))
            except Exception:                                       # noqa: BLE001
                continue
            if (meta.get("precision") == "w3a16" and meta.get("positions", "text") == "text"
                    and first["A"].shape[0] == hidden):
                cands.append((int(meta.get("qa_per_image", 0) or 0) > 0, str(p)))
    if not cands:
        return None, "no matching corrector found"
    cands.sort(key=lambda c: (not c[0], c[1]))
    return cands[0][1], f"found {len(cands)} candidate(s); using the first (+QA preferred)"


def load_row_model(spec: str, args):
    cfg = QuantConfig.from_ladder(spec, group_size=128, targets=("language",))
    if not cfg.real_kernels:
        raise SystemExit(f"{spec} is a simulated rung (backend=sim): it runs dequantized FP16 "
                         "matmuls, so it is never timed. Use fp16, bnb-nf4 or a packed ckpt.")
    lm, cfg, info = load_quantized(args.model_id, spec, args.coco_root,
                                   dtype=resolve_dtype(args.dtype), device=args.device,
                                   keep_fp=False, seed=args.seed, group_size=128)
    return lm, cfg, info


def fix_resolution(lm, side: int) -> dict:
    """Pin Qwen's resize to exactly side x side; LLaVA's processor is fixed already."""
    ip = getattr(lm.processor, "image_processor", None)
    if getattr(lm.model.config, "model_type", "") != "qwen2_5_vl" or ip is None:
        return {"family": "llava", "resolution": "336x336 (processor resize + center crop)"}
    if side % 28:
        raise SystemExit("--qwen-side must be a multiple of 28 (patch 14 x merge 2)")
    px = side * side
    ip.min_pixels = ip.max_pixels = px
    if isinstance(getattr(ip, "size", None), dict):
        ip.size = {"shortest_edge": px, "longest_edge": px}
    return {"family": "qwen2_5_vl", "resolution": f"{side}x{side}", "min_pixels": px,
            "max_pixels": px}


def prepare(lm, samples, side: int | None, B: int = 1) -> list[dict]:
    out = []
    for i in range(0, len(samples) - B + 1, B):
        chunk = samples[i:i + B]
        imgs = [open_image(s.image_path) for s in chunk]
        if side:
            imgs = [im.resize((side, side)) for im in imgs]
        out.append(prepare_batch(lm, imgs, [s.question for s in chunk]))
    return out


def n_visual(lm, batches) -> list[int]:
    return sorted({int((b["input_ids"][j] == lm.image_token_id).sum())
                   for b in batches for j in range(b["input_ids"].shape[0])})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--coco-root", required=True)
    p.add_argument("--eval-subset", default="val2014")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rows", default="", help="comma list of specs (default: fp16,bnb-nf4"
                                              "[,the official Qwen AWQ checkpoint])")
    p.add_argument("--lorast", default="", help="LoRAS-T corrector .pt (default: search runs*/)")
    p.add_argument("--no-lorast", action="store_true")
    p.add_argument("--no-fused", action="store_true", help="skip the fused LoRAS-T timing")
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-images", type=int, default=32)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--batch-sweep", default="1,4,16")
    p.add_argument("--sweep-repeats", type=int, default=3)
    p.add_argument("--qwen-side", type=int, default=672)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs_4bit/latency")
    p.add_argument("--name", default="", help="output basename (default: llava | qwen)")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    name = args.name or ("qwen" if "qwen" in args.model_id.lower() else "llava")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json, out_md = out_dir / f"{name}.json", out_dir / f"{name}.md"
    env = versions()
    print("[versions]", env, flush=True)
    specs = [s.strip() for s in (args.rows or ",".join(default_rows(args.model_id))).split(",")
             if s.strip()]
    for s in specs:                                     # fail on sim rungs before loading
        if not QuantConfig.from_ladder(s).real_kernels:
            raise SystemExit(f"{s}: simulated rungs are never timed (backend=sim)")
    sweep_bs = [int(b) for b in args.batch_sweep.split(",") if b.strip()]
    keys = ("model_id", "dtype", "n_images", "repeats", "max_new_tokens", "qwen_side",
            "batch_sweep", "n_warmup", "seed")
    result = {"config": vars(args), "versions": env, "rows": {}, "notes": []}
    if args.resume and out_json.exists():
        prev = json.loads(out_json.read_text())
        if all(prev["config"].get(k) == getattr(args, k) for k in keys):
            result["rows"] = prev["rows"]
            result["notes"] = prev.get("notes", [])
            print(f"[resume] reusing {len(result['rows'])} row(s): {list(result['rows'])}")
        else:
            print("[resume] settings differ from the saved file; starting fresh")
    samples = build_caption_set(args.coco_root, args.eval_subset, n_images=args.n_images,
                                seed=args.seed)
    if len(samples) < max(sweep_bs + [1]):
        raise SystemExit(f"need >= {max(sweep_bs)} images for the batch sweep")

    def save():
        result["versions"] = env
        save_json(result, out_json)
        out_md.write_text(L.markdown_table(result["rows"], name, args.max_new_tokens) + "\n")

    energy = L.Energy(args.device)
    if not energy.ok:
        print(f"[energy] {energy.why}; J/caption and J/token will be empty", flush=True)
        if energy.why not in result["notes"]:
            result["notes"].append(energy.why)

    for spec in specs:
        variants = ["off"] if args.no_lorast else ["off", "lorast"] + ([] if args.no_fused else ["fused"])
        base = "fp16" if spec == "fp16" else ("bnb-nf4" if spec == "bnb-nf4"
                                              else precision_slug(spec.split("ckpt=")[-1]))
        todo = [v for v in variants if (base if v == "off" else f"{base}+lorast"
                                        + ("_fused" if v == "fused" else "")) not in result["rows"]]
        if not todo:
            print(f"[skip] {spec}: all rows present")
            continue
        print(f"\n================ {spec} ================", flush=True)
        t0 = time.perf_counter()
        lm, cfg, info = load_row_model(spec, args)
        res = fix_resolution(lm, args.qwen_side)
        side = args.qwen_side if res["family"] == "qwen2_5_vl" else None
        b1 = prepare(lm, samples, side, 1)
        nv = n_visual(lm, b1)
        if len(nv) != 1:
            raise SystemExit(f"visual tokens vary across images ({nv}): resolution is not fixed")
        sweep = {B: prepare(lm, samples, side, B) for B in sweep_bs}
        wbytes = L.weight_bytes(lm.model)
        print(f"  loaded in {time.perf_counter() - t0:.0f}s: {info.get('method', spec)}; "
              f"{nv[0]} visual tokens; weights {wbytes / 2**30:.2f} GB", flush=True)
        corr_path = None
        if any(v != "off" for v in todo):
            hidden = lm.model.config.get_text_config().hidden_size
            corr_path, why = find_corrector(hidden, args.lorast)
            print(f"  LoRAS-T corrector: {corr_path} ({why})", flush=True)
            if corr_path is None:
                msg = (f"no LoRAS-T corrector for hidden size {hidden}: calibrate one with "
                       f"scripts/12_calibrate_lorast.py --model-id {args.model_id} --dtype "
                       f"{args.dtype} --precision w3a16 --qa-per-image 2 --out "
                       f"runs_4bit/lorast/{name}_w3_rtn_qa.pt, then rerun with --resume")
                print(f"  [warn] {msg}", flush=True)
                if msg not in result["notes"]:
                    result["notes"].append(msg)
                todo = [v for v in todo if v == "off"]
        for v in todo:
            row = base if v == "off" else f"{base}+lorast" + ("_fused" if v == "fused" else "")
            LT.detach_hidden(lm.layers)
            meta = {}
            if v != "off":
                corr, meta = LT.load(corr_path, lm.device, lm.dtype, fused=(v == "fused"))
                LT.attach_hidden(lm.layers, corr)
            print(f"  -- {row}", flush=True)
            t1 = time.perf_counter()
            m = L.measure_row(lm, b1, n_new=args.max_new_tokens, n_warmup=args.n_warmup,
                              repeats=args.repeats, sweep=sweep, sweep_repeats=args.sweep_repeats,
                              device=args.device, energy=energy)
            m.update({"spec": spec, "backend": cfg.row_backend, "lorast_mode": v,
                      "lorast_file": corr_path if v != "off" else None,
                      "lorast_meta": {k: meta.get(k) for k in ("precision", "rank", "params",
                                                               "qa_per_image")} if meta else None,
                      "weights_gb": L.weight_bytes(lm.model) / 2 ** 30,
                      "dtype": str(lm.dtype), "visual_tokens": nv[0], "resolution": res,
                      "load_info": info, "wall_min": (time.perf_counter() - t1) / 60})
            result["rows"][row] = m
            save()
            print(f"  [{row}] TTFT {m['ttft_ms']['median']:.1f} ms  TPOT {m['tpot_ms']['median']:.2f} ms  "
                  f"weights {m['weights_gb']:.2f} GB  peak {m['peak_alloc_gb'] or 0:.2f} GB  "
                  f"({m['wall_min']:.1f} min)", flush=True)
        LT.detach_hidden(lm.layers)
        del lm
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    save()
    print("\n" + out_md.read_text())
    print(f"written to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
