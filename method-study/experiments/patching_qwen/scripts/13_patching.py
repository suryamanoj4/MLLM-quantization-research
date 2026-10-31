#!/usr/bin/env python
"""Cross-precision causal patching: locate where quantization loses grounding.

For each eval image: FP16 greedy caption -> teacher-force it through the quantized
model -> restore FP16 at one site at a time and measure how much of the log-prob
lost on *object* tokens (vs. all other tokens) comes back. See phoenix/patching.py.

    # LLaVA-1.5-7B, W3 RTN (the Study 2 headline rung), 100 captions, ~1 h on an A6000
    python scripts/13_patching.py --coco-root $COCO --precision w3a16 --n 100

    # Qwen2.5-VL-7B from a local snapshot (or set HF_HUB_CACHE to the shared cache)
    python scripts/13_patching.py --coco-root $COCO --precision w3a16 --n 100 \\
        --model-id /path/to/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<hash>

Resumable: per-caption statistics are appended to <out>/captions.jsonl; rerunning
skips images already there. `--summarize-only` recomputes tables / figure from it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from phoenix import patching as P  # noqa: E402
from phoenix.utils import resolve_dtype, save_json, set_seed  # noqa: E402


def parse_layers(spec: str, n: int) -> list[int]:
    if spec in ("all", ""):
        return list(range(n))
    if ":" in spec:                                   # start:stop:step
        a, b, *c = (int(x) if x else None for x in spec.split(":"))
        return list(range(n))[slice(a, b, c[0] if c else None)]
    return [int(x) for x in spec.split(",")]


def get_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--coco-root", required=True)
    p.add_argument("--precision", default="w3a16", help="quantized side (any ladder spec)")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n", type=int, default=100, help="eval captions")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--layers", default="all", help="all | 0:32:2 | 0,4,8")
    p.add_argument("--modes", default="act,w", help="act (activation) and/or w (weights)")
    p.add_argument("--batch", type=int, default=8, help="interventions per decoder forward")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def caption_one(lm, sample, scorer, max_new_tokens):
    from phoenix.chair import DOUBLE_WORDS
    from phoenix.data import open_image
    from phoenix.model import CAPTION_PROMPT, prepare_batch
    from phoenix.quant import quant_mode
    img = open_image(sample.image_path)
    batch = prepare_batch(lm, [img], [CAPTION_PROMPT])
    Pn = batch["input_ids"].shape[1]
    with quant_mode(lm.model, False):
        out = lm.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                num_beams=1, temperature=None, top_p=None, top_k=None)
    gen = out[0, Pn:].tolist()
    eos = set(lm.eos_ids)
    pad = lm.processor.tokenizer.pad_token_id
    cut = next((i for i, t in enumerate(gen) if t in eos or t == pad), len(gen))
    gen = gen[:cut]
    if len(gen) < 4:
        return None
    tags, text = P.tag_tokens(lm.processor.tokenizer, gen, scorer.gt_objects(sample.image_id),
                              scorer.mapping, DOUBLE_WORDS)
    ids = torch.cat([batch["input_ids"], torch.tensor([gen], device=lm.device)], dim=1)
    inputs = dict(batch, input_ids=ids, attention_mask=torch.ones_like(ids))
    return inputs, Pn, tags, text


def tables(summary: dict, gaps: dict, layers: list[int], modes: list[str]) -> str:
    lines = []
    g = gaps
    lines.append("log-prob cost of quantization per caption token (FP16 caption, teacher-forced)")
    for c in P.CLASSES:
        pt, lo, hi = g[c]["gap_per_token"]
        fp, flo, fhi = g[c]["flip_rate"]
        lines.append(f"  {c:6s} n={g[c]['n_tokens']:6d}  gap {pt:6.3f} [{lo:.3f},{hi:.3f}]  "
                     f"top-1 flips {fp:.3f} [{flo:.3f},{fhi:.3f}]")
    pt, lo, hi = g["obj_minus_other_gap"]
    lines.append(f"  obj - other gap: {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")

    def row(prefix, label):
        vals = []
        for l in layers:
            r = summary.get(f"{prefix}:{l}")
            vals.append("   .  " if r is None else f"{r['obj'][0]:6.2f}")
        lines.append(f"  {label:18s}" + "".join(vals))

    lines.append("\nR_obj by layer (fraction of the object-token loss restored)")
    lines.append("  " + " " * 18 + "".join(f"{l:6d}" for l in layers))
    if "act" in modes:
        for site in P.SITES:
            for ps in ("text", "image") + (("all",) if site == "resid" else ()):
                row(f"act:{site}:{ps}", f"act {site}@{ps}")
    if "w" in modes:
        for kind in ("restore", "damage"):
            for comp in ("attn", "mlp"):
                row(f"w:{kind}:{comp}", f"w {kind} {comp}" + (" (D)" if kind == "damage" else ""))
    return "\n".join(lines)


def plot(summary: dict, layers: list[int], modes: list[str], path: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[patching] matplotlib missing; skipping the figure")
        return
    rows = []
    if "act" in modes:
        rows += [(f"act:{s}:{p}", f"{s} @ {p}") for s in ("attn", "mlp")
                 for p in ("text", "image")]
        rows += [("act:resid:text", "resid @ text"), ("act:resid:image", "resid @ image")]
    if "w" in modes:
        rows += [("w:restore:attn", "W-restore attn"), ("w:restore:mlp", "W-restore mlp")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * len(rows) + 1.2), sharey=True)
    for ax, cls in zip(axes, ("obj", "other")):
        M = [[(summary.get(f"{k}:{l}") or {}).get(cls, [float("nan")])[0] for l in layers]
             for k, _ in rows]
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.05, max(
            max(v for v in r if v == v) if any(v == v for v in r) else 0 for r in M)))
        ax.set_xticks(range(0, len(layers), max(1, len(layers) // 8)))
        ax.set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers) // 8))])
        ax.set_xlabel("layer")
        ax.set_title(f"{cls} tokens")
        fig.colorbar(im, ax=ax, fraction=0.04)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[1] for r in rows], fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"[patching] figure -> {path}")


# --------------------------------------------------------------------------- #
def main():
    a = get_args()
    set_seed(a.seed)
    from phoenix.model import detect_family
    from phoenix.quant import precision_slug
    fam = detect_family(a.model_id)
    out = Path(a.out or f"runs/patching/{fam}_{precision_slug(a.precision)}")
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "captions.jsonl"
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    done = {}
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["image_id"]] = r

    if not a.summarize_only:
        from phoenix.chair import ChairScorer
        from phoenix.data import build_caption_set
        from phoenix.ptq import load_quantized
        t0 = time.perf_counter()
        lm, cfg, info = load_quantized(a.model_id, a.precision, a.coco_root,
                                       dtype=resolve_dtype(a.dtype), device=a.device,
                                       keep_fp=True, seed=a.seed, group_size=a.group_size)
        print(f"[patching] {fam} {a.precision} ready ({time.perf_counter() - t0:.0f}s), "
              f"{lm.n_layers} layers, dtype {lm.dtype}", flush=True)
        layers = parse_layers(a.layers, lm.n_layers)
        cfgs = []
        if "act" in modes:
            cfgs += P.act_configs(lm.n_layers, layers)
        if "w" in modes:
            cfgs += P.weight_configs(lm.n_layers, layers)
        sanity = {"mode": "act", "site": "resid", "pos": "all", "layer": lm.n_layers - 1}
        if P.config_key(sanity) not in {P.config_key(c) for c in cfgs}:
            cfgs.append(sanity)
        sl = sorted({(c["site"], c["layer"]) for c in cfgs if c["mode"] == "act"})
        print(f"[patching] {len(cfgs)} interventions per caption "
              f"({sum(c['mode'] == 'act' for c in cfgs)} activation, "
              f"{sum(c['mode'] == 'w' for c in cfgs)} weight)", flush=True)
        scorer = ChairScorer(a.coco_root, "val2014")
        samples = build_caption_set(a.coco_root, "val2014", n_images=a.n, seed=a.seed)
        t1, k = time.perf_counter(), 0
        with open(jl, "a") as fh:
            for s in samples:
                if s.image_id in done:
                    continue
                got = caption_one(lm, s, scorer, a.max_new_tokens)
                if got is None:
                    continue
                inputs, Pn, tags, text = got
                prep = P.prepare(lm, inputs, Pn, tags, sl)
                res = P.run_caption(lm, prep, cfgs, batch=a.batch)
                sums = P.caption_sums(prep, res)
                s_key = P.config_key(sanity)
                gap = float((prep.lp_fp - prep.lp_q).sum())
                san = float((res[s_key]["lp"] - prep.lp_q).sum()) / gap if abs(gap) > 1e-6 else 1.0
                rec = {"image_id": s.image_id, "caption": text, "n_tok": len(tags),
                       "tags": tags, "sanity_R": san, **sums}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[s.image_id] = rec
                del prep, res
                k += 1
                el = time.perf_counter() - t1
                left = sum(1 for x in samples if x.image_id not in done)
                print(f"[patching] {len(done)}/{len(samples)}  {len(tags)} tok "
                      f"({tags.count('obj')} obj)  sanity R={san:.3f}  "
                      f"{el / k:.0f}s/cap, ETA {el / k * left / 60:.0f} min", flush=True)
                if abs(san - 1) > 0.02:
                    print("[patching] WARNING: patching the last residual stream should "
                          "restore everything (R=1); hooks are not taking effect", flush=True)
        n_layers = lm.n_layers
    else:
        any_r = next(iter(done.values()))
        n_layers = 1 + max(int(k.rsplit(":", 1)[1]) for k in any_r["cfg"])
        layers = parse_layers(a.layers, n_layers)

    per = list(done.values())
    if not per:
        print("[patching] nothing to summarize")
        return
    keys = sorted(per[0]["cfg"])
    summary = P.summarize(per, keys, n_boot=a.n_boot, seed=a.seed)
    gaps = P.gap_summary(per, n_boot=a.n_boot, seed=a.seed)
    txt = tables(summary, gaps, layers, modes)
    print("\n" + txt)
    save_json({"config": vars(a), "family": fam, "n_captions": len(per),
               "gaps": gaps, "summary": summary,
               "sanity_R_mean": sum(r["sanity_R"] for r in per) / len(per)},
              out / "patching_report.json")
    (out / "patching_tables.txt").write_text(txt + "\n")
    plot(summary, layers, modes, out / "patching_heatmap.pdf",
         f"{fam} {a.precision}: fraction of quantization's log-prob loss restored")
    print(f"\nwritten to {out}/patching_report.json, patching_tables.txt, patching_heatmap.pdf")


if __name__ == "__main__":
    main()
