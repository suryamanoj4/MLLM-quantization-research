#!/usr/bin/env python
"""Visual-token attention under quantization (H2/H4), for any supported VLM.

The method-study counterpart of the evidence study's AttentionTracker
(evidence-study/src/experiments/experiment/decoding.py), which only runs on LLaVA.
For every generated token, in every layer and head, it takes the new token's
attention row and records

  mass     share of attention on the image-token positions (mean over heads, layers)
  entropy  entropy (bits) of that attention renormalised over the image tokens
  drift    how often the most-attended image token changes between consecutive steps

Per item these are averaged over the generated tokens (POPE answers and captions);
captions also keep the per-step mass for the first-10% -> last-10% decay curve.
Every rung sees the same items, so changes vs FP16 are paired:
(sum rung - sum FP16) / sum FP16, with an image-cluster bootstrap CI.

    python scripts/13_attention_probe.py --coco-root data/coco-mini \\
        --model-id Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 \\
        --precisions fp16,w8a8,w4a16:method=gptq:calib=wiki,w4a8 \\
        --out runs_qwen/attention --resume

Qwen's image-token count varies per image, so the entropy ceiling (log2 n_image)
varies too; the report prints the mean ceiling next to the entropy.
"""
import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from phoenix.acab import _cache_len, _needs_cache_position, _set_attn_impl
from phoenix.data import build_caption_set, check_root, get_pope, open_image
from phoenix.model import DEFAULT_MODEL, load_model, prepare_batch
from phoenix.ptq import apply_ptq
from phoenix.quant import QuantConfig, quantize_model, release_fp_weights
from phoenix.stats import paired_bootstrap
from phoenix.utils import human_env, resolve_dtype, save_json, set_seed


@torch.no_grad()
def tracked_decode(lm, inputs: dict, max_new_tokens: int) -> dict:
    """Greedy decode (batch 1) recording visual attention for each generated token.

    The prefill row belongs to the last prompt token, not a generated one, so as in
    the evidence study only the decode steps are recorded."""
    model, eos = lm.model, lm.processor.tokenizer.eos_token_id
    vis = inputs["input_ids"][0] == lm.image_token_id
    out = model(**inputs, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    attn_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
    mass, ent, argmax = [], [], []
    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1)
        if int(nxt) == eos:
            break
        attn_mask = torch.cat([attn_mask, attn_mask.new_ones(1, 1)], 1)
        kw = {}
        if _needs_cache_position(lm):
            plen = _cache_len(past)
            if plen >= 0:
                kw["cache_position"] = torch.arange(plen, plen + 1, device=lm.device)
        o = model(input_ids=nxt[:, None], attention_mask=attn_mask, past_key_values=past,
                  use_cache=True, output_attentions=True, **kw)
        past, logits = o.past_key_values, o.logits[:, -1, :]
        if not o.attentions or o.attentions[0] is None:
            raise RuntimeError("no attention weights returned; eager attention not active")
        m_l, e_l, a_l = [], [], []
        for A in o.attentions:
            row = A[0, :, -1, :].float()                          # [H, kv]
            cols = F.pad(vis, (0, row.shape[-1] - vis.shape[0]), value=False)
            v = row[:, cols]                                      # [H, n_image]
            s = v.sum(-1)
            p = v / (s[:, None] + 1e-9)
            m_l.append(float(s.mean()))
            e_l.append(float((-(p * torch.log2(p + 1e-12)).sum(-1)).mean()))
            a_l.append(v.argmax(-1).cpu())
        mass.append(sum(m_l) / len(m_l))
        ent.append(sum(e_l) / len(e_l))
        argmax.append(torch.stack(a_l))                           # [L, H]
        del o
    flips = [float((a != b).float().mean()) for a, b in zip(argmax[:-1], argmax[1:])]
    n = len(mass)
    return {"mass": sum(mass) / n if n else float("nan"),
            "entropy": sum(ent) / n if n else float("nan"),
            "drift": sum(flips) / len(flips) if flips else 0.0,
            "n_steps": n, "n_image": int(vis.sum()), "steps": mass}


def deciles(series: list[float], k: int = 10) -> list[float]:
    n = len(series)
    if n < k:
        return []
    e = [int(i * n / k) for i in range(k + 1)]
    return [sum(series[e[i]:e[i + 1]]) / (e[i + 1] - e[i]) for i in range(k)]


def run_rung(spec, args, pope, caps) -> dict:
    cfg = QuantConfig.from_ladder(spec, group_size=args.group_size, targets=("language",))
    if cfg.backend == "bnb":
        raise SystemExit(f"{spec}: real bitsandbytes rungs are not supported here; "
                         "use the simulated 'nf4'")
    lm = load_model(args.model_id, dtype=resolve_dtype(args.dtype), device=args.device)
    info = apply_ptq(lm, cfg, coco_root=args.coco_root, seed=args.seed)
    quantize_model(lm.model, cfg)
    release_fp_weights(lm.model)
    _set_attn_impl(lm.model, "eager")
    rows = {"ptq": info, "pope": [], "chair": []}
    t0 = time.perf_counter()
    for task, items, mnt in (("pope", pope, args.pope_max_new_tokens),
                             ("chair", caps, args.caption_max_new_tokens)):
        for k, s in enumerate(items):
            b = prepare_batch(lm, [open_image(s.image_path)], [s.question])
            r = tracked_decode(lm, b, mnt)
            if task == "pope":
                r.pop("steps")
                r["split"] = s.split
            else:
                r["deciles"] = deciles(r.pop("steps"))
            r["image_id"] = s.image_id
            rows[task].append(r)
            if (k + 1) % 100 == 0 or k + 1 == len(items):
                print(f"  [{task}] {k + 1}/{len(items)}  ({(time.perf_counter() - t0) / 60:.1f} min)",
                      flush=True)
    del lm
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows


def rel_change(a, b, ids, key, n_boot):
    """(sum a - sum b) / sum b over the same items, image-cluster bootstrap.
    Items where either rung generated nothing (no attention row) are left out of both."""
    keep = [not (math.isnan(x[key]) or math.isnan(y[key])) for x, y in zip(a, b)]
    xa = [x[key] if k else 0.0 for x, k in zip(a, keep)]
    xb = [y[key] if k else 0.0 for y, k in zip(b, keep)]
    return paired_bootstrap(lambda idx: sum(xa[i] for i in idx) / max(sum(xb[i] for i in idx), 1e-12) - 1,
                            lambda idx: 0.0, ids, n_boot)


def report(res: dict, n_boot: int):
    fp = res.get("fp16")
    def mean(rows, k):
        v = [r[k] for r in rows if not math.isnan(r[k])]
        return sum(v) / max(len(v), 1)
    print("\n" + "=" * 100)
    print(f"{'rung':<34}{'mass POPE':>10}{'chg vs FP16':>24}{'mass CHAIR':>11}{'chg vs FP16':>24}")
    out = {}
    for spec, r in res.items():
        if spec.startswith("_"):
            continue
        splits = sorted({x["split"] for x in r["pope"]})
        pm = sum(mean([x for x in r["pope"] if x["split"] == sp], "mass") for sp in splits) / len(splits)
        cm = mean(r["chair"], "mass")
        row = {"mass_pope": pm, "mass_chair": cm,
               "entropy_pope": mean(r["pope"], "entropy"), "entropy_chair": mean(r["chair"], "entropy"),
               "drift_pope": mean(r["pope"], "drift"), "drift_chair": mean(r["chair"], "drift"),
               "log2_n_image": mean([{"v": math.log2(max(x["n_image"], 1))} for x in r["chair"]], "v")}
        dec = [x["deciles"] for x in r["chair"] if x["deciles"]]
        if dec:
            row["decile_profile"] = [sum(d[i] for d in dec) / len(dec) for i in range(10)]
        cp = cc = ""
        if fp is not None and spec != "fp16":
            ok = lambda a, b: len(a) == len(b) and all(x["image_id"] == y["image_id"] for x, y in zip(a, b))
            if ok(r["pope"], fp["pope"]) and ok(r["chair"], fp["chair"]):
                row["chg_pope"] = rel_change(r["pope"], fp["pope"], [x["image_id"] for x in r["pope"]], "mass", n_boot)
                row["chg_chair"] = rel_change(r["chair"], fp["chair"], [x["image_id"] for x in r["chair"]], "mass", n_boot)
                for sp in sorted({x["split"] for x in r["pope"]}):
                    a = [x for x in r["pope"] if x["split"] == sp]
                    b = [x for x in fp["pope"] if x["split"] == sp]
                    row[f"chg_pope_{sp}"] = rel_change(a, b, [x["image_id"] for x in a], "mass", n_boot)
                f = lambda d: (f"{100 * d['delta']:+.1f}% [{100 * d['lo']:+.1f},{100 * d['hi']:+.1f}]"
                               + ("*" if d["significant"] else " "))
                cp, cc = f(row["chg_pope"]), f(row["chg_chair"])
        out[spec] = row
        print(f"{spec:<34}{100 * pm:>9.1f}%{cp:>24}{100 * cm:>10.1f}%{cc:>24}")
    print(f"\n{'rung':<34}{'entropy POPE/CHAIR (bits)':>28}{'ceiling':>9}{'drift POPE/CHAIR':>19}"
          f"{'CHAIR first->last 10%':>24}")
    for spec, row in out.items():
        d = row.get("decile_profile")
        dd = f"{d[0]:.3f} -> {d[-1]:.3f} ({d[-1] / d[0]:.2f})" if d else "-"
        print(f"{spec:<34}{row['entropy_pope']:>13.2f} / {row['entropy_chair']:<12.2f}"
              f"{row['log2_n_image']:>9.2f}{row['drift_pope']:>10.3f} / {row['drift_chair']:<6.3f}{dd:>24}")
    print("\nper split (change in POPE image attention vs FP16):")
    for spec, row in out.items():
        ks = sorted(k for k in row if k.startswith("chg_pope_"))
        if ks:
            print(f"  {spec:<32}" + "  ".join(f"{k[9:]} {100 * row[k]['delta']:+.1f}%"
                                              + ("*" if row[k]["significant"] else "") for k in ks))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--coco-root", required=True)
    p.add_argument("--eval-subset", default="val2014")
    p.add_argument("--precisions", default="fp16,w8a8,w4a16:method=gptq:calib=wiki,w4a8")
    p.add_argument("--pope-splits", default="random,popular,adversarial")
    p.add_argument("--n-pope-images", type=int, default=100)
    p.add_argument("--n-chair-images", type=int, default=100)
    p.add_argument("--pope-max-new-tokens", type=int, default=16)
    p.add_argument("--caption-max-new-tokens", type=int, default=256,
                   help="256 as in the LLaVA evidence study")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/attention")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    set_seed(a.seed)
    check_root(a.coco_root)
    print("[env]", human_env(), flush=True)
    specs = [s.strip() for s in a.precisions.split(",") if s.strip()]
    for s in specs:
        QuantConfig.from_ladder(s)                       # fail on typos before loading
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "attention.json"
    keys = ("model_id", "n_pope_images", "n_chair_images", "pope_splits",
            "pope_max_new_tokens", "caption_max_new_tokens", "seed", "group_size")
    res = {"_config": {k: getattr(a, k) for k in keys}}
    if a.resume and out_file.exists():
        prev = json.loads(out_file.read_text())
        if prev.get("_config") == res["_config"]:
            res = prev
            print(f"[resume] reusing {[k for k in res if not k.startswith('_')]}")
        else:
            print("[resume] settings differ from the saved run; starting fresh")

    pope = [s for sp in a.pope_splits.split(",") if sp.strip()
            for s in get_pope(a.coco_root, a.eval_subset, sp.strip(), n_images=a.n_pope_images,
                              seed=a.seed)]
    caps = build_caption_set(a.coco_root, a.eval_subset, n_images=a.n_chair_images, seed=a.seed)
    print(f"[data] {len(pope)} POPE questions, {len(caps)} captions", flush=True)

    for spec in specs:
        if spec in res:
            print(f"[skip] {spec}")
            continue
        print(f"\n================ {spec} ================", flush=True)
        t0 = time.perf_counter()
        res[spec] = run_rung(spec, a, pope, caps)
        res[spec]["minutes"] = (time.perf_counter() - t0) / 60
        print(f"  done in {res[spec]['minutes']:.1f} min", flush=True)
        save_json(res, out_file)

    summary = report(res, a.n_boot)
    save_json(summary, out_dir / "attention_report.json")


if __name__ == "__main__":
    main()
