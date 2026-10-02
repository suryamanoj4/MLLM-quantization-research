#!/usr/bin/env python
"""Where on the precision ladder does the model actually break -- and how?

Every rung is scored on the same POPE questions and the same CHAIR images, then
compared to the first rung (normally fp16) with a *paired, image-clustered
bootstrap*, and given a health verdict:

    ok        no significant change vs the reference
    degraded  a significant POPE drop or CHAIR rise, but still fluent -- this is the
              Lexical Fallback regime, and where LoRAS / A-CAB have something to fix
    BROKEN    the model stopped producing usable text (unparseable POPE answers,
              near-empty or repetitive captions). A low CHAIR here means nothing:
              a model that says nothing hallucinates nothing.

Rungs are precision specs: named (fp16, w8a8, w4a8, ...), any w<bits>a<bits>, and
options after a colon:

    w4a5  w4a6                            intermediate activation widths
    w4a4:clip=0.999                       percentile activation clipping
    w4a4:skip=down_proj                   keep the outlier-heavy down_proj in FP16
    w4a8:targets=language+projector       also quantize the vision-language projector
    w4a8:targets=language+projector+vision
    w3a16  w2a16:gs=64                    weight-only

    python scripts/00_precision_ladder.py --coco-root data/coco-mini --out runs/ladder_fine \\
        --ladder "fp16,w4a8,w4a6,w4a5,w4a4,w4a4:clip=0.999,w4a4:skip=down_proj"

Published PTQ methods (phoenix/ptq) are rungs too -- this is the Gate-1 study:

    nf4                                   bitsandbytes NF4 (HF load_in_4bit), simulated
    bnb-nf4                               the same with the real bitsandbytes kernels
    w4a16:method=awq:calib=wiki           weight-only, calibrated on generic text
    w3a16:method=gptq                     ... on image+caption data (calib=mm, default)
    w3a16:method=awq:calib=text           ... on the same captions, no image
    w8a8:method=sq     w4a4:method=rot+gptq

    python scripts/00_precision_ladder.py --coco-root data/coco-mini --preset gate1 \
        --out runs/gate1 --n-pope-images 100 --n-chair-images 100 --resume

Besides F1 and CHAIR, every rung is scored on what separates the two failure modes:
POPE precision / recall / yes-ratio (pooled over splits), caption coverage and length,
and the language model's health with no image at all (blind perplexity). The report
classifies each change as fallback-like (more "yes", more invented objects),
omission-like (more misses, emptier captions), mixed, or none.

--resume skips rungs already in the output file, so a crash or a longer ladder
only runs what is missing.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from phoenix.chair import ChairScorer
from phoenix.data import build_caption_set, check_root, get_pope
from phoenix.evaluate import run_captions, run_pope
from phoenix.model import DEFAULT_MODEL, load_model
from phoenix.probes import blind_captions, blind_fluency
from phoenix.ptq import apply_ptq
from phoenix.quant import QuantConfig, quantize_model, release_fp_weights
from phoenix.stats import (auroc, compare_auroc, compare_chair, compare_mean, compare_pope,
                           compare_ratio, fmt_delta)
from phoenix.utils import human_env, resolve_dtype, save_json, set_seed

# Gate 1: does practical quantization change grounding, and in which direction?
PRESETS = {
    # Calibration matters as much as the method (09_validate_ptq.py, 1 Oct): on WikiText,
    # GPTQ calibrated on image+caption data is *worse* than RTN at W3, the same GPTQ
    # calibrated on generic text closes 35% of RTN's gap. So every calibrated method runs
    # with the toolchain default (calib=wiki); at W3 also with mm and text (same captions,
    # no image) -- the calibration-data experiment.
    "gate1": [
        "fp16",
        # weight-only 4-bit: what people actually deploy
        "nf4", "w4a16", "w4a16:method=awq:calib=wiki", "w4a16:method=gptq:calib=wiki",
        # weight-only 3-bit: where methods and calibration data start to matter
        "w3a16",
        "w3a16:method=awq:calib=wiki", "w3a16:method=awq", "w3a16:method=awq:calib=text",
        "w3a16:method=gptq:calib=wiki", "w3a16:method=gptq", "w3a16:method=gptq:calib=text",
        # weight + activation (SmoothQuant keeps mm: it needs image-token activation ranges)
        "w8a8", "w8a8:method=sq", "w4a8:method=sq",
        "w4a4:method=rot", "w4a4:method=rot+gptq:calib=wiki",
    ],
    "gate1-weights": ["fp16", "nf4", "w4a16", "w4a16:method=awq:calib=wiki",
                      "w4a16:method=gptq:calib=wiki", "w3a16",
                      "w3a16:method=awq:calib=wiki", "w3a16:method=awq",
                      "w3a16:method=awq:calib=text", "w3a16:method=gptq:calib=wiki",
                      "w3a16:method=gptq", "w3a16:method=gptq:calib=text"],
    "gate1-acts": ["fp16", "w8a8", "w8a8:method=sq", "w4a8:method=sq",
                   "w4a4:method=rot", "w4a4:method=rot+gptq:calib=wiki"],
    # Gate 2: is the omission real, visual, and specific to quantization? Run with
    # --n-chair-images 300. Controls: Gaussian weight noise with RTN's error energy
    # (method=noise), and RTN-W3 weights applied only at image / only at text positions.
    "gate2": [
        "fp16",
        "w4a16:method=gptq:calib=wiki",
        "w3a16", "w3a16:method=gptq:calib=wiki", "w3a16:method=gptq",
        "w3a16:method=awq:calib=wiki",
        "w4a4:method=rot+gptq:calib=wiki",
        "w4a16:method=noise", "w3a16:method=noise",
        "w3a16:wtok=image", "w3a16:wtok=text",
    ],
    "diagnostic": ["fp16", "w4a8", "w4a6", "w4a5", "w4a4", "w4a4:abits_text=8",
                   "w4a4:abits_image=8"],
}
BLIND_PPL_LIMIT = 1.10      # blind PPL more than 10% above the reference = LM damaged

# health thresholds -- deliberately loose: they flag collapse, not mild damage
MAX_UNPARSED = 0.20        # POPE answers that are neither yes nor no
MIN_COVERAGE_FRAC = 0.40   # coverage relative to the reference rung
MIN_LEN_FRAC = 0.40        # caption length relative to the reference rung
MAX_REP4 = 0.30            # 4-gram repetition rate


def health(row: dict, ref: dict | None) -> tuple[str, list[str]]:
    why = []
    unp = max(row.get(f"pope_{s}_unparsed", 0.0) for s in row.get("splits", []) or [""])
    if unp > MAX_UNPARSED:
        why.append(f"{100*unp:.0f}% unparseable POPE answers")
    if row.get("rep4", 0) > MAX_REP4:
        why.append(f"rep4={row['rep4']:.2f} (looping)")
    if ref is not None and ref is not row:
        if row["coverage"] < MIN_COVERAGE_FRAC * ref["coverage"]:
            why.append(f"coverage {row['coverage']:.2f} vs {ref['coverage']:.2f} at reference")
        if row["avg_len"] < MIN_LEN_FRAC * ref["avg_len"]:
            why.append(f"captions {row['avg_len']:.0f} words vs {ref['avg_len']:.0f}")
    return ("BROKEN" if why else "healthy"), why


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--coco-root", required=True)
    p.add_argument("--eval-subset", default="val2014")
    p.add_argument("--ladder", default="fp16,w8a8,w4a16,w4a8,w4a4")
    p.add_argument("--preset", default="", choices=[""] + sorted(PRESETS),
                   help="a named list of rungs (overrides --ladder); gate1 = the PTQ study")
    p.add_argument("--n-blind-captions", type=int, default=40,
                   help="eval-pool captions for the no-image perplexity (0 = skip)")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--quant-targets", default="language")
    p.add_argument("--pope-splits", default="random,popular,adversarial")
    p.add_argument("--n-pope-images", type=int, default=60)
    p.add_argument("--n-chair-images", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ladder-extra", default="",
                   help="rungs appended to --preset / --ladder, e.g. "
                        "'w3a16+lorast=runs/lorast/w3_rtn.pt' (LoRAS-T correctors, "
                        "optionally @scale: ...pt@0.5)")
    p.add_argument("--out", default="runs/ladder")
    p.add_argument("--resume", action="store_true",
                   help="skip rungs already present in <out>/ladder.json")
    p.add_argument("--synonyms", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    check_root(args.coco_root)
    print("[env]", human_env(), flush=True)
    scorer = ChairScorer(args.coco_root, args.eval_subset, args.synonyms)
    splits = [s.strip() for s in args.pope_splits.split(",") if s.strip()]
    if args.preset:
        args.ladder = ",".join(PRESETS[args.preset])
    rungs = [x.strip() for x in args.ladder.split(",") if x.strip()]
    rungs += [x.strip() for x in args.ladder_extra.split(",") if x.strip() and x.strip() not in rungs]
    for r in rungs:                                     # fail on typos before loading
        base, lt = split_rung(r)
        QuantConfig.from_ladder(base)
        if lt and not Path(lt[0]).exists():
            raise SystemExit(f"LoRAS-T file not found: {lt[0]}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ladder.json"

    table: dict = {}
    if args.resume and out_file.exists():
        prev = json.loads(out_file.read_text())
        same = all(prev["config"].get(k) == getattr(args, k.replace("-", "_"))
                   for k in ("n_pope_images", "n_chair_images", "max_new_tokens",
                             "pope_splits", "seed", "group_size", "quant_targets"))
        if same:
            # rows from older ladder formats lack per-item data / blind PPL; re-run those
            need = ["pope", "captions", "per_caption_cov"] + (["blind"] if args.n_blind_captions else [])
            table = {k: v for k, v in prev["table"].items() if all(n in v for n in need)}
            print(f"[resume] reusing {len(table)} rung(s): {list(table)}")
        elif args.ladder_extra:
            diff = {k: (prev["config"].get(k), getattr(args, k)) for k in
                    ("n_pope_images", "n_chair_images", "max_new_tokens", "pope_splits", "seed",
                     "group_size", "quant_targets") if prev["config"].get(k) != getattr(args, k)}
            raise SystemExit(f"[resume] --ladder-extra must match the saved ladder's settings "
                             f"(saved, given): {diff}")
        else:
            print("[resume] settings differ from the saved ladder; starting fresh")

    tq = tuple(t.strip() for t in args.quant_targets.split(",") if t.strip())

    def save(complete: bool):
        save_json({"config": vars(args), "table": table, "complete": complete}, out_file)

    for prec in rungs:
        if prec in table:
            print(f"\n[skip] {prec} (already in {out_file})")
            continue
        print(f"\n================ {prec} ================", flush=True)
        t_rung = time.perf_counter()
        base_spec, lorast = split_rung(prec)
        cfg = QuantConfig.from_ladder(base_spec, group_size=args.group_size, targets=tq)
        if cfg.backend == "bnb":
            lm = load_model(args.model_id, dtype=resolve_dtype(args.dtype), device=args.device,
                            bnb_4bit=tuple(cfg.targets))
            ptq_info = {"method": "bitsandbytes nf4 (real kernels)"}
        else:
            lm = load_model(args.model_id, dtype=resolve_dtype(args.dtype), device=args.device)
            ptq_info = apply_ptq(lm, cfg, coco_root=args.coco_root, seed=args.seed)
            quantize_model(lm.model, cfg)
            release_fp_weights(lm.model)
        if lorast:
            from phoenix import loras_text as LT
            corr, meta = LT.load(lorast[0], lm.device, lm.dtype)
            if meta.get("precision") != base_spec:
                print(f"  [warn] correctors were fitted for {meta.get('precision')}, "
                      f"not {base_spec}")
            for c in corr.values():
                c.scale = lorast[1]
            LT.attach_hidden(lm.layers, corr)
            ptq_info = {**ptq_info, "lorast": {"path": lorast[0], "scale": lorast[1],
                                               "rank": meta.get("rank"), "params": meta.get("params")}}
            print(f"  LoRAS-T: {len(corr)} correctors (rank {meta.get('rank')}, "
                  f"scale {lorast[1]}) from {lorast[0]}", flush=True)

        row = {"spec": prec, "quant": {k: v for k, v in cfg.__dict__.items()},
               "ptq": ptq_info, "splits": splits, "pope": {}}
        for split in splits:
            samples = get_pope(args.coco_root, args.eval_subset, split,
                               n_images=args.n_pope_images, seed=args.seed, verbose=False)
            m = run_pope(lm, samples, None, batch_size=8, desc=f"{prec} pope/{split}")
            row[f"pope_{split}_f1"] = m["f1"]
            row[f"pope_{split}_acc"] = m["accuracy"]
            row[f"pope_{split}_yes"] = m["yes_ratio"]
            row[f"pope_{split}_unparsed"] = m["unparsed"]
            row[f"pope_{split}_precision"] = m["precision"]
            row[f"pope_{split}_recall"] = m["recall"]
            row[f"pope_{split}_auroc"] = auroc(m["p_yes"], m["labels"]) if m.get("p_yes") \
                else float("nan")
            row["pope"][split] = {"preds": m["preds"], "labels": m["labels"],
                                  "image_ids": m["image_ids"], "p_yes": m.get("p_yes", []),
                                  "examples": m["samples"][:4]}
            print(f"  POPE/{split:12s} F1={m['f1']:.4f} acc={m['accuracy']:.4f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"AUROC={row[f'pope_{split}_auroc']:.3f} "
                  f"yes={m['yes_ratio']:.3f} unparsed={m['unparsed']:.3f}", flush=True)

        cap = build_caption_set(args.coco_root, args.eval_subset,
                                n_images=args.n_chair_images, seed=args.seed)
        r = run_captions(lm, cap, None, batch_size=4,
                         max_new_tokens=args.max_new_tokens, desc=f"{prec} captions")
        sc = scorer.score(r["records"])
        row.update({"CHAIR_s": sc["CHAIR_s"], "CHAIR_i": sc["CHAIR_i"],
                    "coverage": scorer.coverage(r["records"]), **r["fluency"]})
        row["captions"] = r["records"]
        row["per_caption"] = sc["per_caption"]
        row["per_caption_cov"] = scorer.per_caption_coverage(r["records"])
        ex = r["records"][0]["caption"].replace("\n", " ") if r["records"] else ""
        print(f"  CHAIR_s={row['CHAIR_s']:.4f} CHAIR_i={row['CHAIR_i']:.4f} "
              f"coverage={row['coverage']:.4f} len={row['avg_len']:.1f} rep4={row['rep4']:.3f}")
        print(f"  example caption: {ex[:170]}{'...' if len(ex) > 170 else ''}")
        if args.n_blind_captions:
            b = blind_fluency(lm, blind_captions(args.coco_root, args.n_blind_captions))
            row["blind"] = {"ppl": b["blind_ppl"], **b["fluency"], "example": b["answers"][0]}
            print(f"  no image: caption PPL {b['blind_ppl']:.2f}  len {b['fluency']['avg_len']:.1f}"
                  f"  rep4 {b['fluency']['rep4']:.3f}", flush=True)
        row["wall_min"] = (time.perf_counter() - t_rung) / 60
        table[prec] = row
        save(False)
        print(f"  [{prec}] done in {row['wall_min']:.1f} min -> {out_file}", flush=True)

        del lm
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    save(True)
    report(table, rungs, splits, args.n_boot, out_dir)


def split_rung(name: str):
    """'w3a16+lorast=path.pt@0.5' -> ('w3a16', ('path.pt', 0.5)); plain specs -> (name, None)."""
    base, sep, rest = name.partition("+lorast=")
    if not sep:
        return name, None
    path, _, scale = rest.partition("@")
    return base, (path, float(scale) if scale else 1.0)


def _pooled(row, splits, key="preds"):
    preds, labels, ids = [], [], []
    for sp in splits:
        if sp in row["pope"]:
            d = row["pope"][sp]
            preds += d.get(key, [])
            labels += d["labels"]
            ids += d["image_ids"]
    return preds, labels, ids


def compare_rows(row, ref, splits, n_boot) -> dict:
    """Paired, image-clustered deltas of every metric (row minus reference)."""
    cells = {}
    for split in ("adversarial", "random"):
        if split in row["pope"] and split in ref["pope"]:
            a, b = row["pope"][split], ref["pope"][split]
            if a["image_ids"] == b["image_ids"]:
                cells[split] = compare_pope(a["preds"], b["preds"], a["labels"],
                                            a["image_ids"], "f1", n_boot)
    pa, la, ia = _pooled(row, splits)
    pb, lb, ib = _pooled(ref, splits)
    if ia and ia == ib:
        for metric in ("f1", "precision", "recall", "yes"):
            cells[f"pope_{metric}"] = compare_pope(pa, pb, la, ia, metric, n_boot)
        sa, _, _ = _pooled(row, splits, "p_yes")
        sb, _, _ = _pooled(ref, splits, "p_yes")
        if sa and len(sa) == len(sb) == len(la):
            cells["pope_auroc"] = compare_auroc(sa, sb, la, ia, min(n_boot, 1000))
    ids_a = [c["image_id"] for c in row["captions"]]
    ids_b = [c["image_id"] for c in ref["captions"]]
    if ids_a == ids_b:
        cells["chair_i"] = compare_chair(row["per_caption"], ref["per_caption"],
                                         ids_a, "CHAIR_i", n_boot)
        if "per_caption_cov" in row and "per_caption_cov" in ref:
            cells["coverage"] = compare_ratio(row["per_caption_cov"], ref["per_caption_cov"],
                                              ids_a, n_boot)
        la_ = [len(c["caption"].split()) for c in row["captions"]]
        lb_ = [len(c["caption"].split()) for c in ref["captions"]]
        cells["length"] = compare_mean(la_, lb_, ids_a, n_boot)
    if "blind" in row and "blind" in ref:
        cells["blind_ppl_ratio"] = row["blind"]["ppl"] / max(ref["blind"]["ppl"], 1e-9)
    return cells


def direction(cells: dict) -> str:
    """fallback | omission | mixed | none, from significant changes only.

    fallback = says yes to absent objects more (yes-ratio up and precision down) or
               invents more objects in captions without saying less (CHAIR_i up,
               coverage not down)
    omission = misses present objects more without a yes-bias (recall down, yes-ratio
               not up) or mentions fewer present objects without inventing more
               (coverage down, CHAIR_i not up)
    """
    up = lambda k: k in cells and cells[k]["significant"] and cells[k]["delta"] > 0
    down = lambda k: k in cells and cells[k]["significant"] and cells[k]["delta"] < 0
    fb = (up("pope_yes") and down("pope_precision")) or (up("chair_i") and not down("coverage"))
    om = (down("pope_recall") and not up("pope_yes")) or (down("coverage") and not up("chair_i"))
    return "mixed" if fb and om else "fallback" if fb else "omission" if om else "none"


def report(table: dict, rungs: list, splits: list, n_boot: int, out_dir: Path):
    order = [r for r in rungs if r in table] + [r for r in table if r not in rungs]
    ref_name = order[0]
    ref = table[ref_name]
    print("\n" + "=" * 141)
    print(f"paired comparison vs '{ref_name}' (95% CI from an image-clustered bootstrap, "
          f"* = CI excludes 0; POPE pooled over {len(splits)} splits)")
    hdr = (f"{'rung':<34}{'dF1':>9}{'dAUROC':>9}{'dPrec':>9}{'dRecall':>9}{'dYes':>9}{'dCHAIR_i':>10}"
           f"{'dCover':>9}{'dLen':>8}{'blindPPL':>10}  direction  verdict")
    print(hdr)
    summary = {}
    for name in order:
        row = table[name]
        status, why = health(row, ref)
        cells = compare_rows(row, ref, splits, n_boot) if name != ref_name else {}
        sig_bad = lambda k, neg=True: (k in cells and cells[k]["significant"]
                                       and (cells[k]["delta"] < 0) == neg)
        dirn = direction(cells) if status != "BROKEN" else "n/a"
        lm_hurt = cells.get("blind_ppl_ratio", 1.0) > BLIND_PPL_LIMIT
        if status == "BROKEN":
            verdict = "BROKEN: " + "; ".join(why)
        elif (sig_bad("adversarial") or sig_bad("random") or sig_bad("pope_f1")
              or sig_bad("chair_i", neg=False) or dirn != "none"):
            verdict = "degraded" + (" (LM damaged too)" if lm_hurt else
                                    " (LM intact: grounding-specific)" if "blind_ppl_ratio" in cells
                                    else "")
        else:
            verdict = "reference" if name == ref_name else (
                "ok (no significant change)" + (" but LM damaged" if lm_hurt else ""))

        def c(k, w):
            if k not in cells:
                return f"{'-':>{w}}"
            r = cells[k]
            return f"{r['delta']:+.3f}{'*' if r['significant'] else ' '}".rjust(w)
        bp = (f"{cells['blind_ppl_ratio']:.2f}x" if "blind_ppl_ratio" in cells
              else (f"{row['blind']['ppl']:.1f}" if "blind" in row else "-"))
        print(f"{name:<34}{c('pope_f1', 9)}{c('pope_auroc', 9)}{c('pope_precision', 9)}{c('pope_recall', 9)}"
              f"{c('pope_yes', 9)}{c('chair_i', 10)}{c('coverage', 9)}{c('length', 8)}"
              f"{bp:>10}  {dirn:<9}  {verdict}")
        summary[name] = {"verdict": verdict, "direction": dirn, "health_reasons": why,
                         "lm_damaged": lm_hurt, "vs_reference": cells,
                         "ptq": row.get("ptq", {})}

    n_cmp = max(len(order) - 1, 0) * 8
    print(f"\nCIs are unadjusted: across {n_cmp} comparisons expect ~{0.05 * n_cmp:.0f} "
          "false stars. Trust a direction that repeats across methods/bit widths, and "
          "confirm any single star at a larger n before building on it.")
    candidates = [n for n, s in summary.items() if s["verdict"].startswith("degraded")]
    specific = [n for n in candidates if not summary[n]["lm_damaged"]]
    print()
    if candidates:
        print("Degraded-but-fluent rungs:")
        for n in candidates:
            print(f"    {n:<34} direction={summary[n]['direction']:<9} "
                  f"{'grounding-specific' if n in specific else 'LM damaged too'}")
        if specific:
            print("Gate 1 passes for the grounding-specific ones: confirm them at a larger n, "
                  "then run attribution (abits_image/abits_text arms) on them.")
    else:
        broken = [n for n, s in summary.items() if s["verdict"].startswith("BROKEN")]
        healthy = [n for n in order if n not in broken and n != ref_name]
        print("No degraded-but-fluent rung found.")
        if broken and healthy:
            print(f"  The model is fine at {healthy[-1]} and broken at {broken[0]}: "
                  "add rungs between them.")
    save_json({"reference": ref_name, "summary": summary, "candidates": candidates,
               "grounding_specific": specific}, out_dir / "ladder_report.json")
    print(f"\nwritten to {out_dir/'ladder.json'} and {out_dir/'ladder_report.json'}")


if __name__ == "__main__":
    main()
