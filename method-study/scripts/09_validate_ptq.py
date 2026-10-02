#!/usr/bin/env python
"""Are our AWQ / GPTQ / SmoothQuant / rotation implementations faithful?

Reviewers will ask. The standard answer is WikiText-2 perplexity (2048-token windows of
the test split), the number every PTQ paper reports. Two ways to run it:

  1. On LLaVA's own language model (no extra download). There are no published numbers
     for it, so the check is *relative*: at the same bit width, GPTQ and AWQ must close
     most of RTN's perplexity gap to FP16, rotation must rescue W4A4, SmoothQuant W8A8
     must sit near FP16.

        python scripts/09_validate_ptq.py --coco-root data/coco-mini

  2. On Llama-2-7B (13 GB download, gated on the hub) with WikiText calibration, which
     can be compared with the published tables.

        python scripts/09_validate_ptq.py --model-id meta-llama/Llama-2-7b-hf --calib wiki

Needs `pip install pyarrow` (WikiText is read from the hub's parquet files).
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

from phoenix.model import DEFAULT_MODEL, LoadedModel, load_model
from phoenix.ptq import apply_ptq
from phoenix.ptq.calib import wikitext
from phoenix.quant import QuantConfig, quantize_model, release_fp_weights
from phoenix.utils import human_env, resolve_dtype, save_json, set_seed

DEFAULT_SPECS = ("fp16,w4a16,w4a16:method=awq:calib=wiki,w4a16:method=gptq:calib=wiki,"
                 "w3a16,w3a16:method=awq:calib=wiki,w3a16:method=gptq:calib=wiki,nf4,"
                 "w8a8,w8a8:method=sq,w4a4,w4a4:method=rot,w4a4:method=rot+gptq")

# Llama-2-7B, WikiText-2, seqlen 2048, group 128, as reported in Table 4 of the AWQ paper
# (Lin et al., MLSys 2024; arXiv:2306.00978). Exact agreement is not expected: the
# calibration data (AWQ: Pile, GPTQ: C4) and our min/max grid convention differ.
# Note how little GPTQ closes at W3 there (19% of RTN's gap; on LLaMA-1-7B it is worse
# than RTN without act-order), so the thresholds below are deliberately modest.
LLAMA2_7B_REF = {"fp16": 5.47, "w4a16": 5.73, "w4a16:method=gptq": 5.69,
                 "w4a16:method=awq": 5.60, "w3a16": 6.66, "w3a16:method=gptq": 6.43,
                 "w3a16:method=awq": 6.24}


def load_any(model_id, dtype, device):
    """LLaVA through phoenix.model; any other causal LM directly."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    if getattr(cfg, "model_type", "") == "llava":
        return load_model(model_id, dtype=dtype, device=device), True
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    m.eval().requires_grad_(False).to(device)
    return LoadedModel(model=m, processor=tok, image_token_id=-1,
                       device=torch.device(device), dtype=dtype), False


@torch.no_grad()
def wiki_ppl(lm, ids: torch.Tensor, seqlen: int, max_windows: int) -> float:
    n = min(ids.numel() // seqlen, max_windows)
    nll = 0.0
    for i in range(n):
        x = ids[i * seqlen:(i + 1) * seqlen][None].to(lm.device)
        logits = lm.model(input_ids=x, use_cache=False).logits[:, :-1].float()
        nll += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum"))
    return math.exp(nll / (n * (seqlen - 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--coco-root", default=None, help="needed for calib=mm/text")
    p.add_argument("--specs", default=DEFAULT_SPECS)
    p.add_argument("--calib", default=None,
                   help="default calibration for specs that don't set one "
                        "(mm for LLaVA, wiki otherwise)")
    p.add_argument("--n-calib", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=40,
                   help="2048-token test windows (the full test split is ~160)")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/validate_ptq")
    a = p.parse_args()

    set_seed(a.seed)
    print("[env]", human_env(), flush=True)
    specs = [s.strip() for s in a.specs.split(",") if s.strip()]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file = out_dir / "validate.json"
    results = json.loads(res_file.read_text()) if res_file.exists() else {}
    if results.get("_model") not in (None, a.model_id):
        results = {}
    results["_model"] = a.model_id
    test_text = wikitext("test")
    ids = None

    for spec in specs:
        if spec in results:
            print(f"[skip] {spec}: {results[spec]['ppl']:.3f}")
            continue
        t0 = time.perf_counter()
        lm, is_llava = load_any(a.model_id, resolve_dtype(a.dtype), a.device)
        calib = a.calib or ("mm" if is_llava else "wiki")
        cfg = QuantConfig.from_ladder(spec, calib=calib, n_calib=a.n_calib)
        if cfg.calib in ("mm", "text") and not is_llava:
            raise SystemExit(f"{spec}: calib={cfg.calib} needs a LLaVA model; use --calib wiki")
        if cfg.backend == "bnb":
            print(f"[skip] {spec}: real-kernel rungs are validated in the ladder, not here")
            continue
        if ids is None:
            tok = getattr(lm.processor, "tokenizer", lm.processor)
            ids = tok(test_text, return_tensors="pt")["input_ids"][0]
        print(f"\n================ {spec} (calib={cfg.calib if cfg.method != 'rtn' else '-'}) "
              "================", flush=True)
        info = apply_ptq(lm, cfg, coco_root=a.coco_root, seed=a.seed)
        quantize_model(lm.model, cfg)
        release_fp_weights(lm.model)
        ppl = wiki_ppl(lm, ids, a.seqlen, a.max_windows)
        results[spec] = {"ppl": ppl, "ptq": info, "minutes": (time.perf_counter() - t0) / 60}
        print(f"  WikiText-2 PPL {ppl:.3f}   ({results[spec]['minutes']:.1f} min)", flush=True)
        save_json(results, res_file)
        del lm
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    report(results, a.model_id)


def report(results: dict, model_id: str):
    r = {k: v["ppl"] for k, v in results.items() if not k.startswith("_")}
    fp = r.get("fp16")
    ref = LLAMA2_7B_REF if "llama-2-7b" in model_id.lower() else None
    print("\n" + "=" * 78)
    print(f"{'spec':<28}{'PPL':>9}{'gap to fp16':>13}" + (f"{'published':>11}" if ref else ""))
    for k, v in r.items():
        gap = f"{v - fp:+.3f}" if fp else "-"
        pub = f"{ref[k]:.2f}" if ref and k in ref else ""
        print(f"{k:<28}{v:>9.3f}{gap:>13}" + (f"{pub:>11}" if ref else ""))
    if not fp:
        return
    print("\nchecks (relative; these must hold whatever the model):")

    def closes(method, base, need):
        if method in r and base in r and r[base] - fp > 1e-6:
            frac = 1 - (r[method] - fp) / (r[base] - fp)
            ok = frac >= need
            print(f"  [{'ok' if ok else '??'}] {method:<24} closes {100 * frac:5.1f}% of "
                  f"{base}'s gap (expect >= {100 * need:.0f}%)")
    # published Llama-2-7B fractions: W3 AWQ 35%, GPTQ 19%; W4 AWQ 50%, GPTQ 15%
    need = {("w3a16", "awq"): 0.25, ("w3a16", "gptq"): 0.10,
            ("w4a16", "awq"): 0.20, ("w4a16", "gptq"): 0.0}
    # the thresholds are for generic-text calibration (what the published numbers use);
    # caption / image calibration is a different experiment, reported but not checked
    for (b, m), n in need.items():
        wiki = f"{b}:method={m}:calib=wiki"
        closes(wiki if wiki in r else f"{b}:method={m}", b, n)
    closes("w4a4:method=rot", "w4a4", 0.5)
    closes("w4a4:method=rot+gptq", "w4a4", 0.5)
    if "w8a8:method=sq" in r:
        g = r["w8a8:method=sq"] - fp
        print(f"  [{'ok' if g < 0.2 else '??'}] w8a8:method=sq within 0.2 PPL of fp16 "
              f"({g:+.3f})")
    print("  '??' is not automatically a bug -- but explain it before using that method's rungs.")
    if any(k.endswith(":calib=wiki") for k in r) and any("method=" in k and "calib=" not in k
                                                         for k in r):
        print("  (image-calibrated rows are expected to lose on WikiText: they were fitted to a "
              "different input distribution)")


if __name__ == "__main__":
    main()
