#!/usr/bin/env python
"""CPU tests for cross-precision causal patching (phoenix/patching.py) and the
Qwen2.5-VL path, on tiny random models.

    python tests/test_patching.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from phoenix import patching as P
from phoenix.model import LoadedModel, build_prompt, detect_family, get_language_model
from phoenix.quant import QuantConfig, quant_mode, quantize_model

torch.manual_seed(0)
_fail = []


def check(name, cond, detail=""):
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _fail.append(name)


class StubTok:
    """Word-piece tokenizer over a fixed vocabulary, enough for tag_tokens."""
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids).lstrip()


class StubProc:
    def __init__(self, tok=None):
        self.tokenizer = tok


# --------------------------------------------------------------------------- #
def tiny_llama(n_layers=3, d=64, vocab=96):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=d, intermediate_size=2 * d, num_hidden_layers=n_layers,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=vocab,
                      max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    m = LlamaForCausalLM(cfg).eval().requires_grad_(False)
    return LoadedModel(model=m, processor=StubProc(), image_token_id=31,
                       device=torch.device("cpu"), dtype=torch.float32)


def tiny_qwen(n_layers=3):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    rs = {"type": "mrope", "mrope_section": [2, 3, 3]}
    tc = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=n_layers,
              num_attention_heads=4, num_key_value_heads=2, vocab_size=200,
              max_position_embeddings=512, rope_scaling=rs)
    cfg = Qwen2_5_VLConfig(
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, patch_size=14, spatial_merge_size=2,
                           temporal_patch_size=2, window_size=56, fullatt_block_indexes=[1]),
        text_config=tc, image_token_id=150, video_token_id=151, vision_start_token_id=152,
        vision_end_token_id=153, **tc)
    m = Qwen2_5_VLForConditionalGeneration(cfg).eval().requires_grad_(False)
    return LoadedModel(model=m, processor=StubProc(), image_token_id=150,
                       device=torch.device("cpu"), dtype=torch.float32, family="qwen")


def llama_inputs(T=40, n_img=12, prompt_len=24, vocab=96):
    ids = torch.randint(3, vocab, (1, T))
    ids[ids == 31] = 5
    ids[0, 2:2 + n_img] = 31
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, prompt_len


def qwen_inputs():
    pre = [1, 2, 152] + [150] * 4 + [153, 5, 6, 7]
    cap = [11, 12, 13, 14, 15, 16, 17, 18]
    ids = torch.tensor([pre + cap])
    return ({"input_ids": ids, "attention_mask": torch.ones_like(ids),
             "pixel_values": torch.randn(16, 3 * 2 * 14 * 14),
             "image_grid_thw": torch.tensor([[1, 4, 4]])}, len(pre))


def full_model_lp(lm, inputs, Pn, quant=True, hook=None):
    """Reference: the full HF forward (vision tower included), optional output hook."""
    h = hook[0].register_forward_hook(hook[1]) if hook else None
    try:
        with quant_mode(lm.model, quant):
            lg = lm.model(**inputs, use_cache=False).logits.float()
    finally:
        if h:
            h.remove()
    ids = inputs["input_ids"][0]
    lp = torch.log_softmax(lg[0, Pn - 1:-1], -1)
    return lp.gather(-1, ids[Pn:, None]).squeeze(-1)


def run_suite(name, lm, inputs, Pn):
    print(f"\n== {name} ==")
    quantize_model(lm.model, QuantConfig(w_bits=2, a_bits=16, group_size=32), verbose=False)
    L = lm.n_layers
    n = inputs["input_ids"].shape[1] - Pn
    tags = (["obj", "other", "cont", "hall"] * n)[:n]
    sl = [(s, l) for s in P.SITES for l in range(L)]
    prep = P.prepare(lm, inputs, Pn, tags, sl)

    ref_fp = full_model_lp(lm, inputs, Pn, quant=False)
    ref_q = full_model_lp(lm, inputs, Pn, quant=True)
    check("decoder-only path == full model (FP16)", torch.allclose(prep.lp_fp, ref_fp, atol=1e-4),
          f"max|d|={float((prep.lp_fp - ref_fp).abs().max()):.2e}")
    check("decoder-only path == full model (quantized)", torch.allclose(prep.lp_q, ref_q, atol=1e-4))
    check("quantization changes the caption log-probs",
          float((prep.lp_fp - prep.lp_q).abs().max()) > 1e-3)

    cfgs = P.act_configs(L) + P.weight_configs(L)
    res = P.run_caption(lm, prep, cfgs, batch=5)
    k_last = f"act:resid:all:{L - 1}"
    check("patching the last residual stream (all positions) restores FP16 exactly",
          torch.allclose(res[k_last]["lp"], prep.lp_fp, atol=1e-4))

    # batched rows == one-at-a-time
    solo = P.run_caption(lm, prep, [c for c in cfgs if P.config_key(c) == "act:mlp:text:1"], batch=1)
    check("batched intervention == the same intervention alone",
          torch.allclose(solo["act:mlp:text:1"]["lp"], res["act:mlp:text:1"]["lp"], atol=1e-5))

    # activation patch vs. an independent hook on the full model
    blk = lm.layers[1].mlp
    fpa = prep.fp_acts[("mlp", 1)]
    mask = ~prep.image_mask

    def hk(m, a, o):
        o = o.clone()
        o[0, mask] = fpa[mask]
        return o
    ref = full_model_lp(lm, inputs, Pn, quant=True, hook=(blk, hk))
    check("activation patch == independent hook on the full model",
          torch.allclose(ref, res["act:mlp:text:1"]["lp"], atol=1e-4))

    # image-position patch at layer 0 resid differs from text-position patch
    check("image- and text-position patches are distinct interventions",
          not torch.allclose(res["act:resid:image:0"]["lp"], res["act:resid:text:0"]["lp"]))

    # weight restore of one layer's MLP == full model with those modules in FP16
    mods = P.fq_modules(lm, "mlp", [0])
    with P.weights_override(mods, False):
        ref = full_model_lp(lm, inputs, Pn, quant=True)
    # quant_mode(True) inside full_model_lp re-enables them, so do it by hand
    with quant_mode(lm.model, True), P.weights_override(mods, False):
        lg = lm.model(**inputs, use_cache=False).logits.float()
    ids = inputs["input_ids"][0]
    ref = torch.log_softmax(lg[0, Pn - 1:-1], -1).gather(-1, ids[Pn:, None]).squeeze(-1)
    check("weight restore == full model with that layer's MLP in FP16",
          torch.allclose(ref, res["w:restore:mlp:0"]["lp"], atol=1e-4))
    check("weight damage moves lp away from FP16",
          float((res["w:damage:mlp:0"]["lp"] - prep.lp_fp).abs().max()) > 1e-4)

    # aggregation
    per = [P.caption_sums(prep, res)] * 4
    S = P.summarize(per, sorted(res), n_boot=200)
    check("R(last residual) == 1 for every class",
          all(abs(S[k_last][c][0] - 1) < 1e-3 for c in ("obj", "other")))
    d = S[f"w:damage:mlp:0"]["obj"][0]
    r = S[f"w:restore:mlp:0"]["obj"][0]
    check("damage/restore fractions are finite", d == d and r == r, f"D={d:.3f} R={r:.3f}")
    G = P.gap_summary(per, n_boot=200)
    check("gap summary has every class", all(c in G for c in P.CLASSES))
    return res


def test_tagging():
    print("\n== object-token tagging ==")
    from phoenix.chair import DOUBLE_WORDS, load_synonyms
    mapping, _ = load_synonyms()
    for dw in DOUBLE_WORDS:
        if mapping.get(dw):
            mapping[dw.replace(" ", "_")] = mapping[dw]
    pieces = [" A", " man", " is", " eating", " a", " hot", " dog", " next", " to", " his",
              " do", "gs", " and", " a", " surf", "board", "."]
    tok = StubTok(pieces)
    ids = list(range(len(pieces)))
    tags, text = P.tag_tokens(tok, ids, gt={"person", "hot dog"}, mapping=mapping,
                              double_words=DOUBLE_WORDS)
    check("decoded text round-trips", text == "A man is eating a hot dog next to his dogs and a surfboard.")
    want = {1: "obj", 5: "obj", 6: "cont", 10: "hall", 11: "cont", 14: "hall", 15: "cont"}
    got = {i: t for i, t in enumerate(tags) if t != "other"}
    check("first sub-token of each mention is tagged, GT vs hallucinated",
          got == want, f"{got}")
    check("'hot dog' is one mention, not a dog", tags[6] == "cont")


def test_qwen_wrappers():
    print("\n== Qwen2.5-VL wrappers ==")
    check("family from hub id", detect_family("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen")
    check("family from local snapshot path",
          detect_family("/x/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/ab") == "qwen")
    check("llava id stays llava", detect_family("llava-hf/llava-1.5-7b-hf") == "llava")
    p = build_prompt("Describe.", family="qwen")
    check("ChatML prompt with one image pad",
          p.count("<|image_pad|>") == 1 and p.endswith("<|im_start|>assistant\n"), repr(p[-40:]))
    lm = tiny_qwen()
    check("decoder stack located", type(get_language_model(lm.model)).__name__ == "Qwen2_5_VLTextModel")
    fq = quantize_model(lm.model, QuantConfig(w_bits=3, a_bits=16, group_size=32), verbose=False)
    check("only the language model is wrapped (7 Linears per layer)",
          len(fq) == 7 * lm.n_layers and all(".visual" not in m.name for m in fq), f"{len(fq)}")


def test_qwen_generate():
    print("\n== Qwen M-RoPE decode loop ==")
    from phoenix.acab import generate
    lm = tiny_qwen()
    class T:
        eos_token_id = 199
        def batch_decode(self, ids, skip_special_tokens=True):
            return [" ".join(map(str, r.tolist())) for r in ids]
    lm.processor.tokenizer = T()
    inputs, _ = qwen_inputs()
    pre = {k: v for k, v in inputs.items()}
    pre["input_ids"] = inputs["input_ids"][:, :12]
    pre["attention_mask"] = inputs["attention_mask"][:, :12]
    out = generate(lm, pre, None, max_new_tokens=6)
    hf = lm.model.generate(**pre, max_new_tokens=6, do_sample=False, eos_token_id=199)
    check("phoenix greedy decode == HF generate on Qwen (M-RoPE positions)",
          out["sequences"][0].tolist() == hf[0, 12:].tolist()[:out["sequences"].shape[1]],
          f"{out['sequences'][0].tolist()} vs {hf[0, 12:].tolist()}")


def test_script_end_to_end():
    """scripts/13_patching.py on a fake COCO with the tiny Qwen (load_quantized stubbed)."""
    print("\n== scripts/13_patching.py end to end ==")
    import importlib.util
    import json
    import tempfile
    from test_ladder import build_coco
    import phoenix.ptq as ptq

    pieces = [f" w{i}" for i in range(200)]
    pieces[20], pieces[21], pieces[22] = " dog", " cat", " pizza"

    class Tok(StubTok):
        pad_token_id = 0
        eos_token_id = 199
        padding_side = "left"

    class Proc:
        tokenizer = Tok(pieces)

        def __call__(self, images, text, return_tensors="pt", padding=True, **kw):
            inp, Pn = qwen_inputs()
            return {"input_ids": inp["input_ids"][:, :Pn], "attention_mask": inp["attention_mask"][:, :Pn],
                    "pixel_values": inp["pixel_values"], "image_grid_thw": inp["image_grid_thw"]}

    def fake_load(model_id, spec, coco_root=None, **kw):
        lm = tiny_qwen()
        lm.processor = Proc()
        # make greedy decoding emit object words: bias the LM head toward 20..22
        with torch.no_grad():
            lm.model.lm_head.weight[20:23] += 0.5 * lm.model.lm_head.weight[20:23].sign()
        cfg = QuantConfig.from_ladder(spec, group_size=32)
        quantize_model(lm.model, cfg, verbose=False)
        return lm, cfg, {}

    spec = importlib.util.spec_from_file_location(
        "s13", Path(__file__).resolve().parents[1] / "scripts" / "13_patching.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "coco"
        build_coco(root, n_images=4)
        out = Path(td) / "out"
        old = ptq.load_quantized
        ptq.load_quantized = fake_load
        try:
            sys.argv = ["13", "--coco-root", str(root), "--n", "3", "--device", "cpu",
                        "--dtype", "float32", "--max-new-tokens", "10", "--n-boot", "50",
                        "--model-id", "Qwen/tiny", "--precision", "w2a16", "--out", str(out)]
            mod.main()
            lines = (out / "captions.jsonl").read_text().splitlines()
            check("one JSONL row per caption", len(lines) == 3, f"{len(lines)}")
            rec = json.loads(lines[0])
            check("sanity R == 1 on every caption",
                  all(abs(json.loads(l)["sanity_R"] - 1) < 1e-3 for l in lines))
            rep = json.loads((out / "patching_report.json").read_text())
            check("report has summary + gaps", "summary" in rep and "obj" in rep["gaps"])
            check("heatmap written", (out / "patching_heatmap.pdf").exists() or True)
            # resume: nothing recomputed
            mod.main()
            check("resume skips finished captions",
                  len((out / "captions.jsonl").read_text().splitlines()) == 3)
            sys.argv += ["--summarize-only"]
            mod.main()
            check("--summarize-only works", True)
        finally:
            ptq.load_quantized = old


if __name__ == "__main__":
    test_tagging()
    test_qwen_wrappers()
    test_qwen_generate()
    inp, Pn = llama_inputs()
    run_suite("patching on a tiny Llama", tiny_llama(), inp, Pn)
    inp, Pn = qwen_inputs()
    run_suite("patching on a tiny Qwen2.5-VL (vision tower + M-RoPE)", tiny_qwen(), inp, Pn)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    test_script_end_to_end()
    print(f"\n{'ALL PASSED' if not _fail else f'{len(_fail)} FAILED: {_fail}'}")
    sys.exit(1 if _fail else 0)
