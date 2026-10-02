#!/usr/bin/env python
"""Regression tests for the three bugs the first full run exposed, plus the
token-selective quantizer and the token probe -- on a tiny *LLaVA* (CLIP vision tower,
projector, image-token splicing), not a bare LLaMA, because the multiplicative-A-CAB
crash only appears when a CLIP attention module is present.

    python tests/test_fixes.py
"""
from __future__ import annotations

import importlib.util
import io
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

from phoenix.acab import ACABConfig, ACABController, generate, install_acab
from phoenix.loras import LoRASCorrector, attach_loras, steer_mask
from phoenix.model import LoadedModel
from phoenix.quant import FakeQuantLinear, QuantConfig, quantize_activation, quantize_model

OK, FAIL = "  ok  ", " FAIL "
_failures = []
IMG = 31          # image token id
N_VIS = 4         # (28 / 14)^2 patches


def check(name, cond, detail=""):
    print(f"[{OK if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


class Tok:
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "left"

    def _ids(self, text, n=10):
        g = torch.Generator().manual_seed(sum(map(ord, text)) % 10_000)
        ids = torch.randint(3, 30, (n,), generator=g)
        return ids

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        if isinstance(text, str):
            ids = self._ids(text, len(text.split()) + (1 if add_special_tokens else 0))
            if return_tensors == "pt":
                return {"input_ids": ids[None], "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            return {"input_ids": ids.tolist()}
        ids = torch.stack([self._ids(t) for t in text])
        return _Enc({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def convert_tokens_to_ids(self, t):
        return {"▁Yes": 40, "Yes": 41, "▁No": 44, "No": 45}.get(t, -1)

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in row) for row in ids]


class _Enc(dict):
    def to(self, device):
        return _Enc({k: v.to(device) for k, v in self.items()})


class Proc:
    """Mimics LlavaProcessor: <image> expanded to N_VIS tokens, real pixel tensors."""
    tokenizer = Tok()

    def __call__(self, images, text, return_tensors="pt", padding=True, **kw):
        B, T = len(text), 16
        ids = torch.stack([torch.randint(3, 30, (T,), generator=torch.Generator().manual_seed(
            sum(map(ord, t)) % 10_000)) for t in text])
        ids[:, 3:3 + N_VIS] = IMG
        px = torch.stack([torch.tensor(np.asarray(im.resize((28, 28)), dtype=np.float32)
                                       ).permute(2, 0, 1) / 255.0 for im in images])
        return {"input_ids": ids, "attention_mask": torch.ones(B, T, dtype=torch.long),
                "pixel_values": px}


def tiny_llava(seed=0):
    torch.manual_seed(seed)
    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, image_size=28, patch_size=14, projection_dim=32)
    tc = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=4, vocab_size=64,
                     max_position_embeddings=256)
    cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=IMG,
                      vision_feature_layer=-2, vision_feature_select_strategy="default")
    m = LlavaForConditionalGeneration(cfg).eval().requires_grad_(False)
    return LoadedModel(model=m, processor=Proc(), image_token_id=IMG,
                       device=torch.device("cpu"), dtype=torch.float32)


def inputs(lm, n=2):
    imgs = [Image.fromarray(np.random.RandomState(k).randint(0, 255, (40, 40, 3), dtype=np.uint8))
            for k in range(n)]
    return lm.processor(images=imgs, text=[f"q{k}" for k in range(n)])


def ctrl_for(lm, **kw):
    return ACABController(ACABConfig(**kw), lm.n_layers, lm.model.config.text_config.num_attention_heads)


# =========================================================================== #
def test_mult_on_llava():
    print("\n== bug 1: multiplicative A-CAB on a model with a CLIP tower ==")
    lm = tiny_llava()
    batch = inputs(lm)
    ctrl = ctrl_for(lm, mode="mult", gate="none", alpha=1.0)
    undo = install_acab(lm.model, lm.layers, ctrl)
    try:
        r = generate(lm, batch, ctrl, max_new_tokens=4)
        ok, err = True, ""
    except Exception as e:                                   # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    undo()
    check("generate() runs with mult mode installed", ok, err[:100])
    vis_impl = lm.model.config.vision_config._attn_implementation
    check("the vision tower keeps its own attention implementation", vis_impl != "acab_mult",
          f"vision={vis_impl}")
    check("uninstall restores the language model's implementation",
          lm.model.config.text_config._attn_implementation != "acab_mult")


def test_first_token():
    print("\n== bug 3: A-CAB now reaches the first answer token ==")
    lm = tiny_llava()
    batch = inputs(lm)
    base = generate(lm, batch, None, max_new_tokens=5)

    ctrl0 = ctrl_for(lm, mode="add", gate="none", alpha=0.0)       # installed, delta = 0
    undo = install_acab(lm.model, lm.layers, ctrl0)
    zero = generate(lm, batch, ctrl0, max_new_tokens=5)
    undo()
    check("split prefill with delta = 0 reproduces the ordinary prefill",
          torch.allclose(base["first_logits"], zero["first_logits"], atol=1e-4)
          and torch.equal(base["sequences"], zero["sequences"]),
          f"max|dlogit|={float((base['first_logits'] - zero['first_logits']).abs().max()):.1e}")

    ctrl = ctrl_for(lm, mode="add", gate="none", alpha=3.0, delta_max=3.0)
    undo = install_acab(lm.model, lm.layers, ctrl)
    on = generate(lm, batch, ctrl, max_new_tokens=5, record=True)
    undo()
    dl = float((on["first_logits"] - base["first_logits"]).abs().max())
    check("with delta > 0 the FIRST answer token's logits change (POPE can now see A-CAB)",
          dl > 1e-3, f"max|dlogit|={dl:.3f}, first-token delta={on['trace']['first_delta']}")

    ctrl_old = ctrl_for(lm, mode="add", gate="none", alpha=3.0, delta_max=3.0, first_token=False)
    undo = install_acab(lm.model, lm.layers, ctrl_old)
    old = generate(lm, batch, ctrl_old, max_new_tokens=5)
    undo()
    check("first_token=False reproduces the old blind spot (for the ablation)",
          torch.allclose(old["first_logits"], base["first_logits"], atol=1e-5))

    gated = ctrl_for(lm, mode="add", gate="prev", alpha=1.0, tau=1e9)
    undo = install_acab(lm.model, lm.layers, gated)
    g = generate(lm, batch, gated, max_new_tokens=5, record=True)
    undo()
    check("the first token is gated too: a threshold nobody reaches leaves it untouched",
          torch.allclose(g["first_logits"], base["first_logits"], atol=1e-4)
          and all(x == 0 for x in g["trace"]["first_delta"]))


def big_corrector(d):
    return LoRASCorrector(torch.zeros(d, 1), torch.zeros(1, d), torch.full((d,), 5.0))


def test_loras_under_split_and_ppl():
    print("\n== LoRAS still applies under the split prefill; bug 2: perplexity ==")
    lm = tiny_llava()
    batch = inputs(lm)
    ctrl = ctrl_for(lm, mode="add", gate="none", alpha=1.0)
    undo = install_acab(lm.model, lm.layers, ctrl)
    a = generate(lm, batch, ctrl, max_new_tokens=3)["first_logits"]
    d = lm.layers[1].self_attn.k_proj.out_features
    attach_loras(lm.layers, {(1, "k"): big_corrector(d), (1, "v"): big_corrector(d)})
    b = generate(lm, batch, ctrl, max_new_tokens=3)["first_logits"]
    undo()
    check("a corrector changes the answer when the prefill is split (mask stays aligned)",
          float((a - b).abs().max()) > 1e-3, f"max|dlogit|={float((a - b).abs().max()):.3f}")

    from phoenix.data import Sample
    from phoenix.metrics import caption_perplexity
    tmp = Path(tempfile.mkdtemp())
    try:
        samples = []
        for k in range(3):
            fp = tmp / f"{k}.jpg"
            Image.fromarray(np.random.RandomState(k).randint(0, 255, (40, 40, 3), dtype=np.uint8)).save(fp)
            samples.append(Sample(k, str(fp), "describe", None, None, "caption"))
        refs = {k: ["a dog on a sofa"] for k in range(3)}
        lm2 = tiny_llava()
        p0 = caption_perplexity(lm2, samples, refs, batch_size=3)
        attach_loras(lm2.layers, {(1, "k"): big_corrector(d), (1, "v"): big_corrector(d)})
        p1 = caption_perplexity(lm2, samples, refs, batch_size=3)
        check("reference perplexity now reflects LoRAS (was identical in all four cells)",
              abs(p0 - p1) > 1e-3, f"{p0:.3f} -> {p1:.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_token_selective():
    print("\n== token-selective activation quantization ==")
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32)
    cfg = QuantConfig.from_ladder("w4a4:abits_text=8")
    fq = FakeQuantLinear(lin, cfg)
    x = torch.randn(2, 10, 64) * 3
    x[:, :, 7] = 60.0                                    # an outlier channel
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 2:6] = True
    with steer_mask(mask):
        got = fq._quantize_by_token_type(x)
    ok_img = torch.allclose(got[mask], quantize_activation(x[mask], 4), atol=1e-6)
    ok_txt = torch.allclose(got[~mask], quantize_activation(x[~mask], 8), atol=1e-6)
    check("image rows get 4 bits, text rows 8 bits", ok_img and ok_txt)
    zero4 = float((got[mask] == 0).float().mean())
    zero8 = float((got[~mask] == 0).float().mean())
    check("the outlier erases most of an image token at A4, little at A8",
          zero4 > 0.6 and zero8 < 0.2, f"zeroed: image {zero4:.2f}, text {zero8:.2f}")
    with steer_mask(mask):
        dec = fq._quantize_by_token_type(x[:, :1])       # decode step: shape mismatch
    check("decode steps count as text", torch.allclose(dec, quantize_activation(x[:, :1], 8), atol=1e-6))
    c2 = QuantConfig.from_ladder("w4a4:abits_image=8")
    check("the mirror arm parses", c2.bits_for("image") == 8 and c2.bits_for("text") == 4
          and c2.token_selective)
    check("a plain rung is not token-selective", not QuantConfig.from_ladder("w4a4").token_selective)

    lm = tiny_llava()
    quantize_model(lm.model, QuantConfig.from_ladder("w4a4:abits_text=8"), verbose=False)
    r = generate(lm, inputs(lm), None, max_new_tokens=4)
    check("a token-selective rung generates end to end", r["sequences"].shape[1] > 0)


def test_probe_script():
    print("\n== 08_token_probe: census and blind fluency on the tiny LLaVA ==")
    spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts" / "08_token_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lm = tiny_llava()
    tmp = Path(tempfile.mkdtemp())
    try:
        from phoenix.data import Sample
        samples = []
        for k in range(4):
            fp = tmp / f"{k}.jpg"
            Image.fromarray(np.random.RandomState(k).randint(0, 255, (40, 40, 3), dtype=np.uint8)).save(fp)
            samples.append(Sample(k, str(fp), "describe", None, None, "calib"))
        # the tiny model's module names match LLaVA's
        res = mod.census(lm, samples, batch_size=2)
        groups = {r["group"] for r in res["rows"]}
        kinds = {r["kind"] for r in res["rows"]}
        sane = all(0.0 <= r["erased4"] <= 1.0 and r["erased8"] <= r["erased4"] + 1e-9
                   for r in res["rows"])
        check("census covers image and text tokens and all four input kinds",
              groups == {"image", "text"} and len(kinds) == 4, f"{sorted(kinds)}")
        check("erased@8 never exceeds erased@4, all within [0, 1]", sane)
        with redirect_stdout(io.StringIO()):
            mod.print_census(res)
        from phoenix.probes import BLIND_PROMPTS, blind_fluency
        b = blind_fluency(lm, ["a dog on a sofa", "two cats"], max_new_tokens=4)
        check("blind fluency runs with no image and scores the caption tokens",
              len(b["answers"]) == len(BLIND_PROMPTS) and b["scored_tokens"] > 0
              and b["blind_ppl"] > 1.0, f"ppl={b['blind_ppl']:.1f} over {b['scored_tokens']} tokens")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_mult_on_llava()
    test_first_token()
    test_loras_under_split_and_ppl()
    test_token_selective()
    test_probe_script()
    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        sys.exit(1)
    print("all fix tests passed")
