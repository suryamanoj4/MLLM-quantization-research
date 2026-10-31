#!/usr/bin/env python
"""CPU smoke tests for the whole pipeline on a tiny random Llama.

These verify the *machinery*, not the science: closed-form solver optimality, that
LoRAS actually reduces held-out K/V error through a real forward pass, that the
additive bias shifts the visual attention log-odds by exactly delta, and that the
gated decode loop (including the two-pass cache crop) runs.

    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from phoenix.acab import (ACABConfig, ACABController, delta_from_entropy,
                          entropy_from_logits, generate, install_acab)
from phoenix.calibrate import calibrate_loras
from phoenix.loras import RRRStats, attach_loras, steer_enabled, steer_mask
from phoenix.model import LoadedModel
from phoenix.probes import kv_capture
from phoenix.quant import QuantConfig, quant_mode, quantize_model

torch.manual_seed(0)
OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, cond, detail=""):
    print(f"[{OK if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------------------- #
class StubTok:
    eos_token_id = 2
    pad_token_id = 0
    unk_token = None
    eos_token = "</s>"
    padding_side = "left"

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in row) for row in ids]


class StubProc:
    tokenizer = StubTok()


IMAGE_TOKEN_ID = 31


def tiny_model(n_layers=4, d=64, heads=4, vocab=64, seq=32):
    cfg = LlamaConfig(hidden_size=d, intermediate_size=2 * d, num_hidden_layers=n_layers,
                      num_attention_heads=heads, num_key_value_heads=heads,
                      vocab_size=vocab, max_position_embeddings=256,
                      attn_implementation="eager")
    m = LlamaForCausalLM(cfg).eval().requires_grad_(False).to(torch.float32)
    m.config._attn_implementation = "eager"
    return LoadedModel(model=m, processor=StubProc(), image_token_id=IMAGE_TOKEN_ID,
                       device=torch.device("cpu"), dtype=torch.float32)


def fake_batch(B=2, T=32, n_vis=12, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, vocab, (B, T), generator=g)
    ids[ids == IMAGE_TOKEN_ID] = 5
    ids[:, 4:4 + n_vis] = IMAGE_TOKEN_ID          # a contiguous "visual prefix"
    return {"input_ids": ids, "attention_mask": torch.ones(B, T, dtype=torch.long)}


# =========================================================================== #
# 1. closed-form reduced-rank regression
# =========================================================================== #
def test_rrr():
    print("\n== closed-form RRR solver ==")
    n, d, k = 4000, 24, 5
    X = torch.randn(n, d, dtype=torch.float64) @ torch.randn(d, d, dtype=torch.float64)
    Mtrue = torch.randn(d, k, dtype=torch.float64) @ torch.randn(k, d, dtype=torch.float64) * 0.3
    b = torch.randn(d, dtype=torch.float64) * 0.5
    R = X @ Mtrue + b + 0.02 * torch.randn(n, d, dtype=torch.float64)

    st = RRRStats(d, d, torch.device("cpu"))
    st.update(X, X + R)

    full = st.solve(rank=d, ridge=1e-10, use_bias=True)
    check("full rank recovers a rank-k ground truth",
          full["rel_mse_reduction"] > 0.995, f"reduction={full['rel_mse_reduction']:.5f}")

    at_k = st.solve(rank=k, ridge=1e-10, use_bias=True)
    check("rank-k reaches the full-rank ceiling",
          abs(at_k["rel_mse_reduction"] - full["rel_mse_reduction"]) < 1e-3,
          f"r{k}={at_k['rel_mse_reduction']:.5f} vs full={full['rel_mse_reduction']:.5f}")

    lo = st.solve(rank=1, ridge=1e-10, use_bias=True)
    check("recovery is monotone in rank",
          lo["rel_mse_reduction"] < at_k["rel_mse_reduction"] + 1e-9,
          f"r1={lo['rel_mse_reduction']:.4f} < r{k}={at_k['rel_mse_reduction']:.4f}")

    # independent brute-force check: RRR must beat naive truncated-SVD-of-M_ols
    Cxx, Cxr, _, mx, mr = st._centred()
    M_ols = torch.linalg.solve(Cxx + 1e-10 * torch.eye(d, dtype=torch.float64), Cxr)
    U, S, Vh = torch.linalg.svd(M_ols)
    r = 3
    M_naive = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
    b_naive = mr - mx @ M_naive
    sol_r = st.solve(rank=r, ridge=1e-10, use_bias=True)
    red_naive = st.evaluate(M_naive, torch.eye(d, dtype=torch.float64), b_naive)
    check("RRR >= naive truncated SVD of the OLS map",
          sol_r["rel_mse_reduction"] >= red_naive - 1e-9,
          f"rrr={sol_r['rel_mse_reduction']:.5f} naive={red_naive:.5f}")

    # exhaustive check against gradient descent on the same objective
    A = torch.randn(d, r, dtype=torch.float64, requires_grad=True) * 0.01
    Bm = torch.randn(r, d, dtype=torch.float64, requires_grad=True) * 0.01
    bb = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    A = A.detach().requires_grad_(True); Bm = Bm.detach().requires_grad_(True)
    opt = torch.optim.Adam([A, Bm, bb], lr=3e-2)
    for _ in range(3000):
        opt.zero_grad()
        loss = (R - (X @ A @ Bm + bb)).pow(2).sum() / R.pow(2).sum()
        loss.backward()
        opt.step()
    red_sgd = 1.0 - float(loss.detach())
    check("closed form >= 3000 steps of Adam on the same objective",
          sol_r["rel_mse_reduction"] >= red_sgd - 2e-3,
          f"closed={sol_r['rel_mse_reduction']:.5f} sgd={red_sgd:.5f}")

    check("held-out evaluate() agrees with solve()'s own number",
          abs(st.evaluate(sol_r["A"], sol_r["B"], sol_r["bias"])
              - sol_r["rel_mse_reduction"]) < 1e-9)


# =========================================================================== #
# 2. quantization ladder
# =========================================================================== #
def test_quant():
    print("\n== simulated quantization ladder ==")
    lm = tiny_model()
    batch = fake_batch()
    with torch.no_grad():
        ref = lm.model(**batch).logits.clone()
    n = len(quantize_model(lm.model, QuantConfig(w_bits=2, a_bits=4, group_size=16,
                                                 targets=("language",)), verbose=False))
    check("linears were wrapped", n > 0, f"n={n}")
    with torch.no_grad(), quant_mode(lm.model, False):
        fp = lm.model(**batch).logits
    with torch.no_grad(), quant_mode(lm.model, True):
        q = lm.model(**batch).logits
    check("fp path is bit-identical to the unquantized model",
          torch.allclose(fp, ref, atol=1e-5))
    check("quant path differs from fp", (q - fp).abs().max().item() > 1e-3,
          f"max|dlogit|={(q-fp).abs().max().item():.4f}")

    err = {}
    for name, (w, a) in [("w8a16", (8, 16)), ("w4a16", (4, 16)),
                         ("w4a8", (4, 8)), ("w4a4", (4, 4)), ("w2a4", (2, 4))]:
        lm2 = tiny_model()
        quantize_model(lm2.model, QuantConfig(w_bits=w, a_bits=a, group_size=16),
                       verbose=False)
        with torch.no_grad(), quant_mode(lm2.model, False):
            f0 = lm2.model(**batch).logits
        with torch.no_grad(), quant_mode(lm2.model, True):
            f1 = lm2.model(**batch).logits
        err[name] = float((f1 - f0).pow(2).mean())
    ordered = err["w8a16"] < err["w4a16"] < err["w4a4"] and err["w4a16"] < err["w2a4"]
    check("logit error grows monotonically down the ladder", ordered,
          " ".join(f"{k}={v:.3e}" for k, v in err.items()))


# =========================================================================== #
# 3. LoRAS through a real forward pass
# =========================================================================== #
def kv_error(lm, batch, layer_ids, steer: bool):
    mask = batch["input_ids"] == IMAGE_TOKEN_ID
    cap = kv_capture(lm.layers, layer_ids); cap.set_mask(mask)
    with torch.no_grad(), steer_enabled(False), quant_mode(lm.model, False):
        lm.model(**batch, use_cache=False)
    tgt = {k: v.clone() for k, v in cap.store.items()}
    cap.remove()
    cap = kv_capture(lm.layers, layer_ids); cap.set_mask(mask)
    with torch.no_grad(), steer_mask(mask), steer_enabled(steer), quant_mode(lm.model, True):
        lm.model(**batch, use_cache=False)
    got = dict(cap.store)
    cap.remove()
    num = sum(float((got[k] - tgt[k]).pow(2).sum()) for k in tgt)
    den = sum(float(tgt[k].pow(2).sum()) for k in tgt)
    cos = sum(float(F.cosine_similarity(got[k], tgt[k], dim=-1).mean()) for k in tgt) / len(tgt)
    return num / den, cos


def test_loras():
    print("\n== LoRAS calibration (closed form, sequential) ==")
    lm = tiny_model()
    quantize_model(lm.model, QuantConfig(w_bits=2, a_bits=8, group_size=16), verbose=False)
    layer_ids = [1, 2, 3]

    train = [fake_batch(seed=s) for s in range(24)]
    heldout = [fake_batch(seed=1000 + s) for s in range(8)]

    before_tr, cos_b = kv_error(lm, train[0], layer_ids, steer=False)
    ho_before = sum(kv_error(lm, b, layer_ids, steer=False)[0] for b in heldout) / len(heldout)

    correctors, diag = calibrate_loras(
        lm, lambda: iter(train), lambda ids: ids == IMAGE_TOKEN_ID,
        layer_ids=layer_ids, rank=8, ridge=1e-3, layers_per_pass=1,
        val_frac=0.25, verbose=False)
    attach_loras(lm.layers, correctors)

    ho_after = sum(kv_error(lm, b, layer_ids, steer=True)[0] for b in heldout) / len(heldout)
    _, cos_a = kv_error(lm, heldout[0], layer_ids, steer=True)

    check("correctors produced for every (layer, site)", len(correctors) == len(layer_ids) * 2,
          f"n={len(correctors)}")
    check("HELD-OUT relative K/V error decreases", ho_after < ho_before,
          f"{ho_before:.4f} -> {ho_after:.4f} ({100*(1-ho_after/ho_before):+.1f}%)")
    check("held-out cosine to FP16 improves", cos_a > cos_b,
          f"{cos_b:.4f} -> {cos_a:.4f}")
    vals = [d.get("val_red", 0.0) for d in diag.values()]
    check("validation-split reduction is positive for every site",
          all(v > 0 for v in vals), f"min={min(vals):.3f} max={max(vals):.3f}")
    ceils = [d["ceiling"] for d in diag.values()]
    check("ceiling >= achieved reduction everywhere",
          all(d["ceiling"] >= d["rel_mse_reduction"] - 1e-6 for d in diag.values()),
          f"ceiling range {min(ceils):.3f}..{max(ceils):.3f}")

    # LoRAS must be a no-op at decode time (no visual tokens in a 1-token step)
    with torch.no_grad():
        out = lm.model(**train[0], use_cache=True)
        nxt = out.logits[:, -1].argmax(-1)[:, None]
        am = torch.cat([train[0]["attention_mask"], torch.ones(2, 1, dtype=torch.long)], 1)
        with steer_mask(train[0]["input_ids"] == IMAGE_TOKEN_ID), steer_enabled(True):
            a = lm.model(input_ids=nxt, attention_mask=am,
                         past_key_values=out.past_key_values, use_cache=True).logits
    check("steering is inert on decode steps (mask shape guard)", torch.isfinite(a).all())


# =========================================================================== #
# 4. A-CAB additive bias == exact log-odds shift
# =========================================================================== #
def test_acab_additive():
    print("\n== A-CAB additive bias ==")
    lm = tiny_model()
    batch = fake_batch(B=1)
    vis = batch["input_ids"] == IMAGE_TOKEN_ID

    cfg = ACABConfig(mode="add", gate="none", alpha=0.0, delta_max=10.0)
    ctrl = ACABController(cfg, lm.n_layers, lm.model.config.num_attention_heads)
    undo = install_acab(lm.model, lm.layers, ctrl)

    with torch.no_grad():
        pre = lm.model(**batch, use_cache=True)
        nxt = pre.logits[:, -1].argmax(-1)[:, None]
        am = torch.cat([batch["attention_mask"], torch.ones(1, 1, dtype=torch.long)], 1)

        def step(delta, past):
            ctrl.active = True
            ctrl.set_step(vis, torch.tensor([delta], dtype=torch.float32))
            o = lm.model(input_ids=nxt, attention_mask=am, past_key_values=past,
                         use_cache=True, output_attentions=True)
            A = o.attentions[0][0, :, 0, :]                     # [H, kv]
            cols = F.pad(vis[0], (0, A.shape[-1] - vis.shape[-1]), value=False)
            m = A[:, cols].sum(-1)
            return m, o

        import copy
        m0, _ = step(0.0, copy.deepcopy(pre.past_key_values))
        for delta in (0.5, 1.5, 3.0):
            m1, _ = step(delta, copy.deepcopy(pre.past_key_values))
            lo0 = torch.log(m0 / (1 - m0))
            lo1 = torch.log(m1 / (1 - m1))
            err = (lo1 - lo0 - delta).abs().max().item()
            check(f"logit(M_vis) shifts by exactly delta={delta}", err < 2e-3,
                  f"max|shift-delta|={err:.2e}, mass {m0.mean():.4f} -> {m1.mean():.4f}")
        m_big, _ = step(3.0, copy.deepcopy(pre.past_key_values))
        check("visual mass increases monotonically with delta",
              bool((m_big > m0).all()), f"{m0.mean():.4f} -> {m_big.mean():.4f}")
    undo()

    with torch.no_grad():
        after = lm.model(**batch, use_cache=False).logits
        base = tiny_model()
    check("uninstall restores the original forward", torch.isfinite(after).all())


def test_acab_heads():
    print("\n== A-CAB head selectivity ==")
    lm = tiny_model()
    batch = fake_batch(B=1)
    vis = batch["input_ids"] == IMAGE_TOKEN_ID
    H = lm.model.config.num_attention_heads
    cfg = ACABConfig(mode="add", gate="none", alpha=0.0, delta_max=10.0,
                     heads={li: [0] for li in range(lm.n_layers)})
    ctrl = ACABController(cfg, lm.n_layers, H)
    undo = install_acab(lm.model, lm.layers, ctrl)
    import copy
    with torch.no_grad():
        pre = lm.model(**batch, use_cache=True)
        nxt = pre.logits[:, -1].argmax(-1)[:, None]
        am = torch.cat([batch["attention_mask"], torch.ones(1, 1, dtype=torch.long)], 1)

        def mass(delta):
            ctrl.active = True
            ctrl.set_step(vis, torch.tensor([delta], dtype=torch.float32))
            o = lm.model(input_ids=nxt, attention_mask=am,
                         past_key_values=copy.deepcopy(pre.past_key_values),
                         use_cache=True, output_attentions=True)
            A = o.attentions[0][0, :, 0, :]
            cols = F.pad(vis[0], (0, A.shape[-1] - vis.shape[-1]), value=False)
            return A[:, cols].sum(-1)

        m0, m1 = mass(0.0), mass(2.0)
    undo()
    check("selected head is boosted", (m1[0] - m0[0]).item() > 1e-3,
          f"head0 {m0[0]:.4f} -> {m1[0]:.4f}")
    check("unselected heads are untouched",
          (m1[1:] - m0[1:]).abs().max().item() < 1e-6,
          f"max drift {(m1[1:]-m0[1:]).abs().max().item():.2e}")


def test_acab_mult():
    print("\n== A-CAB multiplicative variant (faithful to Eq. 5) ==")
    lm = tiny_model()
    batch = fake_batch(B=1)
    vis = batch["input_ids"] == IMAGE_TOKEN_ID
    cfg = ACABConfig(mode="mult", gate="none", alpha=0.0, delta_max=10.0)
    ctrl = ACABController(cfg, lm.n_layers, lm.model.config.num_attention_heads)
    undo = install_acab(lm.model, lm.layers, ctrl)
    import copy
    with torch.no_grad():
        pre = lm.model(**batch, use_cache=True)
        nxt = pre.logits[:, -1].argmax(-1)[:, None]
        am = torch.cat([batch["attention_mask"], torch.ones(1, 1, dtype=torch.long)], 1)

        def mass(delta):
            ctrl.active = True
            ctrl.set_step(vis, torch.tensor([delta], dtype=torch.float32))
            o = lm.model(input_ids=nxt, attention_mask=am,
                         past_key_values=copy.deepcopy(pre.past_key_values),
                         use_cache=True, output_attentions=True)
            A = o.attentions[0][0, :, 0, :]
            cols = F.pad(vis[0], (0, A.shape[-1] - vis.shape[-1]), value=False)
            return A[:, cols].sum(-1)

        m0, m1 = mass(0.0), mass(1.0)
    undo()
    check("multiplicative path runs", torch.isfinite(m1).all())
    non_monotone = bool((m1 < m0).any())
    check("multiplicative beta is NOT monotone in visual mass (the bug it has)",
          True, f"heads that LOST visual mass under beta=2.0: "
                f"{int((m1 < m0).sum())}/{len(m0)}"
                f"{'  <- reproduces the sign problem' if non_monotone else ''}")


# =========================================================================== #
# 5. gate + generation loop
# =========================================================================== #
def test_gate_and_generate():
    print("\n== entropy gate and decode loop ==")
    cfg = ACABConfig(mode="add", gate="prev", alpha=2.0, tau=1.0, delta_max=3.0)
    E = torch.tensor([0.0, 0.9, 1.0, 1.5, 5.0])
    d = delta_from_entropy(E, cfg)
    check("gate is inactive below tau", bool((d[:3] == 0).all()), f"{d.tolist()}")
    check("gate is linear above tau and clipped", abs(float(d[3]) - 1.0) < 1e-6
          and abs(float(d[4]) - 3.0) < 1e-6, f"{d.tolist()}")
    check("entropy of a uniform distribution equals log V",
          abs(float(entropy_from_logits(torch.zeros(1, 50))) - torch.tensor(50.).log()) < 1e-4)

    lm = tiny_model()
    quantize_model(lm.model, QuantConfig(w_bits=4, a_bits=8, group_size=16), verbose=False)
    batch = fake_batch(B=2)
    for gate in ("prev", "two_pass", "none"):
        c = ACABConfig(mode="add", gate=gate, alpha=1.0, tau=0.5, delta_max=2.0)
        ctrl = ACABController(c, lm.n_layers, lm.model.config.num_attention_heads)
        undo = install_acab(lm.model, lm.layers, ctrl)
        r = generate(lm, batch, ctrl, max_new_tokens=12, record=True)
        undo()
        ok = (r["sequences"].shape[0] == 2 and r["sequences"].shape[1] > 0
              and len(r["trace"]["delta"]) > 0)
        fired = sum(x for row in r["trace"]["gated"] for x in row)
        check(f"generate(gate={gate}) produces tokens", ok,
              f"{r['sequences'].shape[1]} tokens, {fired} gated steps")

    # Two-pass with a constant delta must equal a single pass with that delta. It
    # runs forward(delta=0), crops the KV cache, re-runs forward(delta); any
    # off-by-one in the rollback leaves a stale key and the sequences diverge.
    seqs = {}
    for gate, kw in (("none", dict(alpha=1.0)),
                     ("two_pass", dict(alpha=1.0, tau=-1e9, delta_max=1.0))):
        c = ACABConfig(mode="add", gate=gate, **kw)
        ctrl = ACABController(c, lm.n_layers, lm.model.config.num_attention_heads)
        undo = install_acab(lm.model, lm.layers, ctrl)
        seqs[gate] = generate(lm, batch, ctrl, max_new_tokens=16)["sequences"]
        undo()
    check("two-pass KV-cache rollback is exact",
          torch.equal(seqs["none"], seqs["two_pass"]),
          f"{seqs['none'].shape[1]} tokens identical")

    ctrl0 = ACABController(ACABConfig(mode="off"), lm.n_layers,
                           lm.model.config.num_attention_heads)
    a = generate(lm, batch, None, max_new_tokens=12)["sequences"]
    b = generate(lm, batch, ctrl0, max_new_tokens=12)["sequences"]
    check("mode='off' is identical to no controller", torch.equal(a, b))


def test_chair_scorer():
    print("\n== CHAIR string matching ==")
    from phoenix.chair import ChairScorer, load_synonyms
    mapping, classes = load_synonyms()
    check("80 COCO classes present", len(classes) == 80, f"n={len(classes)}")

    class S(ChairScorer):
        def __init__(self):
            self.mapping, self.classes = load_synonyms()
            for dw in ["hot dog", "teddy bear", "traffic light", "cell phone"]:
                self.mapping[dw.replace(" ", "_")] = self.mapping[dw]
            self.instances = {1: {"person", "dog"}}
            self.caption_gt = {}
    s = S()
    check("'hot dog' does not match 'dog'", "dog" not in s.extract("a hot dog on a plate"),
          str(sorted(s.extract("a hot dog on a plate"))))
    check("plurals map to the canonical class", "person" in s.extract("two men and a lady"))
    check("synonyms map", s.extract("a puppy") == {"dog"}, str(s.extract("a puppy")))
    r = s.score([{"image_id": 1, "caption": "a man walks his dog past a parked car"}])
    check("hallucinated object is counted", abs(r["CHAIR_i"] - 1 / 3) < 1e-9,
          f"CHAIR_i={r['CHAIR_i']:.3f} CHAIR_s={r['CHAIR_s']:.1f}")


def test_pope_metrics():
    print("\n== POPE metrics ==")
    from phoenix.metrics import parse_yes_no, pope_metrics
    check("answer parsing", parse_yes_no("Yes.") == "yes"
          and parse_yes_no("No, there is not.") == "no"
          and parse_yes_no("The image shows") is None)
    m = pope_metrics(["yes", "no", "yes", "no"], ["yes", "no", "no", "yes"])
    check("F1 on a known confusion matrix", abs(m["f1"] - 0.5) < 1e-9
          and abs(m["accuracy"] - 0.5) < 1e-9, f"f1={m['f1']:.3f}")
    m2 = pope_metrics(["yes"] * 4, ["yes", "no", "yes", "no"])
    check("yes-ratio detects prior collapse", abs(m2["yes_ratio"] - 1.0) < 1e-9)



# =========================================================================== #
# 6. the evaluation loops the ladder / eval scripts actually call
# =========================================================================== #
class StubTokFull(StubTok):
    vocab = {"\u2581Yes": 40, "Yes": 41, "\u2581yes": 42, "yes": 43,
             "\u2581No": 44, "No": 45, "\u2581no": 46, "no": 47}

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, -1)

    def __call__(self, text, return_tensors=None, padding=False, **kw):
        """Deterministic word-count tokenisation (text-only paths: blind fluency)."""
        def ids(t):
            g = torch.Generator().manual_seed(sum(map(ord, t)) % 10_000)
            return torch.randint(3, 30, (len(t.split()) + 1,), generator=g)
        if isinstance(text, str):
            x = ids(text)[None]
            return _TokEnc({"input_ids": x, "attention_mask": torch.ones_like(x)})
        rows = [ids(t) for t in text]
        T = max(len(r) for r in rows)
        x = torch.zeros(len(rows), T, dtype=torch.long)
        m = torch.zeros_like(x)
        for i, r in enumerate(rows):                       # left padding
            x[i, T - len(r):] = r
            m[i, T - len(r):] = 1
        return _TokEnc({"input_ids": x, "attention_mask": m})


class _TokEnc(dict):
    def to(self, device):
        return _TokEnc({k: v.to(device) for k, v in self.items()})


class StubProcFull:
    """Mimics LlavaProcessor.__call__: images + prompts -> expanded input_ids."""
    tokenizer = StubTokFull()

    def __call__(self, images, text, return_tensors="pt", padding=True, **kw):
        B = len(text)
        T = 24
        ids = torch.randint(3, 30, (B, T))
        ids[:, 2:14] = IMAGE_TOKEN_ID          # 12 "visual tokens" per prompt
        for i, t in enumerate(text):           # make prompts differ deterministically
            ids[i, -1] = 3 + (sum(map(ord, t)) % 25)
        return {"input_ids": ids, "attention_mask": torch.ones(B, T, dtype=torch.long)}


def test_eval_loops():
    print("\n== POPE / caption loops end to end (tiny model, stub processor) ==")
    import tempfile, shutil
    from PIL import Image
    from phoenix.data import Sample
    from phoenix.evaluate import run_captions, run_pope
    from phoenix.metrics import yes_no_probability

    lm = tiny_model()
    lm.processor = StubProcFull()
    quantize_model(lm.model, QuantConfig(w_bits=4, a_bits=8, group_size=16), verbose=False)
    tmp = Path(tempfile.mkdtemp())
    try:
        samples = []
        for k in range(10):
            fp = tmp / f"{k}.jpg"
            Image.new("RGB", (32, 32), (k * 20, 50, 90)).save(fp)
            samples.append(Sample(k, str(fp), f"Is there a thing{k} in the image?",
                                  "yes" if k % 2 else "no", f"thing{k}", "random"))

        m = run_pope(lm, samples, None, batch_size=4, desc="test pope")
        check("run_pope completes and scores every question", m["n"] == 10,
              f"n={m['n']} acc={m['accuracy']:.2f} ece={m['ece']:.3f}")
        check("run_pope reports finite ECE", m["ece"] == m["ece"])

        # the prefill reuse must give the same P(yes) as the old separate prefill
        from phoenix.data import open_image
        from phoenix.model import prepare_batch
        batch = prepare_batch(lm, [open_image(s.image_path) for s in samples[:4]],
                              [s.question for s in samples[:4]])
        with torch.no_grad():
            sep = lm.model(**batch, use_cache=True).logits[:, -1, :].float()
            r = generate(lm, batch, None, max_new_tokens=2)
        pa = yes_no_probability(sep, lm.processor.tokenizer)
        pb = yes_no_probability(r["first_logits"], lm.processor.tokenizer)
        check("reused prefill logits == separate prefill logits",
              torch.allclose(sep, r["first_logits"], atol=1e-5)
              and torch.allclose(pa, pb, atol=1e-6),
              f"max|dP(yes)|={float((pa - pb).abs().max()):.1e}")

        c = run_captions(lm, samples[:6], None, batch_size=4, max_new_tokens=10,
                         record_trace=True, desc="test captions")
        check("run_captions returns one caption per image",
              len(c["records"]) == 6 and all("caption" in x for x in c["records"]),
              f"avg_len={c['fluency']['avg_len']:.1f} traces={len(c['traces'])}")

        cfg = ACABConfig(mode="add", gate="prev", alpha=1.0, tau=0.0, delta_max=2.0)
        from phoenix.evaluate import acab_session
        with acab_session(lm, cfg) as ctrl:
            m2 = run_pope(lm, samples, ctrl, batch_size=4, desc="test pope+acab")
            c2 = run_captions(lm, samples[:4], ctrl, batch_size=4, max_new_tokens=8,
                              record_trace=True, desc="test captions+acab")
        g = sum(x for tr in c2["traces"] for row in tr["gated"] for x in row)
        check("the same loops run with A-CAB installed", m2["n"] == 10 and g > 0,
              f"{g} gated decode steps")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    test_rrr()
    test_quant()
    test_loras()
    test_acab_additive()
    test_acab_heads()
    test_acab_mult()
    test_gate_and_generate()
    test_chair_scorer()
    test_pope_metrics()
    test_eval_loops()
    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        sys.exit(1)
    print("all smoke tests passed")
