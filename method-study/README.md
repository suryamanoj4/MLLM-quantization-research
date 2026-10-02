# Phoenix — Contributions 2 & 3

A study of how *practically* quantized MLLMs fail at visual grounding (published PTQ
methods: NF4, AWQ, GPTQ, SmoothQuant, QuaRot-style rotation, on LLaVA-1.5-7B), with
**LoRAS** (low-rank activation steering) and **A-CAB** (entropy-gated cross-modal
attention biasing) as the conditional method, plus the profiling and evaluation
needed to show either one works. Start at **section 4.0, Gate 1**.

Target environment: **A6000 (48 GB), torch ≤ 2.6, transformers 4.53.3**.
Everything runs on one GPU. Nothing needs to be trained.

---

## 1. Assessment of the proposal as written

The framing is good: "PTQ papers optimise reconstruction error and are blind to
generation dynamics; hallucination papers assume FP16" is a real gap, and Lexical
Fallback is a crisp name for a mechanism that does exist. Three things in the
technical sections will not survive contact with an implementation, and two more
will produce results that don't mean what you'd want them to mean.

### 1.1 C2 — Eq. 4 has a closed-form solution; don't run SGD on it

The proposal learns `W_A, W_B` by minimising

    L = (1/|D|) Σ ‖ H_FP16 − (H_Q + H_Q W_A W_B) ‖²_F

"within minutes on a single consumer GPU". But this is *reduced-rank ridge
regression*, and it has an exact global optimum:

    M_ols = (Cxx + λI)⁻¹ Cxr          Cxx = XᵀX,  Cxr = Xᵀ(Y − X)
    G     = M_olsᵀ Cxx M_ols
    V_r   = top-r eigenvectors of G
    A     = M_ols V_r ,  B = V_rᵀ

(Izenman 1975. Note the subtlety: you truncate the SVD of the *fitted values*, not
of `M_ols`. Truncating `M_ols` directly is a strictly worse solution — the smoke
test measures the gap, ~3 points of recovery at r=3 on synthetic data.)

Why it matters beyond elegance:

* **It is free.** Sufficient statistics stream in `O(d²)` regardless of calibration
  size; the solve is one `linalg.solve` plus one `eigh` on a 4096×4096 matrix.
  Seconds per layer instead of minutes, and no learning rate, epoch count, warmup or
  seed to defend in review.
* **It is exactly reproducible.** No seed variance table needed.
* **It hands you the rank-vs-recovery curve for free.** The eigenvalues of `G` are
  an explained-variance spectrum of the correctable error, so a single fit tells you
  what every rank from 1 to 4096 would have achieved, and in particular what the
  **full-rank linear ceiling** is.

That last point is the one I'd actually build the section around. If a full-rank
linear map only removes, say, 30% of the K/V error, then *no* rank-16 adapter
trained by any method will do better, and you know that in week 4 instead of week
11. The calibration script prints this ceiling and warns when it is low. It converts
a possible negative result into a reportable finding: *"activation-space quantization
error in MLLM decoders is only X% linearly recoverable"* is a contribution; *"our
adapter didn't help"* is not.

Verified empirically in `tests/test_smoke.py`: the closed form matches 3000 steps of
Adam on the identical objective to 5 decimal places (0.92667 vs 0.92667), in about
1/400 of the time.

### 1.2 C2 — §4.3 asserts the injection site instead of measuring it

> residual steering is selectively attached to the Key and Value projection inputs
> of the visual tokens in transformer blocks (l ≥ 0.75L)

…while §3.2 promises layer-depth drift profiling "to locate the exact network depth
where cross-modal representation drift reaches maximum divergence". You can't do
both, so the placement should come from the measurement.

**What the measurement showed (LLaVA-1.5-7B, W4A8):** drift grows steadily with
depth, and the eight worst layers are 23 and 25–31, nearly identical to the
proposal's top-quartile rule. I had predicted a mid-network peak, and that was wrong
for this model at this precision. The error accumulates rather than being introduced
at one depth.

That leaves a question the ablation should answer: correcting where the drift is
largest is not obviously best. A correction early in the network might stop error
from compounding, and a late one only patches the result. Compare
`--layers 23,25,26,27,28,29,30,31` (max drift) with a spread set such as
`--layers 4,8,12,16,20,24,28,31`, at the same parameter budget.

`scripts/01_profile_drift.py` measures the drift and emits the layer set;
`scripts/02_calibrate_loras.py --drift-json ...` consumes it, and `--layers` forces any
other set.

Two smaller fixes in the same area:

* **Steer the K/V *outputs*, not their inputs.** k_proj and v_proj share an input
  with q_proj, so correcting the input either perturbs the query too or needs a
  duplicated projection. Correcting the outputs (pre-RoPE, hence position
  independent) is surgical. It also has a property worth a sentence in the paper:
  visual tokens exist only in the prefix, so the correction runs **once during
  prefill and the KV cache carries it** — LoRAS costs literally zero at decode time,
  rather than the per-token cost Eq. 3 implies. `scripts/05_latency.py` checks this.
* **Calibrate sequentially.** Fitting every layer against uncorrected quantized
  activations solves a problem that won't exist at inference once the earlier
  correctors are live. Same reason GPTQ/OmniQuant calibrate block by block.
  `--layers-per-pass 1` is strict, `4` is the default compromise.

### 1.3 C3 — the entropy gate as written is not causal

Eq. 7 computes `E_t` from the LM logits at step *t*. Eq. 8 uses `E_t` to set `β_t`,
which changes the attention that **produces those logits**. It's a fixed point, not
a forward pass — you cannot evaluate it in one pass, which is also why the "single
forward pass, <1.5% latency" claim in §5.3 can't be right as stated.

Two honest resolutions, both implemented and switchable with `--gate`:

* `prev` (default) — gate on `E_{t−1}`. Single pass, zero extra cost. Predictive
  entropy is strongly autocorrelated along a caption, so this is a good proxy, and
  the trace logs let you *measure* that autocorrelation and report it.
* `two_pass` — forward with δ=0, read `E_t`, crop the KV cache, re-forward with δ.
  Exact "current-step" semantics, ~2× decode cost **on gated steps only**. This is
  your quality ceiling and the ablation that shows what the lag costs you.

### 1.4 C3 — multiplicative logit scaling is the wrong operator

Eq. 5 sets `S*_vis = β_t · S_vis` with `β_t ≥ 1`. Attention logits are signed. For a
visual key with `S_vis < 0`, multiplying by `β > 1` makes it *more* negative — the
"restorative" multiplier actively suppresses exactly the visual tokens the model was
already ignoring. What `β` does is sharpen the distribution *within* the visual
block; its effect on total visual mass depends on the sign structure of the logits
and is not even guaranteed to be positive.

This is not hypothetical. In the smoke test, with β=2.0 on a small model, **3 of 4
heads lost visual attention mass.**

Replace it with an **additive** bias on the visual logits, `S*_vis = S_vis + δ_t`.
Then for any distribution, exactly:

    logit(M_vis*) = logit(M_vis) + δ_t

So δ is a clean, monotone, interpretable shift of the **log-odds of attending to the
image**, and the ranking inside the visual block is preserved exactly. The gate
becomes `δ_t = min(δ_max, α·ReLU(E_t − τ))`. Verified to 1e-7 in the smoke test.

It is also much cheaper. An additive bias on a subset of keys *is* an additive
attention mask, so it rides the existing `attention_mask` argument: no custom kernel,
no eager fallback, works unchanged under SDPA/FlashAttention. That's how the
`<1.5%` latency claim becomes achievable rather than aspirational. The multiplicative
form is kept as `--acab mult` so you can report the comparison — "the natural
multiplicative formulation is non-monotone in visual mass, we use the additive form"
is a genuinely nice half-page.

### 1.5 C3 — τ in raw nats does not transfer

`τ ∈ [0.5, 3.0]` in nats depends on vocabulary size and on how calibrated the model
is, and quantization *changes the entropy distribution* — which is half your thesis.
So a fixed τ silently means "intervene on 20% of steps" at FP16 and "intervene on
60% of steps" at W4A4, and your ablation confounds gate aggressiveness with
precision. `--tau-percentile 60` sets τ from the model's own decode-time entropy
distribution so the gate means the same thing at every rung of the ladder.

### 1.6 Two evaluation risks worth fixing before you run anything

**CHAIR is trivially gameable by saying less.** A caption of "A dog." has CHAIR_s
near zero. Any intervention that shortens or flattens generation will look like it
reduces hallucination. Every CHAIR number in this repo is reported next to **object
coverage** (fraction of ground-truth objects actually mentioned), caption length, and
4-gram repetition rate. If coverage drops when CHAIR drops, you have a length effect,
not a grounding effect. `scripts/04_acab_sweep.py` plots the CHAIR_i-vs-coverage
operating curve directly, which is the honest way to present an α sweep.

**Weight-only W4 may not degrade enough to have headroom.** AWQ/GPTQ W4A16 on
LLaVA-1.5-7B typically costs only a point or two of POPE F1. If your teammates'
"4-bit degrades" result came from W4A16, LoRAS and A-CAB will be fighting for scraps.
That is why the default here is W4A8/W4A4 with simulated activation quantization:
Table 1 in the proposal *requires* W4A8 and W4A4 rows, and no library provides 4-bit
activation kernels for MLLMs, so simulation isn't a shortcut — it's the only route.
**Run `scripts/00_precision_ladder.py` first** and pick the rung where the model
actually breaks before you calibrate anything.

### 1.7 Smaller notes

* §3.1 "unconditional LM logits" in Eq. 7 conditions on `v`. It's the ordinary
  next-token distribution. Fix the wording; the implementation is the sensible one.
* Eq. 2's `D_KL(P_Q ‖ P_blind)` decreasing with bit-width is good evidence for
  fallback, but it's asymmetric and also decreases if the model just gets more
  uniform. Pair it with the visual-mass measurement (`measure_visual_mass`) so a
  reviewer can't read it as an entropy artefact.
* AWQ contributes W4A16 only, which your Table 1 footnote already flags — good.
  `MiniCPM-V + DocVQA` is a lot of surface area for a course project; I'd cut to
  LLaVA-1.5-7B for everything and add Qwen2-VL only as a generalisation check.
* C2 and C3 interact: if LoRAS restores visual K/V, A-CAB has less to do. The 2×2
  ablation (neither / LoRAS / A-CAB / both) is the interesting table, not two
  separate ones. `run_all.sh` runs exactly those four cells.

---

## 2. What changed, in one table

| | proposal | here | why |
|---|---|---|---|
| LoRAS fit | SGD on MSE | closed-form reduced-rank ridge regression | exact optimum, seconds, reproducible, free rank curve |
| LoRAS layers | asserted `l ≥ 0.75L` | chosen from measured drift (`--layers` to override) | §3.2 and §4.3 contradict each other |
| LoRAS site | K/V projection *inputs* | K/V *outputs*, pre-RoPE | surgical; zero decode cost via the KV cache |
| LoRAS calibration | per-layer independent | sequential, error-propagating | matches what inference will see |
| new diagnostic | — | full-rank linear recoverability ceiling | tells you in week 4 whether C2 can work |
| A-CAB operator | `β·S_vis`, β≥1 | `S_vis + δ`, exact log-odds shift | multiplicative is non-monotone in visual mass |
| A-CAB gate | `E_t` (circular) | `E_{t−1}` (default) or two-pass | Eq. 7/8 are not evaluable in one pass |
| A-CAB threshold | τ in nats | τ = percentile of the model's entropy | nats don't transfer across the ladder |
| A-CAB kernel | implied custom | additive attention mask | SDPA-compatible, no kernel work |
| A-CAB scope | all layers | optional top-fraction visual heads | quality/fluency trade-off |
| CHAIR reporting | CHAIR_s, CHAIR_i | + coverage, length, rep-4, PPL | CHAIR alone rewards saying less |
| quantization | AWQ/GPTQ/MQuant | simulated ladder (+ optional real AWQ) | only way to get W4A8/W4A4; aligned activations for calibration |

---

## 3. Install

```bash
bash scripts/setup_env.sh      # isolated venv + the right torch for this driver (~5.5 GB)
PY=.venv-phoenix/bin/python    # or: source .venv-phoenix/bin/activate
$PY scripts/check_env.py       # ~2 s: GPU, driver/torch match, venv, deps, disk
$PY tests/test_smoke.py        # ~1 min on CPU, no model download, 41 checks
$PY tests/test_data.py         # ~5 s, offline, 29 checks
```

`setup_env.sh` exists because every obvious alternative fails on some machine:

* Ubuntu 23.04+ blocks `pip install` into the system Python (PEP 668,
  "externally-managed-environment"), so it has to be a venv.
* A `(.venv)` prompt doesn't mean `python` is the venv's. A moved or half-built venv
  keeps the prompt but runs `/usr/bin/python`. The script only ever calls
  `$VENV/bin/python` by explicit path, and `run_all.sh` picks up `.venv-phoenix`
  automatically.
* The venv is built without `--system-site-packages`, so a `pip install --user` torch
  in `~/.local` (for example, a CUDA 13 build this driver can't run) can't leak in.
* The torch build comes from `nvidia-smi`: driver CUDA >= 12.4 gets plain
  `torch==2.6.0` from PyPI (the CUDA 12.4 build), and 11.8 to 12.3 gets the cu118
  index. Override with `TORCH_INDEX=...`.
* torch 2.6 plus its CUDA libraries is 5.2 GB installed (measured), and pip also stages
  ~2.7 GB of wheels in `$TMPDIR`. The script checks free space before downloading
  anything. If there isn't enough, it lists what can be reclaimed (pip cache, a stray
  `~/.local` torch, old venvs) with sizes and exact commands, or you can point it at a
  bigger disk: `VENV=/big/disk/venv TMPDIR=/big/disk/tmp bash scripts/setup_env.sh`.
* If `python3-venv` isn't installed (no sudo), it bootstraps pip from pip's own wheel
  on PyPI.

### Data — ~110 MB, not 20 GB

Nothing here is trained, so the full COCO download is pure waste. One command builds
a mini-COCO with exactly what the project consumes:

```bash
python scripts/fetch_data.py --root data/coco-mini              # ~110 MB (dev)
python scripts/fetch_data.py --root data/coco-mini --preset tiny   # ~40 MB
python scripts/fetch_data.py --root data/coco-mini --preset full   # ~200 MB, all 500 POPE images
python scripts/fetch_data.py --root data/coco-mini --verify        # re-check an existing build
```

| preset | eval images | calib images | on disk |
|---|---|---|---|
| `tiny` | 100 | 96 | ~40 MB |
| `dev` (default) | 250 | 192 | ~110 MB |
| `full` | 500 (the whole POPE set) | 256 | ~200 MB |

Three things make it small:

* **Images are fetched individually.** COCO serves every image over plain HTTP, so
  we pull the few hundred ids we selected instead of the 6.2 GB `val2014.zip`.
* **Annotations are extracted by range request, resumably.** `annotations_trainval2014.zip` is
  241 MB and expands to ~1.35 GB, but only two of its six members matter. The script
  reads the zip's central directory over HTTP ranges and streams out just
  `instances_val2014.json` and `captions_val2014.json` (~55 MB), subsets them to the
  selected images, and deletes the originals. Final annotation files are a few MB.
  Every stage has a progress bar. A stalled connection is abandoned after 30 s and
  resumed from the last byte received; Ctrl-C and rerun continues the partial file
  instead of restarting; each member is CRC-checked against the zip's own directory.
  Falls back to a normal download if the server refuses ranges, and
  `--annotations-zip /path/to/local.zip` skips the network entirely.
* **POPE comes from the official question files**, which are ~370 KB each and all
  three splits share the same 500 images — so one image set serves every split, and
  your POPE numbers stay comparable to published ones instead of being locally
  reconstructed.

**Leakage is enforced, not hoped for.** The build writes `splits.json` pinning an
`eval_ids` / `calib_ids` partition; calibration images are drawn only from val2014
images appearing in *no* POPE split. `phoenix/data.py` reads that manifest and
restricts every sampler to its own pool, so `build_calibration_set` cannot return an
image that `build_pope` or `build_caption_set` will later evaluate on — regardless of
seeds or `--n-*` flags. `write_splits` refuses to write an overlapping partition and
`read_splits` refuses to load one.

A full COCO checkout still works unchanged: point `--coco-root` at it and, with no
`splits.json` present, the samplers behave as before (calibration from `train2014`,
or a disjoint `val2014` slice).

If you want a 500-image POPE run later, re-run with `--preset full`; already-present
images are skipped, so it only fetches the delta.

---

## 4. Running it

```bash
bash run_all.sh
```

That builds `data/coco-mini` if it's missing, runs the preflight check, then the whole
loop. To use an existing full COCO checkout instead, run
`COCO=/path/to/coco bash run_all.sh` — as an alternative to the line above, not after it.

or stage by stage (`COCO=data/coco-mini`):

```bash
# 0. Where does the model break?  Run this before anything else.
python scripts/00_precision_ladder.py --coco-root $COCO \
    --ladder fp16,w8a8,w4a16,w4a8,w4a4 --n-pope-images 60 --n-chair-images 40

# 1. Layer-depth drift -> which layers LoRAS should touch
python scripts/01_profile_drift.py --coco-root $COCO --precision w4a8 --n-images 32

# 2. Calibrate LoRAS (closed form, sequential)
python scripts/02_calibrate_loras.py --coco-root $COCO --precision w4a8 \
    --drift-json runs/drift__w4a8/drift.json --n-calib 256 --rank 16

# 3. The 2x2 ablation
python scripts/03_eval.py --coco-root $COCO --precision w4a8 --tag base  --ppl
python scripts/03_eval.py --coco-root $COCO --precision w4a8 --tag loras --ppl \
    --loras runs/loras__w4a8/loras.pt
python scripts/03_eval.py --coco-root $COCO --precision w4a8 --tag acab  --ppl \
    --acab add --alpha 1.0 --tau-percentile 60
python scripts/03_eval.py --coco-root $COCO --precision w4a8 --tag both  --ppl \
    --loras runs/loras__w4a8/loras.pt --acab add --alpha 1.0 --tau-percentile 60

# 4-6. sweep, latency, figures
python scripts/04_acab_sweep.py --coco-root $COCO --precision w4a8 \
    --loras runs/loras__w4a8/loras.pt --acab add --with-pope
python scripts/05_latency.py    --coco-root $COCO --precision w4a8 \
    --loras runs/loras__w4a8/loras.pt
python scripts/06_figures.py --runs runs --out figures
```

Rough costs on one A6000 at the defaults: ladder ≈ 90 min (5 precisions), drift ≈ 5
min, calibration ≈ 15 min, each eval cell ≈ 25 min, sweep ≈ 2 h, latency ≈ 10 min.
Scale up with `--n-pope-images 500 --n-chair-images 500` for the full protocol once
an effect is confirmed.

---

### Finding the fallback regime, and comparing runs

`00_precision_ladder.py` accepts any `w<bits>a<bits>` plus options, so you can put rungs
between "fine" and "broken" and try the standard fixes for activation outliers:

```bash
python scripts/00_precision_ladder.py --coco-root data/coco-mini --out runs/ladder_fine \
    --n-pope-images 100 --n-chair-images 60 \
    --ladder "fp16,w4a8,w4a6,w4a5,w4a4,w4a4:clip=0.999,w4a4:skip=down_proj,\
w4a8:targets=language+projector,w4a8:targets=language+projector+vision,w3a16,w2a16"
```

Each rung is compared with the first one using a paired, image-clustered bootstrap,
and gets one of three verdicts: `ok`, `degraded` (significantly worse but still
fluent: the regime this project is about), or `BROKEN` (unparseable answers, empty
or looping captions). A BROKEN rung's low CHAIR is not a result. The report ends by
naming the degraded rungs to use as `PREC`. `--resume` skips rungs that are already
done.

To compare evaluation runs pairwise, with confidence intervals:

```bash
python scripts/07_compare.py runs/eval__w4a5_base runs/eval__w4a5_{loras,acab,both}
```

### Is the collapse visual? (the token-selective test)

```bash
python scripts/08_token_probe.py --coco-root data/coco-mini --precisions fp16,w4a8,w4a4
python scripts/00_precision_ladder.py --coco-root data/coco-mini --out runs/ladder_tokens \
    --ladder "fp16,w4a8,w4a4,w4a4:abits_text=8,w4a4:abits_image=8"
```

`08_token_probe.py` measures, in fp16, the share of each token's entries that
per-token 4-bit rounding would set to zero, separately for image and text tokens,
layer by layer. It also checks whether the quantized model is fluent with no image
at all. The two extra ladder rungs quantize activations to 4 bits at image positions
only (`abits_text=8`) or at text positions only (`abits_image=8`). Whichever one
breaks the model is where the collapse comes from.

### 4.0 Gate 1: published PTQ methods (start here)

The paper's question: under the quantization people actually deploy, does grounding
degrade beyond the language model's own damage, and in which direction -- *fallback*
(more "yes" to absent objects, more invented objects) or *omission* (more misses,
emptier captions)? Two steps.

**Step 1, validate the methods (~1.5 h).** WikiText-2 perplexity of LLaVA's language
model for each method. Needs `pip install pyarrow`.

```bash
python scripts/09_validate_ptq.py --coco-root data/coco-mini 2>&1 | tee runs/validate.log
```

The checks at the end are relative (published Llama-2-7B fractions in brackets): at
W3, AWQ should close >= 25% of RTN's perplexity gap to FP16 [35%] and GPTQ >= 10%
[19%]; rotation should rescue most of W4A4; SmoothQuant W8A8 should sit within 0.2 of
FP16. Optionally run it on `meta-llama/Llama-2-7b-hf --calib wiki` to compare with the
AWQ paper's table directly.

**Step 2, the grid (~4-5 h at the sizes below).**

```bash
python scripts/00_precision_ladder.py --coco-root data/coco-mini --preset gate1 \
    --out runs/gate1 --n-pope-images 100 --n-chair-images 100 --resume \
    2>&1 | tee runs/gate1.log
```

`--preset gate1` (17 rungs) is: fp16; NF4; RTN, and AWQ/GPTQ calibrated on WikiText (the
toolchain default), at W4A16 and W3A16; at W3A16 also AWQ/GPTQ calibrated on image+caption
data (`calib=mm`) and on the same captions without images (`calib=text`); RTN and
SmoothQuant W8A8; SmoothQuant W4A8; rotation and rotation+GPTQ at W4A4.

Why calibration is a factor: on WikiText (validation run, 1 Oct), GPTQ at W3 calibrated
on image+caption data is worse than RTN (9.87 vs 9.14 PPL; fp16 7.27), while the same
GPTQ calibrated on WikiText closes 35% of RTN's gap (8.49). Which calibration protects
*grounding* is exactly what the grid measures. `gate1-weights` and
`gate1-acts` split it in two (two GPUs, or two nights). Add `bnb-nf4` to check the
simulated NF4 against the real bitsandbytes kernels (`pip install bitsandbytes`).

Every rung gets, against fp16, paired image-clustered CIs for: POPE F1, precision,
recall and yes-ratio (pooled over the three splits), CHAIR_i, coverage and caption
length, plus the no-image caption perplexity. The report gives each rung a
**direction** (fallback / omission / mixed / none) and says whether the language model
itself is damaged (no-image PPL > 1.10x). Gate 1 passes for any rung that is degraded
with the language model intact.

**Gate 2: is the omission real, visual and specific to quantization?**

```bash
python scripts/10_object_analysis.py --coco-root data/coco-mini --ladder runs/gate1/ladder.json
python scripts/00_precision_ladder.py --coco-root data/coco-mini --preset gate2 \
    --out runs/gate2 --n-pope-images 100 --n-chair-images 300 --resume 2>&1 | tee runs/gate2.log
python scripts/10_object_analysis.py --coco-root data/coco-mini --ladder runs/gate2/ladder.json
```

`10_object_analysis.py` (CPU, seconds) reports, per rung, the share of objects the fp16
caption mentions that the quantized caption drops, by object size (largest instance as
a fraction of the image) and by category. `gate2` adds two controls: `method=noise`
(Gaussian weight noise with RTN's per-group error energy, random direction) and
`wtok=image|text` (quantized weights only at image / only at text positions). The
ladder now also saves P(yes) per POPE question and reports a paired dAUROC, which
separates "sees less" (AUROC falls) from "answers yes at a different rate" (AUROC flat).

Spec reference (any rung, any script's `--precision`):

```
nf4                       bitsandbytes NF4 format, block 64 (simulated)
bnb-nf4                   real bitsandbytes kernels (ladder only)
w3a16:method=awq          AWQ scale + clip search            calib=mm|text|wiki, ncal=64
w4a16:method=gptq         GPTQ, true-sequential               actorder=1, damp=0.01
w8a8:method=sq            SmoothQuant                         alpha=0.85
w4a4:method=rot           random-Hadamard rotation (QuaRot-style), RTN weights
w4a4:method=rot+gptq      rotation + GPTQ (QuaRot's recipe)
w4a4:method=rot+gptq:abits_text=8    ... attribution arm: image tokens at A4 only
w3a16:method=noise        control: RTN-matched Gaussian weight noise
w3a16:wtok=image          quantized weights at image positions only (text: FP16)
```

## 5. Repo map

```
phoenix/
  quant.py       simulated PTQ ladder; FakeQuantLinear with an FP/quant toggle
  model.py       LLaVA-1.5 loading, prompt format, visual-token localisation
  data.py        COCO, POPE construction/loading, CHAIR caption set, calibration set
  fetch.py       mini-COCO builder: remote partial-zip reader, image fetch, splits
  env.py         GPU / driver / torch preflight used by check_env and load_model
  stats.py       paired image-clustered bootstrap for POPE and CHAIR differences
  probes.py      drift profiling, KV capture, visual attention mass, KL-to-blind
  loras.py       RRRStats (sufficient statistics + closed-form solver), correctors
  calibrate.py   sequential block-by-block calibration driver
  acab.py        additive/multiplicative bias, entropy gate, gated decode loop
  chair.py       CHAIR scorer with the 80-class synonym table and coverage
  metrics.py     POPE F1/yes-ratio/ECE, fluency guards, reference-caption PPL
  evaluate.py    POPE / CHAIR / latency runners
  ptq/           published PTQ methods on the fake-quant substrate:
    calib.py       calibration sequences: mm (image+caption), text (same captions, no
                   image, same token budget), wiki
    engine.py      layer-by-layer capture/replay shared by the methods
    transforms.py  SmoothQuant scaling, QuaRot-style rotation, scale folding
    awq.py         AWQ scale + clip search
    gptq.py        GPTQ (Cholesky form, lazy blocks, act-order with static groups)
scripts/         setup_env.sh (venv), check_env (preflight), fetch_data (mini-COCO), 00 ladder, 01 drift, 02 calibrate,
                 03 eval, 04 sweep, 05 latency, 06 figures, 07 compare, 08 token probe,
                 09 validate_ptq (WikiText-2 PPL per method)
tests/           test_smoke.py  (41 checks: solver, quant, LoRAS, A-CAB, decode loop)
                 test_data.py   (29 checks: subsetting, split enforcement, scoring)
                 test_ladder.py (16 checks: ladder end to end, verdicts, bootstrap)
                 test_fixes.py  (18 checks, tiny LLaVA with a CLIP tower: the run's three bugs,
                                 token-selective quantization, the token probe)
                 test_ptq.py    (PTQ methods: exactness of every transform, AWQ/GPTQ beat
                                 RTN on their objective, rotation/SmoothQuant cut outlier
                                 damage, NF4 code book, calibration sources)
                 both CPU-only, no GPU, no network, no model download
```

---

## 6. How to read the results

**LoRAS.** The number to look at first is `ceiling` in
`runs/loras__*/loras_diagnostics.json`.

* ceiling > 0.7 — quantization error at that site is largely a linear map. LoRAS
  should work; push rank until the curve flattens.
* ceiling 0.3–0.7 — partial. Report LoRAS as a cheap partial corrector and lean on
  the rank curve as the interesting artefact.
* ceiling < 0.25 — the error is essentially non-linear. **That is a publishable
  negative result**, and it redirects effort to C3 rather than burning weeks. Say so
  explicitly; it also explains why weight-space PTQ methods can't fix this.

Then check `val_red` against `rel_mse_reduction`. A large gap means 256 calibration
images are overfitting a 4096×4096 covariance — raise `--n-calib` or `--ridge`.

**A-CAB.** Read `fig4_acab_sweep.png`, not the CHAIR column. You want a point where
CHAIR_i falls while coverage and caption length hold. If the whole curve just slides
down-left, α is buying you brevity, not grounding. `gated_frac` tells you what
fraction of decode steps fired; if it's ~1.0 you've built a static bias and should
say so.

**The 2×2.** If LoRAS alone and A-CAB alone each help but "both" isn't better than
the best single one, that's the interesting finding — they're correcting the same
failure by different means, which is exactly the Lexical Fallback story. Report it;
don't bury it.

---

## 7. Known limitations (put these in the writeup)

* Plain `w4a4` is per-token RTN, a pessimistic bound; use `w4a4:method=rot+gptq` for
  the realistic W4A4 row. Our rotation is applied online to every quantized Linear's
  input (numerically what QuaRot quantizes for q/k/v and gate/up); it uses one shared
  random Hadamard per width (Hadamard x random orthogonal for 11008 = 256 x 43) rather
  than QuaRot's exact per-head scheme, and does not quantize the KV cache.
* AWQ/GPTQ/SmoothQuant are re-implementations, validated by `09_validate_ptq.py`, not
  the reference code. Known deliberate differences: GPTQ uses our min/max grid (not
  min(0,.)/max(0,.)), and calibration data is ours (COCO image+caption by default).
  MBQ is not implemented; compare against its released code or numbers.
* `scripts/fetch_data.py` installs the official POPE question files and the scripts
  prefer them automatically, so POPE numbers are the published questions on a subset
  of the published images. The local constructor (used only when no official file is
  present) follows POPE's sampling rules but is not the same question list — say
  which one you used. For the full published protocol use `--preset full`, which
  fetches all 500 POPE images.
* The CHAIR synonym table here is a faithful reproduction, not a byte-identical copy
  of `synonyms.txt`. Pass `--synonyms` to use the canonical file.
* LoRAS is calibrated on caption-style prompts and evaluated on both captions and
  POPE questions. Prompt-distribution shift between calibration and evaluation is
  itself worth an ablation — calibrate on POPE-style prompts and see if it transfers.
* `--gate prev` is a lag-1 approximation; `--gate two_pass` quantifies what the lag
  costs. Report both rather than only the cheap one.
* Only the language decoder is quantized by default (`--quant-targets language`),
  which is standard. `--quant-targets language,projector` is the more aggressive and
  arguably more interesting setting given the proposal's claims about visual token
  entropy — worth one row.

---

## 8. Verification

Five suites, all CPU-only and offline. Run them before touching the GPU:

```bash
python tests/test_smoke.py     # ~1 min, tiny random Llama
python tests/test_data.py      # ~5 s, synthetic COCO in a temp dir
python tests/test_ladder.py    # ladder end to end incl. PTQ rungs, verdicts, directions
python tests/test_fixes.py     # tiny LLaVA regressions
python tests/test_ptq.py       # published PTQ methods
```

`test_smoke.py` checks the machinery, not the science:

* closed-form RRR recovers a known rank-k map (1.00000), beats naive truncated SVD,
  and matches 3000 Adam steps to 5 decimals;
* held-out `evaluate()` agrees with the solver's own reduction number;
* the FP path through `FakeQuantLinear` is bit-identical to the unquantized model,
  and logit error increases monotonically down the ladder;
* LoRAS reduces **held-out** K/V error (0.374 → 0.067) and improves cosine to FP16
  (0.822 → 0.962) through a real forward pass;
* the additive bias shifts `logit(M_vis)` by exactly δ (error < 1e-6) and is
  monotone; head selection touches only the selected heads;
* the multiplicative variant is shown to be non-monotone — 3 of 4 heads *lose*
  visual mass at β=2.0;
* the gate fires only above τ and clips at δ_max; all three gate modes decode,
  including the two-pass cache crop;
* CHAIR's double-word handling ("hot dog" ≠ "dog") and POPE's F1/yes-ratio are
  checked against hand-computed values.

`test_data.py` builds a synthetic COCO and checks the data layer:

* annotation subsetting keeps exactly the selected images and drops every stray
  annotation, and the ijson-streaming and in-memory parsers agree exactly;
* `write_splits` refuses an overlapping partition, and with a manifest present
  `build_calibration_set` and `build_caption_set` return **disjoint** image sets;
* `get_pope` prefers the official question file, restricts it to the eval pool, and
  resolves every sample to an image that exists;
* a perfectly grounded caption scores CHAIR 0 / coverage 1.0, one injected object
  gives exactly CHAIR_i = 1/(n+1), and a one-object caption scores CHAIR_i = 0 with
  coverage 0.25 — the gaming failure mode, which is why both are always reported;
* `verify()` passes on a well-formed build and fails when an image is missing.

The remote partial-zip reader was validated against real archives (byte-identical
extraction from a 16 MB wheel, including forced 64 KB multi-chunk streaming), and the
POPE loader against the real `coco_pope_{random,popular,adversarial}.json`.
