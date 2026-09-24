"""Per-rung pre-flight for the precision ladder.

Two modes:

  --config-only (default, no GPU/model download needed)
      Builds each variant's quantization config object and checks the
      required library versions are importable. Catches typos, missing
      deps (torchao/optimum-quanto), and version mismatches (e.g. torchao
      < 0.4) before a multi-hour run even starts.

  --load
      Actually loads each variant (needs the model on disk/hub + a GPU for
      anything beyond fp16 in reasonable time) and runs one generation step
      with output_attentions=True, to confirm attention capture survives
      every quantizer -- the thing decoding.py's AttentionTracker depends on.

Usage:
    uv run python scripts/preflight_variants.py --root .
    uv run python scripts/preflight_variants.py --root . --load --variants w8a8,w4a8
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

ALL_VARIANTS = ["fp16", "w8a8", "w4a16", "w4a8", "w4a4"]
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS" if cond else "FAIL"), name, f"- {detail}" if detail and not cond else "")
    if not cond:
        FAIL.append(name)


def config_only_checks() -> None:
    import importlib.metadata as md

    try:
        import torchao  # noqa: F401

        check("torchao importable", True)
        v = md.version("torchao")
        check("torchao >= 0.4.0", tuple(map(int, v.split(".")[:2])) >= (0, 4), f"found {v}")
    except ImportError as e:
        check("torchao importable", False, str(e))

    try:
        import optimum.quanto  # noqa: F401

        check("optimum-quanto importable", True)
    except ImportError as e:
        check("optimum-quanto importable", False, str(e))

    from transformers import QuantoConfig, TorchAoConfig

    try:
        TorchAoConfig(
            "int8_dynamic_activation_int8_weight",
            modules_to_not_convert=["vision_tower", "multi_modal_projector"],
        )
        check("w8a8 config constructs", True)
    except Exception as e:  # noqa: BLE001
        check("w8a8 config constructs", False, str(e))

    try:
        QuantoConfig(weights="int4", activations="int8", modules_to_not_convert=["vision_tower"])
        check("w4a8 config constructs", True)
    except Exception as e:  # noqa: BLE001
        check("w4a8 config constructs", False, str(e))

    try:
        QuantoConfig(weights="int4", activations=None, modules_to_not_convert=["vision_tower"])
        check("w4a4 base (weight-only int4) config constructs", True)
    except Exception as e:  # noqa: BLE001
        check("w4a4 base (weight-only int4) config constructs", False, str(e))

    # This is the documented-impossible case -- confirm it still fails loudly
    # rather than silently, so a future library upgrade that *does* support it
    # doesn't go unnoticed (in which case the fake-quant hook could be dropped).
    try:
        QuantoConfig(weights="int4", activations="int4")
        check("QuantoConfig(activations='int4') still rejected", False, "expected ValueError, got none")
    except ValueError:
        check("QuantoConfig(activations='int4') still rejected", True)


def load_checks(cfg_path: str, root_path: pathlib.Path, variants: list[str]) -> None:
    import torch

    from experiments.config import Config
    from experiments.models.load import load_variant

    cfg = Config.from_yaml(root_path / cfg_path).resolve_paths(root_path)

    for variant in variants:
        try:
            model, processor, device, _ = load_variant(cfg, variant)
            model.eval()
            inputs = processor(text="USER: <image>\nDescribe. ASSISTANT:", return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=True,
                    output_attentions=True,
                )
            check(f"{variant}: loads", True)
            check(f"{variant}: output_attentions works", out.attentions is not None)
            del model, out
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            check(f"{variant}: loads", False, repr(e))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--root", default=".")
    p.add_argument("--load", action="store_true", help="actually load models (needs GPU/hub access)")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS))
    args = p.parse_args()

    root_path = pathlib.Path(args.root)
    config_only_checks()
    if args.load:
        load_checks(args.config, args.root, args.variants.split(","))

    print()
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
    print("ALL PREFLIGHT CHECKS PASSED")
