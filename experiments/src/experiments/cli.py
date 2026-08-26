from __future__ import annotations

import argparse
import pathlib

from .analysis import export as export_mod
from .analysis import metrics as metrics_mod
from .analysis import plots as plots_mod
from .config import Config
from .experiment import flow
from .models.load import load_variant


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="experiments", description="Lexical fallback evidence study")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--variants", type=str, default=None, help="comma list, e.g. fp16,w8,w4")
    p.add_argument("--sample-images", type=int, default=None, help="resample images per benchmark (None = full set)")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    p.add_argument("--root", type=str, default=".", help="project root for data/checkpoints/results")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-quantize", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    p.add_argument("--no-attention", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    root = pathlib.Path(args.root)
    cfg = Config.from_yaml(root / args.config).resolve_paths(root)
    cfg.apply_overrides(
        variants=args.variants.split(",") if args.variants else None,
        sample_images=args.sample_images,
        device=args.device,
        run_probe=not args.skip_probe,
        capture_attention=not args.no_attention,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("[flow] preparing data (POPE, CHAIR, COCO subset, annotations)...")
        data = flow.prepare_data(cfg)
    else:
        data = None

    if not args.skip_quantize:
        print("[flow] preparing checkpoints (FP16 + GPTQ W8/W4)...")
        flow.prepare_quantized(cfg)

    reports = []
    cells_extra = {}
    cells_binned = {}
    for variant in cfg.variants:
        print(f"[flow] cell: {variant}")
        model, processor, device, _ = load_variant(cfg, variant)
        model.eval()
        if data is None:
            data = flow.prepare_data(cfg)
        cell = flow.run_cell(cfg, variant, model, processor, device, data)
        export_mod.write_cell(cell, cfg.output_dir)
        reports.append(export_mod.cell_report(cell))

        rows = metrics_mod.mention_level_table(cell["chair_records"])
        attn = [r["attention"] for r in rows if r["attention"] is not None]
        grounded = [r["grounded"] for r in rows if r["attention"] is not None]
        cells_extra[variant] = (attn, [1.0 if g else 0.0 for g in grounded])
        cells_binned[variant] = metrics_mod.binned_curve(attn, grounded)
        del model
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    ablation = {"cells": reports}
    export_mod.write_ablation(ablation, cfg.output_dir)
    export_mod.write_summary_md(ablation, cfg.output_dir)
    plots_mod.make_all(ablation, cells_extra, cells_binned, cfg.output_dir / "figures")
    print(f"[flow] done — artifacts in {cfg.output_dir}")


if __name__ == "__main__":
    main()