from __future__ import annotations

import json
import pathlib

import requests

from ..config import Config
from ..data import coco as coco_mod
from ..data.datasets import (
    load_pope_questions,
    resample_chair_images,
    resample_pope,
    base_seed_for,
)
from ..models import quantize as quant_mod
from ..models.download import download_checkpoint, resolve_device
from . import chair as chair_mod
from . import pope as pope_mod
from . import text_only as text_only_mod

POPE_URL = (
    "https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/master/"
    "experiments/data/POPE/coco/coco_pope_{split}.json"
)


def download_pope_questions(data_dir: pathlib.Path) -> None:
    out_dir = data_dir / "pope"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("random", "popular", "adversarial"):
        dest = out_dir / f"coco_pope_{split}.json"
        if dest.exists():
            continue
        url = POPE_URL.format(split=split)
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                dest.write_bytes(r.content)
                break
            except (requests.RequestException, OSError) as e:
                if attempt == 2:
                    raise
                print(f"[download] retry {attempt + 1}/3 for {url}: {e}")
                __import__("time").sleep(2 ** attempt)


def prepare_data(cfg: Config, allow_download: bool = True) -> dict:
    pope_dir = cfg.data_dir / "pope"
    has_pope = all((pope_dir / f"coco_pope_{s}.json").exists() for s in ("random", "popular", "adversarial"))
    ann_dir = cfg.data_dir / "annotations"
    has_ann = all((ann_dir / n).exists() for n in ("instances_val2014.json", "captions_val2014.json", "captions_train2014.json"))
    if not (allow_download or (has_pope and has_ann)):
        raise FileNotFoundError(
            "data missing and --skip-download was passed; run once without --skip-download to populate data/"
        )

    ann_paths = coco_mod.download_annotations(ann_dir)
    instances = coco_mod.load_instances(ann_paths["instances_val2014.json"])
    captions = coco_mod.load_captions(ann_paths["captions_val2014.json"])
    download_pope_questions(cfg.data_dir)

    pope_splits = load_pope_questions(cfg.data_dir)
    pope_images = sorted({q["image"] for q in pope_splits["adversarial"]})

    chair_dir = cfg.data_dir / "chair"
    chair_dir.mkdir(parents=True, exist_ok=True)
    chair_images_path = chair_dir / "chair_images.json"
    if not chair_images_path.exists():
        chair_ids = coco_mod.sample_val2014_filenames(instances, 500, cfg.seed)
        chair_images = [{"image": f, "image_id": coco_mod.filename_to_image_id(f)} for f in chair_ids]
        chair_images_path.write_text(json.dumps(chair_images, indent=2))
    else:
        chair_images = json.loads(chair_images_path.read_text())

    needed = set(pope_images) | {c["image"] for c in chair_images}
    image_dir = cfg.data_dir / "val2014"
    coco_mod.download_images(sorted(needed), "val2014", image_dir, cfg.download_workers)

    return {
        "instances": instances,
        "captions": captions,
        "pope": pope_splits,
        "chair_images": chair_images,
        "image_dir": image_dir,
    }


def prepare_quantized(cfg: Config) -> None:
    base_dir = cfg.checkpoints_dir / "base"
    if not (base_dir / "config.json").exists():
        download_checkpoint(cfg.model_id, base_dir)

    ann = coco_mod.download_annotations(cfg.data_dir / "annotations")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    calib = quant_mod.build_calibration(
        tokenizer,
        cfg.calibration_source,
        ann["captions_train2014.json"],
        cfg.calibration_samples,
        cfg.gptq_seq_len,
        cfg.seed,
    )

    out_dir = cfg.checkpoints_dir / "gptq-llm-w4"
    if not (out_dir / "config.json").exists():
        tmp_llm = cfg.checkpoints_dir / "llm-extracted"
        quant_mod.extract_llm(base_dir, tmp_llm, resolve_device(cfg.device))
        quant_mod.quantize_llm(
            tmp_llm,
            out_dir,
            4,
            calib,
            resolve_device(cfg.device),
            group_size=cfg.gptq_group_size,
            damp_percent=cfg.gptq_damp_percent,
            desc_act=cfg.gptq_desc_act,
        )
        import shutil

        shutil.rmtree(tmp_llm, ignore_errors=True)
        print("[flow] removed extracted LLM temp checkpoint")

    if not cfg.keep_base_checkpoint:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)
        print("[flow] removed base checkpoint (keep_base_checkpoint=false); fp16/w8 load from hub")


def _resampled_questions(pope: dict[str, list[dict]], cfg: Config) -> dict[str, list[dict]]:
    return {
        s: resample_pope(qs, cfg.sample_images, cfg.seed)
        for s, qs in pope.items()
    }


def run_cell(cfg: Config, variant: str, model, processor, device, data) -> dict:
    from ..analysis.metrics import (
        attention_summary_from_records,
        chair_metrics,
        kl_summary,
        pope_metrics,
    )

    pope_rs = {}
    vdir = cfg.output_dir / variant
    for split, qs in _resampled_questions(data["pope"], cfg).items():
        out_path = vdir / f"pope_{split}.jsonl"
        rs = pope_mod.run_pope(
            model, processor, processor.tokenizer, qs, cfg, data["image_dir"], out_path=out_path
        )
        pope_rs[split] = rs

    chair_items = resample_chair_images(data["chair_images"], cfg.sample_images, cfg.seed)
    tokenizer = processor.tokenizer
    chair_rs = chair_mod.run_chair(
        model, processor, tokenizer, chair_items, cfg, data["image_dir"], data["instances"],
        data["captions"], out_path=vdir / "chair_captions.jsonl",
    )

    cell = {
        "variant": variant,
        "pope": {s: pope_metrics(rs) for s, rs in pope_rs.items()},
        "pope_records": pope_rs,
        "chair": chair_metrics(chair_rs),
        "chair_records": chair_rs,
        "attention": attention_summary_from_records(pope_rs, chair_rs),
        "prior": None,
        "dkl": None,
    }

    if cfg.run_probe:
        prior = {}
        for split, qs in _resampled_questions(data["pope"], cfg).items():
            prior[split] = text_only_mod.probe_prior_pope(model, processor, tokenizer, qs, cfg)
        cell["prior"] = prior

        pope_qs = _resampled_questions(data["pope"], cfg)
        chair_sub = chair_items[: cfg.probe_chair_subset]
        items = (
            [{"text": q["text"], "image_name": q["image"]} for q in pope_qs["adversarial"]]
            + [{"text": "Please describe this image in detail.", "image_name": c["image"]} for c in chair_sub]
        )
        cell["dkl"] = kl_summary(
            text_only_mod.run_contrastive_dkl(
                model, processor, tokenizer, items, cfg, cfg.max_new_tokens_pope, data["image_dir"]
            )
        )

    return cell