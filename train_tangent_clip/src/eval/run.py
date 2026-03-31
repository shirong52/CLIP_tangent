"""
cd /root/CLIP_tangent/train_tangent_clip

# 1) 你的 controlled_pairs 测试集
python -m src.eval.run \
  --task controlled \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model

# 2) COCO
python -m src.eval.run \
  --task retrieval \
  --retrieval-dataset coco \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model

# 3) Flickr30k
python -m src.eval.run \
  --task retrieval \
  --retrieval-dataset flickr30k \
  --flickr-split test \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model

# 4) Winoground
python -m src.eval.run \
  --task winoground \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model

# 5) 全部一起（SugarCrepe 缺失会自动跳过）
python -m src.eval.run \
  --task all \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model \
  --output-json /root/autodl-tmp/output/tangent_vitb_controlled_mini/eval_all.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile

import torch
from PIL import Image

try:
    from src.eval.common import (
        encode_images_from_paths,
        encode_images_from_pil,
        encode_texts,
        ensure_exists,
        load_model_and_processor,
        parse_ks,
    )
    from src.eval.datasets import (
        load_coco_retrieval,
        load_controlled_pairs_for_retrieval,
        load_flickr30k_retrieval,
        load_sugarcrepe_samples,
        load_winoground_examples,
    )
    from src.eval.metrics import (
        compute_pairwise_preference_accuracy,
        compute_retrieval_recall,
        compute_winoground_scores,
    )
except ModuleNotFoundError:
    from eval.common import (
        encode_images_from_paths,
        encode_images_from_pil,
        encode_texts,
        ensure_exists,
        load_model_and_processor,
        parse_ks,
    )
    from eval.datasets import (
        load_coco_retrieval,
        load_controlled_pairs_for_retrieval,
        load_flickr30k_retrieval,
        load_sugarcrepe_samples,
        load_winoground_examples,
    )
    from eval.metrics import (
        compute_pairwise_preference_accuracy,
        compute_retrieval_recall,
        compute_winoground_scores,
    )


def _print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n[{title}]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def _save_json_if_needed(path: Optional[str], payload: Dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to: {out}")


def evaluate_controlled_pairs(args, model, processor, device: torch.device) -> Dict[str, float]:
    factors = [x.strip() for x in args.factors.split(",") if x.strip()] if args.factors else None
    image_paths, captions, img_to_texts, text_to_img = load_controlled_pairs_for_retrieval(
        jsonl_path=args.controlled_jsonl,
        dataset_root=args.controlled_root,
        factors=factors,
    )

    image_embeds = encode_images_from_paths(model, processor, image_paths, args.batch_size, device)
    text_embeds = encode_texts(model, processor, captions, args.batch_size, device, max_length=args.max_text_length)
    metrics = compute_retrieval_recall(image_embeds, text_embeds, img_to_texts, text_to_img, args.ks)
    metrics["num_images"] = float(len(image_paths))
    metrics["num_texts"] = float(len(captions))
    return metrics


def evaluate_retrieval(args, model, processor, device: torch.device) -> Dict[str, float]:
    if args.retrieval_dataset == "coco":
        image_paths, captions, img_to_texts, text_to_img = load_coco_retrieval(
            captions_json=args.coco_captions_json,
            image_root=args.coco_image_root,
        )
        dataset_name = "coco"
    else:
        image_paths, captions, img_to_texts, text_to_img = load_flickr30k_retrieval(
            csv_path=args.flickr_csv,
            image_root=args.flickr_image_root,
            split=args.flickr_split,
        )
        dataset_name = f"flickr30k_{args.flickr_split}"

    image_embeds = encode_images_from_paths(model, processor, image_paths, args.batch_size, device)
    text_embeds = encode_texts(model, processor, captions, args.batch_size, device, max_length=args.max_text_length)
    metrics = compute_retrieval_recall(image_embeds, text_embeds, img_to_texts, text_to_img, args.ks)
    metrics["num_images"] = float(len(image_paths))
    metrics["num_texts"] = float(len(captions))
    metrics["dataset"] = dataset_name
    return metrics


def _load_winoground_image(image_id: str, images_dir: Optional[str], images_zip: Optional[str], zip_handle: Optional[ZipFile]):
    candidates = [
        image_id,
        f"{image_id}.png",
        f"{image_id}.jpg",
        f"images/{image_id}.png",
        f"images/{image_id}.jpg",
    ]

    if images_dir:
        root = Path(images_dir)
        for rel in candidates:
            p = root / rel
            if p.exists():
                return Image.open(p).convert("RGB")

    if zip_handle is not None:
        names = set(zip_handle.namelist())
        for rel in candidates:
            if rel in names:
                with zip_handle.open(rel) as f:
                    return Image.open(f).convert("RGB")

    raise FileNotFoundError(f"Winoground image not found for id={image_id}")


def evaluate_winoground(args, model, processor, device: torch.device) -> Dict[str, float]:
    examples = load_winoground_examples(args.winoground_examples_jsonl, args.max_samples)

    zf = ZipFile(args.winoground_images_zip, "r") if args.winoground_images_zip else None
    text_correct = 0
    image_correct = 0
    group_correct = 0

    for ex in examples:
        img0 = _load_winoground_image(ex["image_0"], args.winoground_images_dir, args.winoground_images_zip, zf)
        img1 = _load_winoground_image(ex["image_1"], args.winoground_images_dir, args.winoground_images_zip, zf)

        image_embeds = encode_images_from_pil(model, processor, [img0, img1], batch_size=2, device=device)
        text_embeds = encode_texts(
            model,
            processor,
            [ex["caption_0"], ex["caption_1"]],
            batch_size=2,
            device=device,
            max_length=args.max_text_length,
        )
        out = compute_winoground_scores(image_embeds, text_embeds)
        text_correct += int(out["text_correct"])
        image_correct += int(out["image_correct"])
        group_correct += int(out["group_correct"])

    if zf is not None:
        zf.close()

    n = max(1, len(examples))
    return {
        "text_score": text_correct / n,
        "image_score": image_correct / n,
        "group_score": group_correct / n,
        "num_examples": float(len(examples)),
    }


def _discover_sugarcrepe_annotation(root: str) -> Optional[str]:
    root_path = Path(root)
    if root_path.is_file():
        return str(root_path)
    if not root_path.exists():
        return None

    candidates = []
    for suffix in ("*.jsonl", "*.json"):
        candidates.extend(sorted(root_path.rglob(suffix)))
    if not candidates:
        return None

    # Prefer test/dev-like files when available.
    preferred = [p for p in candidates if any(k in p.name.lower() for k in ["test", "val", "dev"])]
    return str((preferred[0] if preferred else candidates[0]).resolve())


def evaluate_sugarcrepe(args, model, processor, device: torch.device) -> Dict[str, float]:
    annotation = args.sugarcrepe_annotation or _discover_sugarcrepe_annotation(args.sugarcrepe_root)
    if not annotation:
        raise FileNotFoundError(
            "SugarCrepe annotation not found. Please provide --sugarcrepe-annotation or place json/jsonl under sugarcrepe root."
        )

    samples = load_sugarcrepe_samples(annotation, args.max_samples)
    if not samples:
        raise ValueError(f"No valid SugarCrepe samples found in {annotation}")

    image_root = Path(args.sugarcrepe_image_root) if args.sugarcrepe_image_root else Path(annotation).parent
    image_paths = [str((image_root / s["image"]).resolve()) for s in samples]
    pos_caps = [s["positive_caption"] for s in samples]
    neg_caps = [s["negative_caption"] for s in samples]

    image_embeds = encode_images_from_paths(model, processor, image_paths, args.batch_size, device)
    pos_embeds = encode_texts(model, processor, pos_caps, args.batch_size, device, max_length=args.max_text_length)
    neg_embeds = encode_texts(model, processor, neg_caps, args.batch_size, device, max_length=args.max_text_length)

    pos_scores = (image_embeds * pos_embeds).sum(dim=-1)
    neg_scores = (image_embeds * neg_embeds).sum(dim=-1)

    metrics = compute_pairwise_preference_accuracy(pos_scores, neg_scores)
    metrics["num_samples"] = float(len(samples))
    metrics["annotation_file"] = annotation
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone evaluation suite for CLIP_tangent project")

    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint folder (e.g., .../best_model)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-text-length", type=int, default=77)
    parser.add_argument("--ks", type=str, default="1,5,10", help="Comma-separated k values, e.g. 1,5,10")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick debugging")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output json path")

    parser.add_argument("--task", type=str, required=True, choices=["controlled", "retrieval", "winoground", "sugarcrepe", "all"])

    parser.add_argument("--controlled-jsonl", type=str, default="/root/autodl-tmp/dataset/composition/controlled_pairs/test.jsonl")
    parser.add_argument("--controlled-root", type=str, default="/root/autodl-tmp/dataset/composition/controlled_pairs")
    parser.add_argument("--factors", type=str, default="")

    parser.add_argument("--retrieval-dataset", type=str, default="coco", choices=["coco", "flickr30k"])
    parser.add_argument("--coco-captions-json", type=str, default="/root/autodl-tmp/dataset/retrival/coco/annotations/captions_val2017.json")
    parser.add_argument("--coco-image-root", type=str, default="/root/autodl-tmp/dataset/retrival/coco/val2017")
    parser.add_argument("--flickr-csv", type=str, default="/root/autodl-tmp/dataset/retrival/flickr30k/flickr_annotations_30k.csv")
    parser.add_argument("--flickr-image-root", type=str, default="/root/autodl-tmp/dataset/retrival/flickr30k/flickr30k-images")
    parser.add_argument("--flickr-split", type=str, default="test", choices=["train", "val", "test", "all"])

    parser.add_argument("--winoground-examples-jsonl", type=str, default="/root/autodl-tmp/dataset/composition/winoground/data/examples.jsonl")
    parser.add_argument("--winoground-images-dir", type=str, default="")
    parser.add_argument("--winoground-images-zip", type=str, default="/root/autodl-tmp/dataset/composition/winoground/data/images.zip")

    parser.add_argument("--sugarcrepe-root", type=str, default="/root/autodl-tmp/dataset/composition/sugarcrepe")
    parser.add_argument("--sugarcrepe-annotation", type=str, default="")
    parser.add_argument("--sugarcrepe-image-root", type=str, default="")
    parser.add_argument("--skip-missing-sugarcrepe", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.ks = parse_ks(args.ks)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model, processor = load_model_and_processor(args.model_path, device)

    all_metrics: Dict[str, Dict] = {}

    if args.task in ("controlled", "all"):
        ensure_exists(args.controlled_jsonl, "controlled jsonl")
        ensure_exists(args.controlled_root, "controlled dataset root")
        metrics = evaluate_controlled_pairs(args, model, processor, device)
        all_metrics["controlled"] = metrics
        _print_metrics("controlled_pairs", metrics)

    if args.task in ("retrieval", "all"):
        if args.task == "all":
            # In full benchmark mode, always evaluate both datasets.
            coco_args = argparse.Namespace(**vars(args))
            coco_args.retrieval_dataset = "coco"
            ensure_exists(coco_args.coco_captions_json, "COCO captions json")
            ensure_exists(coco_args.coco_image_root, "COCO image root")
            coco_metrics = evaluate_retrieval(coco_args, model, processor, device)
            all_metrics["retrieval_coco"] = coco_metrics
            _print_metrics("retrieval_coco", coco_metrics)

            flickr_args = argparse.Namespace(**vars(args))
            flickr_args.retrieval_dataset = "flickr30k"
            ensure_exists(flickr_args.flickr_csv, "Flickr CSV")
            ensure_exists(flickr_args.flickr_image_root, "Flickr image root")
            flickr_metrics = evaluate_retrieval(flickr_args, model, processor, device)
            all_metrics["retrieval_flickr30k"] = flickr_metrics
            _print_metrics("retrieval_flickr30k", flickr_metrics)
        else:
            if args.retrieval_dataset == "coco":
                ensure_exists(args.coco_captions_json, "COCO captions json")
                ensure_exists(args.coco_image_root, "COCO image root")
            else:
                ensure_exists(args.flickr_csv, "Flickr CSV")
                ensure_exists(args.flickr_image_root, "Flickr image root")
            metrics = evaluate_retrieval(args, model, processor, device)
            all_metrics[f"retrieval_{args.retrieval_dataset}"] = metrics
            _print_metrics(f"retrieval_{args.retrieval_dataset}", metrics)

    if args.task in ("winoground", "all"):
        ensure_exists(args.winoground_examples_jsonl, "Winoground examples jsonl")
        if args.winoground_images_dir:
            ensure_exists(args.winoground_images_dir, "Winoground images dir")
        elif args.winoground_images_zip:
            ensure_exists(args.winoground_images_zip, "Winoground images zip")
        else:
            raise ValueError("Provide either --winoground-images-dir or --winoground-images-zip")

        metrics = evaluate_winoground(args, model, processor, device)
        all_metrics["winoground"] = metrics
        _print_metrics("winoground", metrics)

    if args.task in ("sugarcrepe", "all"):
        try:
            metrics = evaluate_sugarcrepe(args, model, processor, device)
            all_metrics["sugarcrepe"] = metrics
            _print_metrics("sugarcrepe", metrics)
        except (FileNotFoundError, ValueError) as e:
            if args.task == "all" or args.skip_missing_sugarcrepe:
                print(f"\n[sugarcrepe] skipped: {e}")
            else:
                raise

    if not all_metrics:
        raise RuntimeError("No evaluation task ran. Check your --task and dataset paths.")

    _save_json_if_needed(args.output_json, all_metrics)


if __name__ == "__main__":
    main()
