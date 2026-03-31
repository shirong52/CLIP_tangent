import ast
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_controlled_pairs_for_retrieval(
    jsonl_path: str,
    dataset_root: str,
    factors: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[List[int]], List[int]]:
    path = Path(jsonl_path)
    root = Path(dataset_root)

    image_paths: List[str] = []
    captions: List[str] = []
    img_to_texts: List[List[int]] = []
    text_to_img: List[int] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if factors and item.get("factor") not in factors:
                continue

            image_abs = str((root / item["image"]).resolve())
            image_idx = len(image_paths)
            image_paths.append(image_abs)

            caption_idx = len(captions)
            captions.append(item["caption"])
            img_to_texts.append([caption_idx])
            text_to_img.append(image_idx)

    return image_paths, captions, img_to_texts, text_to_img


def load_coco_retrieval(
    captions_json: str,
    image_root: str,
) -> Tuple[List[str], List[str], List[List[int]], List[int]]:
    data = json.load(open(captions_json, "r", encoding="utf-8"))
    image_root = Path(image_root)

    id_to_file: Dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
    image_ids = sorted(id_to_file.keys())
    id_to_idx = {iid: i for i, iid in enumerate(image_ids)}

    image_paths = [str((image_root / id_to_file[iid]).resolve()) for iid in image_ids]
    img_to_texts: List[List[int]] = [[] for _ in image_paths]
    captions: List[str] = []
    text_to_img: List[int] = []

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in id_to_idx:
            continue
        img_idx = id_to_idx[image_id]
        txt_idx = len(captions)
        captions.append(ann["caption"].strip())
        text_to_img.append(img_idx)
        img_to_texts[img_idx].append(txt_idx)

    return image_paths, captions, img_to_texts, text_to_img


def load_flickr30k_retrieval(
    csv_path: str,
    image_root: str,
    split: str = "test",
) -> Tuple[List[str], List[str], List[List[int]], List[int]]:
    image_root = Path(image_root)

    image_paths: List[str] = []
    captions: List[str] = []
    img_to_texts: List[List[int]] = []
    text_to_img: List[int] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = row.get("split", "").strip().lower()
            if split != "all" and row_split != split.lower():
                continue

            filename = row["filename"].strip()
            raw_caps = ast.literal_eval(row["raw"])

            img_idx = len(image_paths)
            image_paths.append(str((image_root / filename).resolve()))
            cap_ids: List[int] = []
            for cap in raw_caps:
                txt_idx = len(captions)
                captions.append(str(cap).strip())
                text_to_img.append(img_idx)
                cap_ids.append(txt_idx)
            img_to_texts.append(cap_ids)

    return image_paths, captions, img_to_texts, text_to_img


def load_winoground_examples(examples_jsonl: str, max_samples: Optional[int] = None) -> List[Dict]:
    examples: List[Dict] = []
    with open(examples_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def load_sugarcrepe_samples(annotation_file: str, max_samples: Optional[int] = None) -> List[Dict]:
    path = Path(annotation_file)

    def _normalize(item: Dict) -> Dict:
        image = (
            item.get("image")
            or item.get("image_path")
            or item.get("filename")
            or item.get("img")
        )
        positive = (
            item.get("positive_caption")
            or item.get("pos_caption")
            or item.get("caption")
            or item.get("caption_true")
        )
        negative = (
            item.get("negative_caption")
            or item.get("neg_caption")
            or item.get("caption_false")
            or item.get("hard_negative_caption")
        )
        if not image or not positive or not negative:
            return {}
        return {"image": image, "positive_caption": positive, "negative_caption": negative}

    samples: List[Dict] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                item = json.loads(line)
                norm = _normalize(item)
                if norm:
                    samples.append(norm)
    elif path.suffix.lower() == ".json":
        data = json.load(path.open("r", encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", [])
        for i, item in enumerate(data):
            if max_samples is not None and i >= max_samples:
                break
            norm = _normalize(item)
            if norm:
                samples.append(norm)
    else:
        raise ValueError(f"Unsupported sugarcrepe annotation format: {path}")

    return samples
