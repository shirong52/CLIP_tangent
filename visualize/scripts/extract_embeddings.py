#!/usr/bin/env python3
"""Extract CLIP embeddings for triplets.

Each triplet yields 3 image embeddings and 3 text embeddings:
  anchor         - A.orig image / caption
  floor_pert     - A.pert image / caption  (large-area: floor color changed)
  obj_color_pert - B.orig image / caption  (small-area: object colors changed)

All embeddings are L2-normalised (CLS token → visual projection for images;
EOS token → text projection for text).
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor


def load_clip_model(model_path: str):
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


@torch.no_grad()
def embed_image(model, path: str, transform, device) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    out = model.vision_model(t)
    cls = out.last_hidden_state[:, 0, :]
    proj = model.visual_projection(cls)
    emb = proj.cpu().numpy().squeeze()
    return emb / np.linalg.norm(emb)


@torch.no_grad()
def embed_text(model, processor, text: str, device) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    out = model.text_model(input_ids=ids, attention_mask=mask)
    eos_pos = mask.sum(dim=1) - 1
    eos = out.last_hidden_state[0, eos_pos[0], :]
    proj = model.text_projection(eos.unsqueeze(0))
    emb = proj.cpu().numpy().squeeze()
    return emb / np.linalg.norm(emb)


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP embeddings for triplets (anchor / floor_pert / obj_color_pert)."
    )
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/clip_vit_b")
    parser.add_argument("--matched_path", default="results/matched_samples.json")
    parser.add_argument(
        "--image_root",
        default="/root/autodl-tmp/dataset/composition/controlled_pairs",
    )
    parser.add_argument("--output_path", default="results/embeddings.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading CLIP from {args.model_path} ...")
    model, processor = load_clip_model(args.model_path)
    model = model.to(device)
    transform = get_image_transform()

    with open(args.matched_path) as f:
        triplets = json.load(f)
    print(f"Processing {len(triplets)} triplets ...")

    image_root = Path(args.image_root)
    # keys: anchor / floor_pert / obj_color_pert  ×  img / txt
    keys = [
        "anchor_img",       "floor_pert_img",       "obj_color_pert_img",
        "anchor_txt",       "floor_pert_txt",        "obj_color_pert_txt",
    ]
    buffers = {k: [] for k in keys}
    metadata = []

    for idx, t in enumerate(triplets):
        print(f"  Triplet {idx + 1}/{len(triplets)}", end="\r", flush=True)

        buffers["anchor_img"].append(
            embed_image(model, str(image_root / t["anchor_image"]), transform, device))
        buffers["floor_pert_img"].append(
            embed_image(model, str(image_root / t["floor_pert_image"]), transform, device))
        buffers["obj_color_pert_img"].append(
            embed_image(model, str(image_root / t["obj_color_pert_image"]), transform, device))

        buffers["anchor_txt"].append(
            embed_text(model, processor, t["anchor_caption"], device))
        buffers["floor_pert_txt"].append(
            embed_text(model, processor, t["floor_pert_caption"], device))
        buffers["obj_color_pert_txt"].append(
            embed_text(model, processor, t["obj_color_pert_caption"], device))

        metadata.append({
            "scene_key":      t["scene_key"],
            "floor_change":   t["floor_change"],
            "obj_color_change": t["obj_color_change"],
        })

    print()
    arrays = {k: np.stack(v, axis=0) for k, v in buffers.items()}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, metadata=np.array(metadata, dtype=object), **arrays)

    dim = arrays["anchor_img"].shape[1]
    print(f"Saved to {output_path}  |  triplets={len(triplets)}  dim={dim}")
    print(f"Arrays: {list(arrays.keys())}")


if __name__ == "__main__":
    main()
