from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel


def load_model_and_processor(model_path: str, device: torch.device):
    model = CLIPModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def _chunks(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _encode_image_batch(model, pixel_values: torch.Tensor) -> torch.Tensor:
    vision_out = model.vision_model(pixel_values=pixel_values)
    pooled = vision_out.pooler_output
    feats = model.visual_projection(pooled)
    return F.normalize(feats, dim=-1)


def _encode_text_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = text_out.pooler_output
    feats = model.text_projection(pooled)
    return F.normalize(feats, dim=-1)


def encode_images_from_paths(
    model,
    processor,
    image_paths: List[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_paths in _chunks(image_paths, batch_size):
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            batch = processor(images=images, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            feats = _encode_image_batch(model, pixel_values)
            outputs.append(feats.cpu())
    return torch.cat(outputs, dim=0)


def encode_images_from_pil(
    model,
    processor,
    images: List[Image.Image],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_images in _chunks(images, batch_size):
            batch = processor(images=batch_images, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            feats = _encode_image_batch(model, pixel_values)
            outputs.append(feats.cpu())
    return torch.cat(outputs, dim=0)


def encode_texts(
    model,
    processor,
    texts: List[str],
    batch_size: int,
    device: torch.device,
    max_length: int = 77,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_texts in _chunks(texts, batch_size):
            batch = processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            feats = _encode_text_batch(model, input_ids, attention_mask)
            outputs.append(feats.cpu())
    return torch.cat(outputs, dim=0)


def parse_ks(ks: str) -> List[int]:
    return [int(k.strip()) for k in ks.split(",") if k.strip()]


def ensure_exists(path: str, what: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}")
