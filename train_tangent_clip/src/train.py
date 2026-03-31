"""
python -m src.train --config configs/tangent_clip_vitb_controlled_mini.json
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

try:
    from src.data.controlled_pairs import ControlledPairsDataset
    from src.losses.clip_losses import compose_total_loss
    from src.modeling.clip_wrapper import build_clip_model, set_trainable_strategy
    from src.utils.config import load_config
    from src.utils.metrics import compute_recall_at_k
    from src.utils.seed import set_seed
except ModuleNotFoundError:
    # Fallback for running as: python src/train.py
    from data.controlled_pairs import ControlledPairsDataset
    from losses.clip_losses import compose_total_loss
    from modeling.clip_wrapper import build_clip_model, set_trainable_strategy
    from utils.config import load_config
    from utils.metrics import compute_recall_at_k
    from utils.seed import set_seed


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def seed_worker(worker_id: int) -> None:
    # Keep dataloader workers reproducible across runs.
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ControlledPairsCollator:
    def __init__(self, processor, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, samples: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        images = [s["image"] for s in samples]
        images_plus = [s["image_plus"] for s in samples]
        texts = [s["caption"] for s in samples]
        texts_plus = [s["caption_plus"] for s in samples]

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        batch_plus = self.processor(
            text=texts_plus,
            images=images_plus,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {"inputs": batch, "inputs_plus": batch_plus}


def build_dataloader(cfg: Dict, processor, split: str, seed: int) -> DataLoader:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    jsonl_path = data_cfg["train_jsonl"] if split == "train" else data_cfg["val_jsonl"]
    dataset = ControlledPairsDataset(
        jsonl_path=jsonl_path,
        dataset_root=data_cfg.get("dataset_root"),
        factors=data_cfg.get("factors"),
    )

    generator = torch.Generator()
    generator.manual_seed(seed + (0 if split == "train" else 1))

    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(split == "train"),
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=(split == "train"),
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=ControlledPairsCollator(
            processor=processor,
            max_length=data_cfg.get("max_text_length", 64),
        ),
    )


def evaluate(model, loader, device: torch.device, ks: List[int]) -> Dict[str, float]:
    model.eval()
    image_embeds_all: List[torch.Tensor] = []
    text_embeds_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            inputs = to_device(batch["inputs"], device)
            outputs = model(**inputs)
            image_embeds_all.append(outputs.image_embeds)
            text_embeds_all.append(outputs.text_embeds)

    image_embeds = torch.cat(image_embeds_all, dim=0)
    text_embeds = torch.cat(text_embeds_all, dim=0)
    return compute_recall_at_k(image_embeds=image_embeds, text_embeds=text_embeds, ks=ks)


def compute_model_selection_score(metrics: Dict[str, float], primary_metrics: List[str]) -> float:
    values = [metrics[k] for k in primary_metrics if k in metrics]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def save_checkpoint(model, optimizer, scheduler, epoch: int, best_score: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
        },
        path,
    )


def train(cfg: Dict) -> None:
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_clip_model(cfg["model"]["model_path"])
    set_trainable_strategy(
        model=model,
        strategy=cfg["model"].get("trainable_strategy", "head_only"),
        unfreeze_last_n=cfg["model"].get("unfreeze_last_n", 0),
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(cfg["model"]["model_path"])

    train_loader = build_dataloader(cfg, processor, split="train", seed=cfg["train"]["seed"])
    val_loader = build_dataloader(cfg, processor, split="val", seed=cfg["train"]["seed"])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"].get("weight_decay", 0.0),
        betas=tuple(cfg["optim"].get("betas", [0.9, 0.98])),
        eps=cfg["optim"].get("eps", 1e-6),
    )

    max_steps = cfg["train"]["epochs"] * len(train_loader)
    warmup_steps = int(max_steps * cfg["optim"].get("warmup_ratio", 0.0))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    global_step = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=True)

        running = {"loss": 0.0, "clip_loss": 0.0, "tan_loss": 0.0, "consistency_loss": 0.0, "reg_loss": 0.0}

        for batch in pbar:
            inputs = to_device(batch["inputs"], device)
            inputs_plus = to_device(batch["inputs_plus"], device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(**inputs)
                out_plus = model(**inputs_plus)
                loss_dict = compose_total_loss(
                    mode=cfg["loss"]["mode"],
                    image_embeds=out.image_embeds,
                    text_embeds=out.text_embeds,
                    image_embeds_plus=out_plus.image_embeds,
                    text_embeds_plus=out_plus.text_embeds,
                    logit_scale=model.logit_scale.exp(),
                    lambda_tan=cfg["loss"].get("lambda_tan", 0.0),
                    lambda_consistency=cfg["loss"].get("lambda_consistency", 0.0),
                    lambda_reg=cfg["loss"].get("lambda_reg", 0.0),
                )

            scaler.scale(loss_dict["loss"]).backward()
            if cfg["optim"].get("max_grad_norm", 0.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["optim"]["max_grad_norm"])

            scale_before = scaler.get_scale() if use_amp else None
            scaler.step(optimizer)
            scaler.update()
            if use_amp:
                # Skip scheduler stepping when optimizer step is skipped due to inf/nan grads.
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            else:
                scheduler.step()

            global_step += 1
            for k in running:
                running[k] += float(loss_dict[k].detach().item())

            if global_step % cfg["train"].get("log_every", 20) == 0:
                avg_loss = running["loss"] / cfg["train"].get("log_every", 20)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                for k in running:
                    running[k] = 0.0

        metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            ks=cfg["eval"].get("ks", [1, 5, 10]),
        )

        primary_metrics = cfg.get("eval", {}).get(
            "primary_metrics",
            ["i2t_r@1", "t2i_r@1", "i2t_r@5", "t2i_r@5"],
        )
        score = compute_model_selection_score(metrics, primary_metrics)

        print(f"\n[epoch {epoch}] eval metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  selection_score({','.join(primary_metrics)}): {score:.4f}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=best_score,
            path=output_dir / "last.pt",
        )

        if score > best_score:
            best_score = score
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_score,
                path=output_dir / "best.pt",
            )
            model.save_pretrained(output_dir / "best_model")
            processor.save_pretrained(output_dir / "best_model")

        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            row = {"epoch": epoch, "score": score, **metrics}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Training complete. Best score={best_score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP with tangent alignment loss")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tangent_clip_vitb_controlled_mini.json",
        help="Path to json config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config).raw

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)
    with open(Path(cfg["train"]["output_dir"]) / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    train(cfg)
