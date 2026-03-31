from typing import Dict

import torch
import torch.nn.functional as F


def contrastive_clip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Standard symmetric CLIP InfoNCE loss."""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()

    targets = torch.arange(image_embeds.size(0), device=image_embeds.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i + loss_t)


def paired_clip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_embeds_plus: torch.Tensor,
    text_embeds_plus: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Average CLIP loss across original and perturbed matched pairs."""
    base = contrastive_clip_loss(image_embeds, text_embeds, logit_scale)
    pert = contrastive_clip_loss(image_embeds_plus, text_embeds_plus, logit_scale)
    return 0.5 * (base + pert)


def tangent_alignment_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_embeds_plus: torch.Tensor,
    text_embeds_plus: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Align finite-difference directions: DeltaI with DeltaT."""
    delta_i = image_embeds_plus - image_embeds
    delta_t = text_embeds_plus - text_embeds

    cos = F.cosine_similarity(delta_i, delta_t, dim=-1, eps=eps)
    return (1.0 - cos).mean()


def augmentation_consistency_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_embeds_plus: torch.Tensor,
    text_embeds_plus: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """A generic within-modality consistency baseline (alternative explanation)."""
    img_cos = F.cosine_similarity(image_embeds, image_embeds_plus, dim=-1, eps=eps)
    txt_cos = F.cosine_similarity(text_embeds, text_embeds_plus, dim=-1, eps=eps)
    return 0.5 * ((1.0 - img_cos).mean() + (1.0 - txt_cos).mean())


def delta_norm_regularizer(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_embeds_plus: torch.Tensor,
    text_embeds_plus: torch.Tensor,
) -> torch.Tensor:
    """Weakly regularize delta magnitudes to avoid degenerate large differences."""
    delta_i = image_embeds_plus - image_embeds
    delta_t = text_embeds_plus - text_embeds
    return 0.5 * (delta_i.norm(dim=-1).mean() + delta_t.norm(dim=-1).mean())


def compose_total_loss(
    mode: str,
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_embeds_plus: torch.Tensor,
    text_embeds_plus: torch.Tensor,
    logit_scale: torch.Tensor,
    lambda_tan: float,
    lambda_consistency: float,
    lambda_reg: float,
) -> Dict[str, torch.Tensor]:
    clip_loss = paired_clip_loss(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        image_embeds_plus=image_embeds_plus,
        text_embeds_plus=text_embeds_plus,
        logit_scale=logit_scale,
    )

    tan_loss = tangent_alignment_loss(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        image_embeds_plus=image_embeds_plus,
        text_embeds_plus=text_embeds_plus,
    )

    consistency_loss = augmentation_consistency_loss(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        image_embeds_plus=image_embeds_plus,
        text_embeds_plus=text_embeds_plus,
    )

    reg_loss = delta_norm_regularizer(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        image_embeds_plus=image_embeds_plus,
        text_embeds_plus=text_embeds_plus,
    )

    if mode == "clip_only":
        total = clip_loss
    elif mode == "clip_tangent":
        total = clip_loss + lambda_tan * tan_loss + lambda_reg * reg_loss
    elif mode == "clip_consistency":
        total = clip_loss + lambda_consistency * consistency_loss + lambda_reg * reg_loss
    else:
        raise ValueError(
            f"Unknown mode={mode}. Expected one of: clip_only, clip_tangent, clip_consistency"
        )

    return {
        "loss": total,
        "clip_loss": clip_loss,
        "tan_loss": tan_loss,
        "consistency_loss": consistency_loss,
        "reg_loss": reg_loss,
    }
