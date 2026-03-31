from typing import Dict, Iterable, List

import torch


def compute_retrieval_recall(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    img_to_texts: List[List[int]],
    text_to_img: List[int],
    ks: Iterable[int],
    chunk_size: int = 512,
) -> Dict[str, float]:
    ks = sorted(set(int(k) for k in ks))
    max_k = max(ks)

    num_images = image_embeds.size(0)
    num_texts = text_embeds.size(0)

    i2t_hits = {k: 0 for k in ks}
    for start in range(0, num_images, chunk_size):
        end = min(start + chunk_size, num_images)
        sims = image_embeds[start:end] @ text_embeds.t()
        top_idx = sims.topk(k=min(max_k, num_texts), dim=1).indices
        for row in range(end - start):
            positives = set(img_to_texts[start + row])
            row_top = top_idx[row]
            for k in ks:
                if any(int(idx) in positives for idx in row_top[:k].tolist()):
                    i2t_hits[k] += 1

    t2i_hits = {k: 0 for k in ks}
    for start in range(0, num_texts, chunk_size):
        end = min(start + chunk_size, num_texts)
        sims = text_embeds[start:end] @ image_embeds.t()
        top_idx = sims.topk(k=min(max_k, num_images), dim=1).indices
        for row in range(end - start):
            positive = int(text_to_img[start + row])
            row_top = top_idx[row]
            for k in ks:
                if positive in row_top[:k].tolist():
                    t2i_hits[k] += 1

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"i2t_r@{k}"] = i2t_hits[k] / max(1, num_images)
        metrics[f"t2i_r@{k}"] = t2i_hits[k] / max(1, num_texts)
    return metrics


def compute_winoground_scores(image_embeds_2x2: torch.Tensor, text_embeds_2x2: torch.Tensor) -> Dict[str, bool]:
    # score[r, c] = sim(image_r, caption_c)
    score = image_embeds_2x2 @ text_embeds_2x2.t()
    s00 = float(score[0, 0].item())
    s01 = float(score[0, 1].item())
    s10 = float(score[1, 0].item())
    s11 = float(score[1, 1].item())

    text_correct = (s00 > s10) and (s11 > s01)
    image_correct = (s00 > s01) and (s11 > s10)
    group_correct = text_correct and image_correct
    return {
        "text_correct": text_correct,
        "image_correct": image_correct,
        "group_correct": group_correct,
    }


def compute_pairwise_preference_accuracy(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> Dict[str, float]:
    wins = (pos_scores > neg_scores).float()
    ties = (pos_scores == neg_scores).float()
    return {
        "pair_acc": float((wins + 0.5 * ties).mean().item()),
        "avg_margin": float((pos_scores - neg_scores).mean().item()),
    }
