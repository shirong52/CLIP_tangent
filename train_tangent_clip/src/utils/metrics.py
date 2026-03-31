from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


def compute_recall_at_k(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    ks: Iterable[int],
) -> Dict[str, float]:
    """Compute I->T and T->I recall@k for aligned batches/datasets."""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    sim = image_embeds @ text_embeds.t()
    n = sim.size(0)
    targets = torch.arange(n, device=sim.device)

    metrics: Dict[str, float] = {}
    for k in ks:
        topk_i2t = sim.topk(k=min(k, n), dim=1).indices
        topk_t2i = sim.t().topk(k=min(k, n), dim=1).indices

        i2t_hits = (topk_i2t == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        t2i_hits = (topk_t2i == targets.unsqueeze(1)).any(dim=1).float().mean().item()

        metrics[f"i2t_r@{k}"] = i2t_hits
        metrics[f"t2i_r@{k}"] = t2i_hits

    return metrics


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    return {k: sum(m[k] for m in metric_list) / len(metric_list) for k in keys}
