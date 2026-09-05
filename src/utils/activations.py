"""Extract per-layer activations from a trained model on a calibration loader."""

from typing import List

import torch
from torch.utils.data import DataLoader

from models.base import BaseNN


def extract_activations(
    model: BaseNN, loader: DataLoader, max_samples: int = 512
) -> List[torch.Tensor]:
    """Collect activations at *every* hidden layer for up to `max_samples`.

    Returns a list ``acts`` where ``acts[l]`` is a tensor of shape
    ``(n, hidden_dim_l)``: what layer ``l`` produces for each sample.
    """
    model.eval()
    per_layer: List[List[torch.Tensor]] = [[] for _ in range(model.num_layers)]

    with torch.no_grad():
        for images, _ in loader:
            if images.dim() == 4:
                images = images.view(images.size(0), -1)
            _, acts = model.forward_with_activations(images)
            for l, a in enumerate(acts):
                per_layer[l].append(a.detach().cpu())
            if loader.batch_size == 1 or sum(len(x) for x in per_layer[0]) >= max_samples:
                break

    result = []
    for l in range(model.num_layers):
        full = torch.cat(per_layer[l], dim=0)
        result.append(full[:max_samples])
    return result