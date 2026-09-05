"""Pruner: turn importance scores into sparse models (paper-3, SparC3 Eq. 3).

    {c}_q = argsort(importance)[1..q]        least relevant q elements
    psi_i -> (1 - 1_{i in {c}_q}) * psi_i    zero them out

Supports unstructured (per-weight) and row-wise (per-output-neuron) sparsity,
each at a given keep-ratio. Importing stays side-effect free.
"""

from typing import List

import numpy as np
import torch

from models.base import BaseNN


def prune_mask(importance: np.ndarray, keep_ratio: float, row_wise: bool = False) -> np.ndarray:
    """Return boolean mask: True = keep. Unstructured or per-output-rows."""
    if row_wise:
        rows, cols = importance.reshape(-1, 1).shape
        row_scores = np.abs(importance.reshape(rows, cols))
        keep_rows = int(keep_ratio * rows) if keep_ratio < 1 else rows
        top = np.argsort(row_scores, axis=0)[-keep_rows:][:, 0]
        mask = np.zeros(rows, dtype=bool)
        mask[top] = True
        return np.repeat(mask, cols)
    keep = int(keep_ratio * importance.size) if keep_ratio < 1 else importance.size
    top = np.argsort(importance)[-keep:]
    mask = np.zeros(importance.size, dtype=bool)
    mask[top] = True
    return mask


def apply_prune(layer: torch.nn.Module, mask_flat: np.ndarray) -> int:
    """Zero out weights behind a flat mask; returns number of kept params.

    If the mask is shorter than the weight (defensive), unpadded positions
    are kept and logged as-is.
    """
    w_flat = layer.weight.reshape(-1)
    w_size = w_flat.numel()
    mask = mask_flat[:w_size]
    if mask.size < w_size:
        mask = np.pad(mask, (0, w_size - mask.size), constant_values=True)
    with torch.no_grad():
        w_flat.copy_(torch.from_numpy(mask.astype(np.float32)) * w_flat)
    return int(mask.sum())


def prune_model(model: BaseNN, importances: List[np.ndarray], keep_ratio: float,
                row_wise: bool = False) -> int:
    """Apply masks to weight tensors. Returns number of parameters kept."""
    kept = 0
    for layer, imp in zip(list(model.linears) + [model.output], importances):
        n_w = layer.weight.numel()
        mask = prune_mask(imp[:n_w], keep_ratio, row_wise=row_wise)
        kept += apply_prune(layer, mask)
    return kept


def prune_model_global(model: BaseNN, importances: List[np.ndarray],
                       keep_ratio: float) -> int:
    """SparC3-style global pruning: rank all weights together, zero the
    least-relevant `keep_ratio` fraction model-wide (biases left intact)."""
    layers = list(model.linears) + [model.output]
    sizes = [layer.weight.numel() for layer in layers]
    flat = np.concatenate([imp[:n] for imp, n in zip(importances, sizes)])
    keep_n = int(keep_ratio * flat.size) if keep_ratio < 1 else flat.size
    top = np.argsort(flat)[-keep_n:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[top] = True
    ptr = 0
    for layer, n in zip(layers, sizes):
        apply_prune(layer, mask[ptr:ptr + n])
        ptr += n
    return keep_n