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


def prune_mask(importance: np.ndarray, keep_ratio: float, row_wise: bool = False,
               shape: tuple | None = None) -> np.ndarray:
    """Return boolean mask: True = keep. Unstructured or per-output-rows.

    * unstructured: keep the ``keep_ratio`` *largest* scores in the flattened
      array (SparC3's global weight masking);
    * row-wise: scores are grouped by output neuron (rows of the weight matrix
      ``(out, in)``); a whole row survives if its summed score ranks in the top
      ``keep_ratio`` rows, so pruned units are removed completely (structured).

    ``shape`` is only needed when ``importance`` is flat (the per-layer arrays
    returned by the importance scorers) and ``row_wise=True``: it reshapes the
    scores to ``(out, in)`` before row-ranking.
    """
    if row_wise:
        rows, cols = (importance.shape if importance.ndim == 2
                      else shape if shape is not None else (importance.size, 1))
        scores = importance.reshape(rows, cols)
        row_scores = np.abs(scores).sum(axis=1)             # per-output-neuron score
        keep_rows = int(keep_ratio * rows) if keep_ratio < 1 else rows
        top = np.argsort(row_scores)[-keep_rows:]
        row_mask = np.zeros(rows, dtype=bool)
        row_mask[top] = True
        return np.repeat(row_mask, cols)                    # flat, aligned with weight
    keep = int(keep_ratio * importance.size) if keep_ratio < 1 else importance.size
    top = np.argsort(importance)[-keep:]
    mask = np.zeros(importance.size, dtype=bool)
    mask[top] = True
    return mask


def apply_prune(layer: torch.nn.Module, mask_flat: np.ndarray) -> int:
    """Zero out weights behind a flat mask; returns number of kept params.

    Only the *weights* are masked; biases are intentionally left untouched
    (biases are few, non-redundant, and SparC3's selection set is the weight
    tensors). Write goes through an explicit copy so the source tensor is
    never aliased in-place under autograd's view machinery.
    """
    w_size = layer.weight.numel()
    mask = mask_flat[:w_size]
    if mask.size < w_size:
        mask = np.pad(mask, (0, w_size - mask.size), constant_values=True)
    mask_t = torch.from_numpy(mask.astype(np.float32)).reshape_as(layer.weight)
    with torch.no_grad():
        layer.weight.data.copy_(layer.weight.detach() * mask_t)
    return int(mask.sum())


def prune_model(model: BaseNN, importances: List[np.ndarray], keep_ratio: float,
                row_wise: bool = False) -> int:
    """Apply masks to weight tensors. Returns number of parameters kept."""
    kept = 0
    for layer, imp in zip(list(model.linears) + [model.output], importances):
        n_w = layer.weight.numel()
        mask = prune_mask(imp[:n_w], keep_ratio, row_wise=row_wise,
                          shape=layer.weight.shape)
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