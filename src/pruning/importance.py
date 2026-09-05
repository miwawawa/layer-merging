"""Importance scoring for pruning (paper-3: attribution-based pruning).

SparC3 aggregates per-weight relevance over reference samples (Eq. 1):
    R_bar_psi_k = (1/n_ref) * sum_x R_psi_k(x)
then prunes the least-relevant `q` elements. We expose two scorers:
    LRPImportance      — R_{i<-j} attributions from LRP (the paper's method)
    MagnitudeImportance — average |w| baseline (no reference data needed)
Both return a per-layer list aligned with each layer's flattened parameters.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import BaseNN
from pruning.lrp import LRPAnalyzer


def _concat_flat(weight: nn.Parameter, bias: nn.Parameter) -> Tuple[np.ndarray, int]:
    flat = np.concatenate([weight.reshape(-1).detach().numpy(),
                           bias.reshape(-1).detach().numpy()])
    return flat, weight.numel()


class Importance(ABC):
    @abstractmethod
    def score(self, model: BaseNN, loader: DataLoader, n_samples: int) -> List[np.ndarray]:
        """Per-layer importance over concatenated (weight, bias) parameters."""


class LRPImportance(Importance):
    def __init__(self, eps: float = 1e-2):
        self.eps = eps

    def score(self, model: BaseNN, loader: DataLoader, n_samples: int) -> List[np.ndarray]:
        analyzer = LRPAnalyzer(model, eps=self.eps)
        layers: List[nn.Linear] = list(model.linears) + [model.output]
        importance = [np.zeros(np.sum([p.numel() for p in (l.weight, l.bias)])) for l in layers]
        seen = 0
        with torch.no_grad():
            for x, y in loader:
                if x.dim() == 4:
                    x = x.view(x.size(0), -1)
                rel = analyzer.weight_relevance(x, y, aggregate=True)  # per hidden layer
                for i, layer in enumerate(layers):
                    _, n_w = _concat_flat(layer.weight, layer.bias)
                    if i < len(rel):
                        # SparC3 ranks weights by the LRP attribution itself.
                        score = np.abs(rel[i].numpy().reshape(-1))
                        score = np.concatenate([score, np.zeros(n_w - score.size)])
                    else:  # output layer: magnitude fallback
                        base, _ = _concat_flat(layer.weight, layer.bias)
                        score = np.abs(base)
                    importance[i][: score.size] += score
                seen += x.size(0)
                if seen >= n_samples:
                    break
        return [imp / max(seen, 1) for imp in importance]


class MagnitudeImportance(Importance):
    def score(self, model: BaseNN, loader: DataLoader, n_samples: int) -> List[np.ndarray]:
        out = []
        for layer in list(model.linears) + [model.output]:
            _, n_w = _concat_flat(layer.weight, layer.bias)
            out.append(np.abs(_concat_flat(layer.weight, layer.bias)[0]))
        return out