"""LayerMerger: orchestrates similarity + strategy to compress a BaseNN.

Facade: the only merging API the experiments touch. Owns the greedy pair
selection (most-similar compatible pair), the fusion, and the model rewire.

Greedy loop (paper-1 Algorithm 1, adapts CoM's "don't trust positional
pairing" lesson):
  1. compute similarity matrix S over all layers
  2. pick argmax S over pairs of *compatible* (same-shape) layers
  3. fuse into one layer, shrink the model
  4. repeat until no pair beats the threshold
"""

from typing import List, Tuple

import numpy as np
import torch

from merging.protocol import MergeStrategy
from merging.weighted import WeightedMerge
from config.default import Config
from models.base import BaseNN
from similarity.protocol import LayerSimilarity


class LayerMerger:
    def __init__(self, model: BaseNN, similarity: LayerSimilarity,
                 strategy: MergeStrategy, threshold: float | None = None):
        # Similarity above which a compatible pair is worth fusing. The value
        # is not hardcoded here: None falls back to the single configured
        # ``Config.merge_threshold``.
        self.model = model
        self.similarity = similarity
        self.strategy = strategy
        self.threshold = (Config().merge_threshold if threshold is None else threshold)
        self.py_params_before = None
        self.py_params_after = None
        #: Diagnostic record of the most recent merge (None before any merge):
        #: ``{"pair": (i, j), "similarity": s, "alpha": alpha, "params_out": int}``.
        self.last_merge: dict | None = None

    # -- public API -----------------------------------------------------
    def merge_once(self) -> bool:
        """Merge the single most-similar compatible pair. Returns True if any."""
        S = self._similarity_matrix()
        i, j, s = self._best_pair(S)
        if i is None or s < self.threshold:
            return False
        alpha = self._alpha(S, i, j)
        self._fuse(i, j, alpha)
        self.last_merge = {"pair": (i, j), "similarity": round(float(s), 6),
                           "alpha": round(float(alpha), 6),
                           "params_out": sum(p.numel() for p in self.model.parameters())}
        return True

    def merge_all(self) -> List[int]:
        """Keep merging until no compatible pair remains. Returns # merges."""
        n = 0
        while self.merge_once():
            n += 1
        return n

    # -- internals ----------------------------------------------------------
    def _similarity_matrix(self) -> np.ndarray:
        from utils.activations import extract_activations
        acts = extract_activations(self.model, self._calibration_loader,
                                   self._n_calibration)
        return self.similarity.compute(acts)

    def _best_pair(self, S: np.ndarray) -> Tuple[int | None, int | None, float]:
        best_i = best_j = None
        best_s = -1.0
        n = self.model.num_layers
        for i in range(n):
            for j in range(i + 1, n):
                if not self._compatible(i, j):
                    continue
                if S[i, j] > best_s:
                    best_i, best_j, best_s = i, j, float(S[i, j])
        return best_i, best_j, best_s

    def _compatible(self, i: int, j: int) -> bool:
        return (self.model.linears[i].weight.shape
                == self.model.linears[j].weight.shape)

    def _alpha(self, S: np.ndarray, i: int, j: int) -> float:
        s_row = S[i]  # context for optional softmax weighting
        if isinstance(self.strategy, WeightedMerge):
            self.strategy.set_context(s_row)
        return self.strategy.weight(S[i, j])

    def _fuse(self, i: int, j: int, alpha: float) -> None:
        w1 = self.model.linears[i].weight.detach().clone()
        w2 = self.model.linears[j].weight.detach().clone()
        b1 = self.model.linears[i].bias.detach().clone()
        b2 = self.model.linears[j].bias.detach().clone()
        merged_w = self.strategy.fuse(w1, w2, alpha)
        merged_b = self.strategy.fuse(b1, b2, alpha)
        self.model.remove_linears([i, j])          # j > i, safe order
        layer = torch.nn.Linear(merged_w.shape[1], merged_w.shape[0])
        with torch.no_grad():
            layer.weight.copy_(merged_w)
            layer.bias.copy_(merged_b)
        self.model.insert_linear(min(i, j), layer)

    # -- calibration ----------- (set by experiments, kept DRY) ----------
    def set_calibration(self, loader, n_samples: int = 512) -> "LayerMerger":
        self._calibration_loader = loader
        self._n_calibration = n_samples
        return self