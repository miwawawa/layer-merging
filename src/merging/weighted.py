"""WeightedMerge: the MKA (paper-1) adaptive mixing.

The merge coefficient for pair (l, m) equals the pair's similarity:
  alpha = S_lm   (clamped to keep both layers' contribution non-trivial)
Optionally a softmax across candidate mates can be requested (paper mentions
it as an alternative). Fusion is a convex combination of parameters:
  theta_c = alpha * theta_l + (1 - alpha) * theta_m
"""

import numpy as np
import torch

from merging.protocol import MergeStrategy


class WeightedMerge(MergeStrategy):
    def __init__(self, min_alpha: float = 0.1, max_alpha: float = 0.9,
                 softmax: bool = False):
        self.min_alpha, self.max_alpha = min_alpha, max_alpha
        self.softmax = softmax
        self._row: np.ndarray | None = None   # S row of the chosen layer pair

    def weight(self, similarity: float) -> float:
        if self.softmax and self._row is not None:
            exps = np.exp(self._row - self._row.max())
            alpha = float(exps[0] / exps.sum() if exps.sum() > 0 else 0.5)
        else:
            alpha = float(np.clip(similarity, self.min_alpha, self.max_alpha))
        return alpha

    def set_context(self, s_row: np.ndarray) -> None:
        """Remember the similarity row so softmax weighting can be computed."""
        self._row = s_row

    @staticmethod
    def fuse(w1: torch.Tensor, w2: torch.Tensor, alpha: float) -> torch.Tensor:
        return alpha * w1 + (1 - alpha) * w2