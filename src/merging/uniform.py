"""UniformMerge: the trivial baseline — every pair is mixed 50/50."""

import torch

from merging.protocol import MergeStrategy


class UniformMerge(MergeStrategy):
    def weight(self, similarity: float) -> float:
        return 0.5

    @staticmethod
    def fuse(w1: torch.Tensor, w2: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        return alpha * w1 + (1 - alpha) * w2