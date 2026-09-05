"""Merging protocol: every 'how do two layers fuse?' strategy implements this."""

from abc import ABC, abstractmethod

import torch


class MergeStrategy(ABC):
    """Computes a per-pair mixing weight and fuses two parameter tensors."""

    @abstractmethod
    def weight(self, similarity: float) -> float:
        """Map a pair's similarity S_ij in [0,1] to the mixing coef alpha."""

    @abstractmethod
    def fuse(self, w1: torch.Tensor, w2: torch.Tensor, alpha: float) -> torch.Tensor:
        """Return the fused parameters: alpha * w1 + (1 - alpha) * w2."""