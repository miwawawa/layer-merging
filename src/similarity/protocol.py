"""Similarity protocol: every 'how similar are layers?' measure implements this."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch


class LayerSimilarity(ABC):
    """Computes an (L, L) similarity matrix over L layers (0..1, higher = more redundant)."""

    @abstractmethod
    def compute(self, activations: List[torch.Tensor]) -> np.ndarray:
        """activations[l] = (n_samples, d_l). Returns symmetric S in [0,1]."""