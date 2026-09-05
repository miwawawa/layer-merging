"""CosineSimilarity: the cheap, classical baseline and the 'structural' signal
that CoM (paper-2) uses for layer matching.

Pairs are scored on activation matrices directly (no manifold step).
"""

from typing import List

import numpy as np
import torch

from similarity.protocol import LayerSimilarity


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened activation matrices."""
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class CosineSimilarity(LayerSimilarity):
    def compute(self, activations: List[torch.Tensor]) -> np.ndarray:
        n = len(activations)
        S = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = _cosine(activations[i].numpy(), activations[j].numpy())
                S[i, j] = S[j, i] = v
        return S