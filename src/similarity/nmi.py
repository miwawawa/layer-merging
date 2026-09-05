"""NmiSimilarity: the paper-1 (MKA) layer-redundancy measure.

Pipeline: activations -> (per-layer) diffusion-map embedding -> Gaussian NMI
between every pair of embedded layers -> (L, L) similarity matrix.
"""

from typing import List

import numpy as np
import torch

from similarity.diffusion import DiffusionMap
from similarity.gaussian_mi import gaussian_nmi
from similarity.protocol import LayerSimilarity


class NmiSimilarity(LayerSimilarity):
    def __init__(self, sigma: float = 1.0, n_dims: int = 8, time: float = 1.0):
        self.dm = DiffusionMap(sigma=sigma, n_dims=n_dims, time=time)

    def compute(self, activations: List[torch.Tensor]) -> np.ndarray:
        n_layers = len(activations)
        emb = [a.numpy() for a in activations]
        emb = self.dm.embed_all(emb)                 # (n, k) per layer
        S = np.eye(n_layers)
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                v = gaussian_nmi(emb[i], emb[j])
                S[i, j] = S[j, i] = v
        return S