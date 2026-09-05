"""DiffusionMap: the manifold-learning step of the MKA paper (Algorithm 1).

Projects each layer's activation point-cloud into a low-dimensional manifold
space via the diffusion-operator eigen-decomposition, so that layer similarity
is measured on learned geometry rather than raw coordinates.

Equations (paper, Sec. 3.2):
  W_ij  = exp(-||H_i - H_j||^2 / sigma^2)     Gaussian affinity over samples
  D_ii  = sum_j W_ij                          degree matrix
  P~    = D^(-1/2) W D^(-1/2)                 symmetric diffusion operator
  Phi_t = (lambda_2^t phi_2, ..., lambda_{k+1}^t phi_{k+1})   diffusion map
"""

from typing import List

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import squareform, pdist


class DiffusionMap:
    def __init__(self, sigma: float = 1.0, n_dims: int = 8, time: float = 1.0):
        self.sigma = sigma
        self.n_dims = n_dims
        self.time = time

    def embed(self, H: np.ndarray) -> np.ndarray:
        """Embed one layer's activations H (n_samples, d) -> (n_samples, n_dims)."""
        n = H.shape[0]
        if n_dims := self.n_dims:
            pass
        dists = squareform(pdist(H, metric="euclidean"))
        W = np.exp(-(dists ** 2) / (self.sigma ** 2))            # affinity
        np.fill_diagonal(W, 0.0)                                  # no self-loops
        D = W.sum(axis=1)
        Dinv = np.diag(1.0 / np.sqrt(D + 1e-12))
        P = Dinv @ W @ Dinv                                        # symmetric operator
        eigvals, eigvecs = eigh(P)                                 # ascending
        idx = np.argsort(eigvals)[::-1]                            # descending
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        lam = eigvals[1 : 1 + self.n_dims]                         # skip trivial λ=1
        phi = eigvecs[:, 1 : 1 + self.n_dims]
        return (lam ** self.time) * phi                            # diffusion map

    def embed_all(self, activations: List[np.ndarray]) -> List[np.ndarray]:
        """Embed every layer; each returns (n_samples, n_dims) in its own manifold."""
        return [self.embed(a) for a in activations]