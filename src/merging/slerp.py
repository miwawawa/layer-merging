"""SlerpMerge: Spherical Linear Interpolation (paper-2, CoM).

Instead of straight-line blending, interpolate along the geodesic on the unit
sphere so weight *magnitudes* stay natural:

  Omega  = arccos( w1 . w2 / (|w1||w2|) )          angle between vectors
  merged = sin((1-alpha) Omega)/sin Omega * w1
         + sin(alpha Omega)/sin Omega * w2

When weights are (near-)collinear (sin Omega ~ 0, both 0deg and 180deg) SLERP
    degenerates to the plain average — that fallback keeps everything stable.
    """

import numpy as np
import torch

from merging.weighted import WeightedMerge


class SlerpMerge(WeightedMerge):
    @staticmethod
    def fuse(w1: torch.Tensor, w2: torch.Tensor, alpha: float) -> torch.Tensor:
        a, b = w1.reshape(-1).double(), w2.reshape(-1).double()
        dot = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12)
        omega = float(torch.acos(torch.clamp(dot, -1.0, 1.0)))
        # Near-parallel AND near-antiparallel vectors make sinc(omega) blow up
        # (sin(pi-eps) ~ eps): fall back to the plain average whenever the two
        # directions are almost collinear, regardless of sign.
        if abs(np.sin(omega)) < 1e-5:
            return alpha * w1 + (1 - alpha) * w2            # parallel -> average
        merged = (np.sin((1 - alpha) * omega) * a
                  + np.sin(alpha * omega) * b) / np.sin(omega)
        return merged.reshape(w1.shape).float()