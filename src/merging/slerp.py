"""SlerpMerge: Spherical Linear Interpolation (paper-2, CoM).

Instead of straight-line blending, interpolate along the geodesic on the unit
sphere so weight *magnitudes* stay natural:

  Omega  = arccos( w1 . w2 / (|w1||w2|) )          angle between vectors
  merged = sin((1-alpha) Omega)/sin Omega * w1
         + sin(alpha Omega)/sin Omega * w2

When weights are (near-)parallel (sin Omega ~ 0) SLERP degenerates to the
plain average — that fallback keeps everything stable.
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
        if abs(np.sin(omega)) < 1e-6:
            return alpha * w1 + (1 - alpha) * w2            # parallel -> average
        merged = (np.sin((1 - alpha) * omega) * a
                  + np.sin(alpha * omega) * b) / np.sin(omega)
        return merged.reshape(w1.shape).float()