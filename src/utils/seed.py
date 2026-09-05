"""Reproducibility: make every random source deterministic."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed torch, numpy, random and the CUDA backend (if any)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)