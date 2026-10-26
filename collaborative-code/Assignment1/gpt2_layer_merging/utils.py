"""General utilities."""

import gc
import random

import numpy as np
import torch

from .config import SEED


def set_seed(seed=SEED):
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_model(model):
    """Release a model and clear CUDA cache when available."""
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def shuffle_same_distribution(tensor, generator):
    """
    Randomly permute tensor entries.

    The resulting tensor preserves shape and empirical distribution while
    destroying the original coordinate structure.
    """
    shape = tensor.shape
    flat = tensor.detach().cpu().reshape(-1)

    perm = torch.randperm(flat.numel(), generator=generator)
    shuffled = flat[perm].reshape(shape)

    return shuffled.to(device=tensor.device, dtype=tensor.dtype)
