"""Affine fitting, affine merge conditions, and affine replacement blocks."""

import torch
import torch.nn as nn

from .config import RIDGE
from .model import load_fresh_model
from .utils import shuffle_same_distribution


def fit_affine(X, Y, ridge=RIDGE):
    """
    Fit a ridge-regularized affine map under the row-vector convention:

        Y ~= X @ A + b
    """
    X = X.double()
    Y = Y.double()

    x_mean = X.mean(dim=0, keepdim=True)
    y_mean = Y.mean(dim=0, keepdim=True)

    Xc = X - x_mean
    Yc = Y - y_mean

    gram = Xc.T @ Xc

    diag_scale = gram.diag().mean().clamp_min(1e-12)

    reg = (
        ridge
        * diag_scale
        * torch.eye(gram.shape[0], dtype=gram.dtype)
    )

    A = torch.linalg.solve(
        gram + reg,
        Xc.T @ Yc,
    )

    b = (y_mean - x_mean @ A).squeeze(0)

    return A.float(), b.float()


def make_affine_conditions(A1, b1, A2, b2):
    """
    Construct the fixed affine versions of Remove, Average, and Multiply.

    Remove:
        A = A2, b = b2

    Average:
        A = 0.5*A1 + 0.5*A2
        b = 0.5*b1 + 0.5*b2

    Multiply:
        A = A1 @ A2
        b = b1 @ A2 + b2
    """
    return {
        "Remove": (A2, b2),
        "Average": (
            0.5 * A1 + 0.5 * A2,
            0.5 * b1 + 0.5 * b2,
        ),
        "Multiply": (
            A1 @ A2,
            b1 @ A2 + b2,
        ),
    }


def make_random_weight_affine_condition(A2, b2, seed):
    """Create the randomized-weight affine condition."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    A_random = shuffle_same_distribution(A2, generator)
    b_random = shuffle_same_distribution(b2, generator)

    A_merge = 0.5 * A2 + 0.5 * A_random
    b_merge = 0.5 * b2 + 0.5 * b_random

    return A_merge, b_merge


@torch.no_grad()
def hidden_mse(X, Y_target, A, b):
    """Compute affine hidden-state reconstruction MSE."""
    pred = X @ A + b
    return torch.mean((pred - Y_target) ** 2).item()


class AffineMergedBlock(nn.Module):
    """A single affine layer used as a replacement Transformer block."""

    def __init__(self, A, b):
        super().__init__()

        d = A.shape[0]
        self.linear = nn.Linear(d, d)

        with torch.no_grad():
            # nn.Linear computes x @ weight.T + bias.
            self.linear.weight.copy_(A.T)
            self.linear.bias.copy_(b)

    def forward(self, hidden_states, *args, **kwargs):
        return self.linear(hidden_states)


def build_affine_merged_model(L, A, b):
    """Replace GPT-2 blocks L and L+1 with one affine block."""
    model = load_fresh_model()
    merged = AffineMergedBlock(A, b)

    blocks = list(model.transformer.h)
    new_blocks = blocks[:L] + [merged] + blocks[L + 2:]

    model.transformer.h = nn.ModuleList(new_blocks)
    model.config.n_layer = len(new_blocks)
    model.config.use_cache = False

    return model
