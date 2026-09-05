"""Shared pytest fixtures and environment for the whole suite.

* Forces matplotlib to the non-interactive Agg backend (no $DISPLAY needed).
* Seeds everything so tests are deterministic.
* Provides a small toy model + toy loader used by many tests.
"""
import os

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from config.default import Config


@pytest.fixture(autouse=True)
def _seed_and_backend():
    """Deterministic RNG for every test."""
    torch.manual_seed(0)
    np.random.seed(0)
    yield


def toy_loader(n: int = 64, in_dim: int = 8, out_dim: int = 3,
               batch: int = 16) -> DataLoader:
    """Small synthetic (x, class-label) loader for toy-scale tests."""
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(n, in_dim)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, out_dim, size=n))
    return DataLoader(TensorDataset(x, y), batch_size=batch)


@pytest.fixture
def loaders():
    """{train, val, test} dict of the same toy loader (separate instances)."""
    return {"train": toy_loader(), "val": toy_loader(), "test": toy_loader()}


@pytest.fixture
def cfg() -> Config:
    """Config with defaults (reads .env if present — harmless)."""
    return Config()


@pytest.fixture
def toy_model():
    """A freshly-initialized 3-hidden-layer plain MLP (8 -> 16 -> 16 -> 16 -> 3)."""
    from models.mlp import SimpleMLP
    return SimpleMLP(input_dim=8, hidden_dims=(16, 16, 16), num_classes=3)


@pytest.fixture
def trained_toy_model(toy_model, loaders):
    """A trained toy model (2 epochs, unknowable but deterministic)."""
    from training import train_model
    train_model(toy_model, loaders, Config(), epochs=2, lr=1e-2)
    return toy_model


def make_idx_gz(path: str, kind: str, n: int, payload) -> None:
    """Write a synthetic MNIST-style IDX ``.gz`` file (images or labels)."""
    import gzip
    import struct

    if kind == "labels":
        header = struct.pack(">II", 2049, n)
    else:
        rows, cols = payload.shape[1], payload.shape[2]
        header = struct.pack(">IIII", 2051, n, rows, cols)
    with gzip.open(path, "wb") as f:
        f.write(header + payload.tobytes())