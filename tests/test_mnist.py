"""Tests for the from-scratch MNIST loader (parsing, caching, split)."""

import gzip
import struct

import numpy as np
import pytest

from data import mnist as M
from data.mnist import _download, _read_idx, get_mnist


def make_idx_gz(path: str, kind: str, n: int, payload) -> None:
    """Write a synthetic MNIST-style IDX ``.gz`` file (images or labels)."""
    if kind == "labels":
        header = struct.pack(">II", 2049, n)
    else:
        rows, cols = payload.shape[1], payload.shape[2]
        header = struct.pack(">IIII", 2051, n, rows, cols)
    with gzip.open(path, "wb") as f:
        f.write(header + payload.tobytes())


def test_read_idx_images(tmp_path):
    imgs = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    path = str(tmp_path / "imgs.gz")
    make_idx_gz(path, "images", 2, imgs)
    out = _read_idx(path)
    assert out.shape == (2, 3, 4)
    assert (out == imgs).all()


def test_read_idx_labels(tmp_path):
    labels = np.array([3, 1, 4, 1, 5], dtype=np.uint8)
    path = str(tmp_path / "labels.gz")
    make_idx_gz(path, "labels", len(labels), labels)
    out = _read_idx(path)
    assert out.shape == (5,)
    assert (out == labels).all()


def test_download_skips_complete_files(tmp_path, monkeypatch):
    # A valid, complete file present on disk -> urlretrieve must NOT be called.
    rng = np.random.default_rng(0)  # incompressible data so the .gz stays > 1000 B
    images = rng.integers(0, 256, size=(28, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, 256, size=(4096,), dtype=np.uint8)
    for name in M._FILES:                       # all four files must exist
        make_idx_gz(str(tmp_path / name), "images" if "images" in name else "labels",
                    28 if "images" in name else 4096,
                    images if "images" in name else labels)

    def boom(*a, **k):  # pragma: no cover - should never be reached
        raise AssertionError("urlretrieve called for a complete file set")

    monkeypatch.setattr(M, "urlretrieve", boom)
    _download(str(tmp_path), "https://example.invalid/mnist/")  # mirror unused


def test_download_fetches_missing_files(tmp_path, monkeypatch):
    calls = []

    def fake_retrieve(url, path):
        calls.append(url)
        # write something big enough to count as "complete" for the others
        make_idx_gz(path, "images", 2, np.zeros((2, 28, 28), dtype=np.uint8))

    monkeypatch.setattr(M, "urlretrieve", fake_retrieve)
    _download(str(tmp_path), "https://example.invalid/mnist/")
    # each of the four MNIST files should have been requested exactly once
    assert len(calls) == 4
    assert all(c.startswith("https://example.invalid/mnist/") for c in calls)


def test_get_mnist_val_split(tmp_path, monkeypatch):
    # Patch out the actual file I/O: give a fake 40-sample training dataset.
    def fake_dataset(root, kind, mirror):
        x = np.zeros((40, 28, 28), dtype=np.uint8)
        y = np.zeros(40, dtype=np.uint8)
        from torch.utils.data import TensorDataset
        import torch
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) / 255.0
        y = torch.tensor(y, dtype=torch.long)
        return TensorDataset(x, y)

    monkeypatch.setattr(M, "_dataset", fake_dataset)
    loaders = get_mnist(root=str(tmp_path), batch_size=32, val_size=10,
                        mirror="https://example.invalid/mnist/")
    assert len(loaders["train"].dataset) == 30        # 40 - val_size
    assert len(loaders["val"].dataset) == 10
    assert loaders["train"].batch_size == 32