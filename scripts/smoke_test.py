"""Fast smoke test: exercises every module on a tiny toy model (no MNIST needed)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config.default import Config
from evaluation.metrics import evaluate, sparsity
from merging.factory import get_merger
from merging.merger import LayerMerger
from models.mlp import SimpleMLP
from pruning.importance import LRPImportance, MagnitudeImportance
from pruning.lrp import LRPAnalyzer
from pruning.pruner import prune_model
from similarity.factory import get_similarity
from utils.activations import extract_activations
from utils.seed import set_seed


def toy_loader(n=64, in_dim=8, out_dim=3, batch=16):
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(n, in_dim)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, out_dim, size=n))
    d = TensorDataset(x, y)
    return DataLoader(d, batch_size=batch)


def main():
    set_seed(0)
    cfg = Config()  # heavy fields unused here
    model = SimpleMLP(input_dim=8, hidden_dims=(16, 16, 16), num_classes=3)
    loaders = {"train": toy_loader(), "val": toy_loader(), "test": toy_loader()}

    from training import train_model
    train_model(model, loaders, cfg, epochs=2)

    n0 = model.count_parameters()
    expected = (8 * 16 + 16) + 2 * (16 * 16 + 16) + (16 * 3 + 3)
    assert n0 == expected, f"param count wrong: {n0}"

    acts = extract_activations(model, loaders["val"], max_samples=32)
    assert len(acts) == 3
    S_nmi = get_similarity("nmi", sigma=1.0, n_dims=4).compute(acts)
    S_cos = get_similarity("cosine").compute(acts)
    assert S_nmi.shape == (3, 3) and S_cos.shape == (3, 3)

    merger = LayerMerger(model, get_similarity("nmi"), get_merger("weighted"))
    merger.set_calibration(loaders["val"], 32)
    assert merger.merge_once()
    assert model.count_parameters() < n0
    print(f"[smoke] merge OK: {n0} -> {model.count_parameters()} params")

    analyzer = LRPAnalyzer(model, eps=1e-1)
    x, y = next(iter(loaders["val"]))
    r = analyzer.weight_relevance(x, y)
    assert all(torch.isfinite(t).all() for t in r), "LRP produced non-finite relevance"
    ri = analyzer.input_relevance(x[:1], y[:1])
    assert ri.shape == (8,) and torch.isfinite(ri).all()

    imp = LRPImportance(eps=1e-1).score(model, loaders["val"], n_samples=32)
    from copy import deepcopy
    clone = deepcopy(model)
    prune_model(clone, imp, keep_ratio=0.5)
    assert sparsity(clone) > 0.3
    print(f"[smoke] prune OK: sparsity={sparsity(clone):.2f}")

    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()