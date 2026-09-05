"""Tests for merge strategies (uniform / weighted / SLERP) and the merger facade."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from merging.merger import LayerMerger
from merging.protocol import MergeStrategy
from merging.uniform import UniformMerge
from merging.weighted import WeightedMerge
from merging.slerp import SlerpMerge
from merging.factory import get_merger
from similarity.factory import get_similarity
from utils.activations import extract_activations


def test_all_strategies_implement_protocol():
    for name in ("uniform", "weighted", "slerp"):
        s = get_merger(name)
        assert isinstance(s, MergeStrategy)
        assert isinstance(s.weight(0.7), float)
        w1 = torch.randn(4, 3)
        w2 = torch.randn(4, 3)
        out = s.fuse(w1, w2, alpha=0.3)
        assert out.shape == w1.shape


def test_uniform_weight_and_fuse():
    u = UniformMerge()
    assert u.weight(0.123) == 0.5
    w1 = torch.ones(2, 2)
    w2 = torch.zeros(2, 2)
    assert torch.equal(u.fuse(w1, w2), 0.5 * torch.ones(2, 2))


def test_weighted_clamps_alpha_to_bounds():
    w = WeightedMerge(min_alpha=0.2, max_alpha=0.8)
    assert w.weight(0.99) == pytest.approx(0.8)    # above max -> clamp
    assert w.weight(0.01) == pytest.approx(0.2)    # below min -> clamp
    assert w.weight(0.5) == pytest.approx(0.5)     # inside -> identity


def test_weighted_fuse_convex_combination():
    w = WeightedMerge()
    w1 = torch.ones(3)
    w2 = torch.zeros(3)
    assert torch.equal(w.fuse(w1, w2, alpha=0.7), 0.7 * torch.ones(3))


def test_slerp_fallback_when_parallel():
    # Parallel vectors -> sin(Omega)=0 -> linear interpolation fallback.
    s = SlerpMerge()
    w1 = torch.ones(5)
    w2 = 3 * torch.ones(5)
    assert torch.equal(s.fuse(w1, w2, alpha=0.25), 0.25 * w1 + 0.75 * w2)


def test_slerp_normal_case_differs_from_linear():
    s = SlerpMerge()
    w1 = torch.tensor([1.0, 0.0, 0.0])
    w2 = torch.tensor([0.0, 1.0, 0.0])             # 90 deg apart
    out = s.fuse(w1, w2, alpha=0.5)
    linear = 0.5 * w1 + 0.5 * w2
    assert not torch.allclose(out, linear, atol=1e-3)
    # SLERP stays on the sphere: midpoint norm ~ 1 (unit vectors, 90 deg).
    assert float(torch.norm(out)) == pytest.approx(1.0, abs=1e-4)


def test_merger_reduces_parameters(trained_toy_model, loaders):
    before = trained_toy_model.count_parameters()
    merger = LayerMerger(trained_toy_model, get_similarity("nmi", sigma=5.0, n_dims=4),
                         get_merger("weighted"), threshold=0.05)
    merger.set_calibration(loaders["val"], 32)
    n_merges = merger.merge_all()
    assert n_merges >= 1
    assert trained_toy_model.count_parameters() < before
    assert trained_toy_model.num_layers < 3


def test_merger_threshold_stops_everything(trained_toy_model, loaders):
    merger = LayerMerger(trained_toy_model, get_similarity("nmi", sigma=5.0, n_dims=4),
                         get_merger("weighted"), threshold=0.9999)
    merger.set_calibration(loaders["val"], 32)
    assert merger.merge_once() is False           # nothing is similar enough


def test_merger_strategy_matrix(trained_toy_model, loaders):
    sims = get_similarity("nmi", sigma=5.0, n_dims=4)
    acts = extract_activations(trained_toy_model, loaders["val"], max_samples=32)
    S = sims.compute(acts)
    assert S.shape == (3, 3)
    assert np.all(S >= 0.0)


def test_merger_only_fuses_compatible_pairs(trained_toy_model, loaders):
    # All hidden layers share shape here; ensure no incompatible fuse crashes.
    n0 = trained_toy_model.num_layers
    merger = LayerMerger(trained_toy_model, get_similarity("nmi", sigma=5.0, n_dims=4),
                         get_merger("weighted"), threshold=0.05)
    merger.set_calibration(loaders["val"], 32)
    while merger.merge_once():
        pass
    assert trained_toy_model.num_layers < n0


def test_merger_merge_log_records_history(trained_toy_model, loaders):
    merger = LayerMerger(trained_toy_model, get_similarity("nmi", sigma=5.0, n_dims=4),
                         get_merger("weighted"), threshold=0.05)
    merger.set_calibration(loaders["val"], 32)
    n_merges = merger.merge_all()
    assert n_merges >= 1
    assert len(merger.merge_log) == n_merges                # one record per merge
    for record in merger.merge_log:
        for key in ("pair", "similarity", "alpha", "params_before", "params_out"):
            assert key in record
    # history is strictly shrinking until the graph has nothing left to fuse
    params = [r["params_out"] for r in merger.merge_log]
    assert params == sorted(params, reverse=True)


def test_slerp_antiparallel_falls_back_to_average():
    # Opposite directions -> Omega = pi -> sin(Omega)=0 -> linear average
    # alpha*w1 + (1-alpha)*w2 (= -0.5 for alpha=0.25 here).
    s = SlerpMerge()
    w1 = torch.tensor([1.0, 0.0, 0.0])
    w2 = torch.tensor([-1.0, 0.0, 0.0])
    out = s.fuse(w1, w2, alpha=0.25)
    assert torch.allclose(out, 0.25 * w1 + 0.75 * w2, atol=1e-6)
    assert torch.allclose(out, torch.tensor([-0.5, 0.0, 0.0]), atol=1e-6)