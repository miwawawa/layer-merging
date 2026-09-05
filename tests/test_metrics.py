"""Tests for the extended metrics (FLOPs, size, latency, summarization)."""

import pytest
import torch

from evaluation.metrics import (accuracy_delta, compression_ratio,
                                count_flops, count_parameters,
                                inference_latency_ms, model_size_bytes,
                                sparsity, summarize, weight_sparsity)
from models.mlp import SimpleMLP
from models.residual_mlp import ResidualMLP


def test_flops_match_linear_formula():
    m = SimpleMLP(input_dim=8, hidden_dims=(16, 16), num_classes=3)
    flops = count_flops(m)
    expected = (2 * 8 * 16 + 1) + (2 * 16 * 16 + 1) + (2 * 16 * 3 + 1)
    assert flops == expected


def test_residual_flops_include_projection():
    m = ResidualMLP(input_dim=8, hidden_dims=(16, 16), num_classes=3,
                    branch_scale=0.1)
    flops = count_flops(m)
    expected = (2 * 8 * 16 + 1) + 2 * (2 * 16 * 16 + 1) + (2 * 16 * 3 + 1)
    assert flops == expected


def test_size_bytes_is_params_times_four():
    m = SimpleMLP(input_dim=8, hidden_dims=(16, 16), num_classes=3)
    assert model_size_bytes(m) == count_parameters(m) * 4


def test_latency_is_positive_and_finite():
    m = SimpleMLP(input_dim=8, hidden_dims=(16, 16), num_classes=3)
    from conftest import toy_loader
    ms = inference_latency_ms(m, toy_loader(), batches=5)
    assert ms > 0 and torch.isfinite(torch.tensor(ms))


def test_accuracy_delta():
    d = accuracy_delta(0.97, 0.9312)
    assert d["acc_drop"] == pytest.approx(0.0388, rel=1e-4)
    assert d["retention"] == pytest.approx(0.96, rel=1e-3)
    assert accuracy_delta(0.9, 0.0)["retention"] == 0.0


def test_weight_sparsity_excludes_biases(trained_toy_model, loaders):
    from pruning.importance import MagnitudeImportance
    from pruning.pruner import prune_model_global

    imp = MagnitudeImportance().score(trained_toy_model, loaders["val"], 8)
    prune_model_global(trained_toy_model, imp, keep_ratio=0.5)
    # Only weights can be zeroed, biases stay put: overall sparsity (with
    # biases) is below the ideal 0.5, weight sparsity is exactly 0.5.
    assert weight_sparsity(trained_toy_model) == pytest.approx(0.5, rel=0.02)
    assert sparsity(trained_toy_model) < weight_sparsity(trained_toy_model)


def test_summarize_snapshot_shape(trained_toy_model, loaders, cfg):
    snap = summarize(trained_toy_model, loaders["test"], stage="toy", arch="plain")
    for key in ("params", "flops", "size_bytes", "size_mb", "sparsity",
                "weight_sparsity", "nonzero", "layers", "acc", "latency_ms",
                "stage", "arch"):
        assert key in snap
    assert snap["params"] == trained_toy_model.count_parameters()
    assert snap["acc"] == pytest.approx(
        summarize(trained_toy_model, loaders["test"])["acc"])


def test_compression_ratio_guard():
    assert compression_ratio(100, 50) == pytest.approx(2.0)
    assert compression_ratio(100, 0) == 100.0  # never divide by zero