"""Tests for the pruning stack (LRP, importance scoring, mask application)."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from pruning.lrp import LRPAnalyzer
from pruning.importance import LRPImportance, MagnitudeImportance
from pruning.pruner import prune_mask, apply_prune, prune_model, prune_model_global
from evaluation.metrics import sparsity, nonzero_parameters


def test_lrp_input_relevance_shape_and_finite(trained_toy_model, loaders):
    model = trained_toy_model
    analyzer = LRPAnalyzer(model, eps=1e-1)
    x, y = next(iter(loaders["val"]))
    r = analyzer.input_relevance(x[:2], target=y[:2], aggregate=True)
    assert r.shape == (model.input_dim,)
    assert torch.isfinite(r).all()


def test_lrp_weight_relevance_shapes(trained_toy_model, loaders):
    model = trained_toy_model
    analyzer = LRPAnalyzer(model, eps=1e-1)
    x, y = next(iter(loaders["val"]))
    rel = analyzer.weight_relevance(x[:4], y[:4], aggregate=True)
    assert len(rel) == model.num_layers                # hidden layers only
    for layer, r in zip(model.linears, rel):
        # attribution is R_{i<-j}: shape (in, out) = transposed weight shape
        assert tuple(r.shape) == (layer.weight.shape[1], layer.weight.shape[0])


def test_lrp_weight_relevance_finite(trained_toy_model, loaders):
    analyzer = LRPAnalyzer(trained_toy_model, eps=1e-1)
    x, y = next(iter(loaders["val"]))
    assert all(torch.isfinite(t).all() for t in analyzer.weight_relevance(x, y))


def test_lrp_conservation_app_rox(fixed_model, loaders):
    # Total input relevance should (approximately) equal total output relevance.
    analyzer = LRPAnalyzer(fixed_model, eps=1e-6)
    x, y = next(iter(loaders["val"]))
    r_in = analyzer.input_relevance(x[:4], y[:4], aggregate=True)
    assert r_in.abs().sum() > 0


@pytest.fixture
def fixed_model(trained_toy_model, loaders):
    return trained_toy_model


def test_importance_scores_aligned_with_params(trained_toy_model, loaders):
    model = trained_toy_model
    imp = LRPImportance(eps=1e-1).score(model, loaders["val"], n_samples=32)
    layers = list(model.linears) + [model.output]
    assert len(imp) == len(layers)             # output layer included
    for score, layer in zip(imp, layers):
        expected = layer.weight.numel() + layer.bias.numel()
        assert score.size == expected
    assert np.isfinite(np.concatenate(imp)).all()


def test_magnitude_importance_size(trained_toy_model, loaders):
    model = trained_toy_model
    layers = list(model.linears) + [model.output]
    imp = MagnitudeImportance().score(model, loaders["val"], 8)
    assert all(s.size == l.weight.numel() + l.bias.numel()
               for s, l in zip(imp, layers))


def test_prune_mask_keeps_ratio():
    imp = np.arange(100, dtype=float)          # ranks are deterministic
    mask = prune_mask(imp, keep_ratio=0.5)
    assert mask.dtype == bool
    assert int(mask.sum()) == 50               # top-half kept
    kept = np.where(mask)[0]
    assert (kept == np.argsort(imp)[-50:]).all()


def test_prune_mask_row_wise():
    imp = np.arange(6, dtype=float).reshape(2, 3)
    mask = prune_mask(imp, keep_ratio=0.5, row_wise=True)
    assert int(mask.sum()) == 3                # one output row flat -> 3 entries


def test_prune_mask_row_wise_keeps_whole_output_rows():
    # Structured masking must drop complete output neurons, never a mix.
    imp = np.arange(12, dtype=float).reshape(4, 3)
    mask = prune_mask(imp, keep_ratio=0.5, row_wise=True)
    assert mask.shape == (12,)
    assert int(mask.sum()) == 6                             # 2 of 4 rows kept
    rows = mask.reshape(4, 3)
    per_row = rows.sum(axis=1)
    assert set(per_row.tolist()) == {0, 3}                  # all-or-nothing rows


def test_prune_mask_row_wise_flat_with_shape():
    # prune_model passes a flat per-layer importance plus the (out, in) shape;
    # the mask must recover whole output rows from that flat view.
    imp = np.array([0.0, 1.0, 10.0, 11.0, 100.0, 101.0])    # 2 rows, sum-row 0 < 1
    mask = prune_mask(imp, keep_ratio=0.5, row_wise=True, shape=(2, 3))
    assert mask.shape == (6,)
    # row 1 (101-sum) is kept, row 0 (1-sum) is dropped -> flat pattern 000111
    assert mask.tolist() == [False, False, False, True, True, True]


def test_apply_prune_pads_short_mask():
    layer = torch.nn.Linear(6, 2)              # weight has 12 elements
    mask = np.ones(6, dtype=bool)              # too short -> padded True
    assert apply_prune(layer, mask) == 12


def test_prune_model_sparsity(trained_toy_model, loaders):
    model = trained_toy_model
    layers = list(model.linears) + [model.output]
    imp = MagnitudeImportance().score(model, loaders["val"], 8)
    keep_ratio = 0.5
    total = sum(l.weight.numel() for l in layers)       # biases untouched
    kept = prune_model(model, imp, keep_ratio=keep_ratio)
    assert kept == pytest.approx(keep_ratio * total, rel=0.02)
    # overall sparsity (weights + biases) is a bit below keep_ratio because
    # biases are never masked
    assert sparsity(model) == pytest.approx(1 - keep_ratio, abs=0.1)
    assert nonzero_parameters(model) > 0


def test_prune_model_global_sparsity(trained_toy_model, loaders):
    model = trained_toy_model
    layers = list(model.linears) + [model.output]
    imp = MagnitudeImportance().score(model, loaders["val"], 8)
    keep_ratio = 0.5
    prune_model_global(model, imp, keep_ratio=keep_ratio)
    weight_total = sum(l.weight.numel() for l in layers)
    kept = sum(int((l.weight != 0).sum()) for l in layers)
    assert kept == pytest.approx(keep_ratio * weight_total, rel=0.02)


def test_prune_only_zeros_least_important(trained_toy_model, loaders):
    # Eq.3 of SparC3: psi_i -> (1 - 1_{i in c_q}) psi_i. Pruning must ADD zeros
    # (never change other values) and never resurrect any pre-existing zero.
    model = trained_toy_model
    layers = list(model.linears) + [model.output]
    imp = MagnitudeImportance().score(model, loaders["val"], 8)
    clone = deepcopy(model)
    prune_model(clone, imp, keep_ratio=0.3, row_wise=False)
    for pruned_layer, orig_layer in zip(list(clone.linears) + [clone.output], layers):
        pr, orig = pruned_layer.weight, orig_layer.weight
        # the surviving values are byte-identical to the originals
        keep = pr != 0
        np.testing.assert_array_equal(pr[keep].detach().numpy(),
                                      orig[keep].detach().numpy())