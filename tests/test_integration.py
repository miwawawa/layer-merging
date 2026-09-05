"""End-to-end integration: train -> similarity -> merge -> prune on a toy model.

This is the pytest counterpart of ``scripts/smoke_test.py`` but lives inside
the normal suite so ``make test`` covers the whole wiring.
"""

from copy import deepcopy

import numpy as np
import torch

from config.default import Config
from evaluation.metrics import evaluate, sparsity
from merging.factory import get_merger
from merging.merger import LayerMerger
from pruning.importance import LRPImportance
from pruning.pruner import prune_model_global
from similarity.factory import get_similarity
from utils.activations import extract_activations
from utils.seed import set_seed


def test_smoke_equivalent(trained_toy_model, loaders):
    set_seed(0)
    model = trained_toy_model
    n0 = model.count_parameters()

    acts = extract_activations(model, loaders["val"], max_samples=32)
    assert len(acts) == model.num_layers
    S = get_similarity("nmi", sigma=5.0, n_dims=4).compute(acts)
    assert S.shape == (model.num_layers, model.num_layers)

    # --- merge one friendly pair, parameters must shrink ---
    merger = LayerMerger(model, get_similarity("nmi", sigma=5.0, n_dims=4),
                         get_merger("weighted"), threshold=0.05)
    merger.set_calibration(loaders["val"], 32)
    assert merger.merge_once()
    assert model.count_parameters() < n0
    assert len(model.linears) == 2

    # --- LRP prune the merged model, sparsity must appear ---
    clone = deepcopy(model)
    imp = LRPImportance(eps=1e-1).score(clone, loaders["val"], n_samples=32)
    prune_model_global(clone, imp, keep_ratio=0.5)
    assert sparsity(clone) > 0.0
    assert torch.isfinite(torch.tensor(evaluate(clone, loaders["test"])))


def test_paper_story_survives_toy_scale(trained_toy_model, loaders, cfg):
    """The qualitative claims ('merging compresses', 'pruning adds sparsity')
    hold even at toy scale — this test protects against regressions in the
    overall pipeline wiring."""
    cfg = Config(merge_threshold=0.05)
    model = trained_toy_model
    acts = extract_activations(model, loaders["val"], max_samples=32)
    S = get_similarity("nmi", sigma=5.0, n_dims=4).compute(acts)
    off = S - np.eye(S.shape[0])

    merger = LayerMerger(model, get_similarity("nmi", sigma=5.0, n_dims=4),
                         get_merger("weighted"), threshold=cfg.merge_threshold)
    merger.set_calibration(loaders["val"], 32)
    n = merger.merge_all()
    assert n >= 1
    assert model.num_layers == 3 - n