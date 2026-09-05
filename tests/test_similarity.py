"""Tests for the layer-similarity stack (diffusion map, Gaussian NMI, cosine)."""

import numpy as np
import pytest
import torch

from similarity.diffusion import DiffusionMap
from similarity.gaussian_mi import gaussian_mi, gaussian_nmi
from similarity.cosine import CosineSimilarity
from similarity.nmi import NmiSimilarity
from similarity.factory import get_similarity

N = 64
K = 4


def _gaussian(n=N, k=K, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, k)).astype(np.float64)


def test_diffusion_map_embed_shape():
    dm = DiffusionMap(sigma=2.0, n_dims=K, time=1.0)
    emb = dm.embed(_gaussian())
    assert emb.shape == (N, K)


def test_diffusion_map_more_dims_ok():
    dm = DiffusionMap(sigma=2.0, n_dims=8, time=2)
    emb = dm.embed(_gaussian(n=80))
    assert emb.shape == (80, 8)                # dim is eigen-based, not feature-bound


def test_gaussian_nmi_identical_is_one():
    psi = _gaussian()
    value = gaussian_nmi(psi, psi)
    assert value == pytest.approx(1.0, abs=1e-3)          # I = H -> NMI = 1


def test_gaussian_nmi_independent_is_near_zero():
    a, b = _gaussian(seed=1), _gaussian(seed=2)
    value = gaussian_nmi(a, b)
    assert 0.0 <= value < 0.15


def test_gaussian_nmi_bounds_and_symmetry():
    a, b = _gaussian(seed=3), _gaussian(seed=4)
    # correlated pair (b = a + noise) -> mid-range NMI, symmetric.
    b = a + _gaussian(seed=4) * 0.5
    v_ab = gaussian_nmi(a, b)
    v_ba = gaussian_nmi(b, a)
    assert v_ab == pytest.approx(v_ba, rel=1e-3)
    assert 0.0 <= v_ab <= 1.0
    assert 0.3 < v_ab < 1.0


def test_gaussian_mi_non_negative():
    a, b = _gaussian(seed=5), _gaussian(seed=6)
    assert gaussian_mi(a, b) >= -1e-6


def test_gaussian_mi_matches_known_1d_formula():
    # For two standardized 1-D Gaussians with correlation rho:
    #   I = -0.5 * ln(1 - rho^2)   (closed form, S = [[1, rho],[rho, 1]])
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.normal(size=(n, 1))
    rho = 0.8
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=(n, 1))
    value = gaussian_mi(x, y)
    expected = -0.5 * np.log(1 - rho**2)            # ~0.5108
    assert value == pytest.approx(expected, abs=0.06)


def test_gaussian_mi_zero_when_independent():
    # I = 0 for uncorrelated Gaussians (sample noise keeps it tiny).
    rng = np.random.default_rng(9)
    a = rng.normal(size=(3000, 1))
    b = rng.normal(size=(3000, 1))
    assert gaussian_mi(a, b) < 0.05


def test_cosine_similarity_matrix():
    acts = [torch.tensor(ar) for ar in (_gaussian(), _gaussian(seed=1))]
    S = CosineSimilarity().compute(acts)
    assert S.shape == (2, 2)
    assert S[0, 0] == pytest.approx(1.0)                  # self-similar
    assert S[1, 0] == pytest.approx(S[0, 1])             # symmetric
    assert all(-1.0 <= v <= 1.0 for row in S for v in row)


def test_nmi_similarity_matrix_symmetric_diag():
    acts = [torch.tensor(a) for a in (_gaussian(), _gaussian(seed=1), _gaussian(seed=2))]
    S = NmiSimilarity(sigma=2.0, n_dims=K, time=1.0).compute(acts)
    assert S.shape == (3, 3)
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0)                   # diagonal reset to 1


def test_factory_builds_both_similarities():
    assert get_similarity("cosine") is not None
    assert get_similarity("nmi", sigma=1.0, n_dims=2) is not None


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        get_similarity("bogus")