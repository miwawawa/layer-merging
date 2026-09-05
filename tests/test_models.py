"""Tests for the model families and the BaseNN contract."""

import torch

from models.mlp import SimpleMLP
from models.residual_mlp import ResidualMLP
from models.base import BaseNN

DIMS = (16, 16, 16)          # 3 hidden layers
IN, OUT = 8, 3
N = 5


def param_count(in_dim, hidden, out):
    p = in_dim * hidden[0] + hidden[0]                       # projection
    p += (len(hidden) - 1) * (hidden[0] ** 2 + hidden[0])    # inner hidden blocks
    p += hidden[-1] * out + out                              # head
    return p


def _check_shapes(model: BaseNN, all_square: bool):
    assert model.num_layers == len(DIMS)
    first_out, rest_out = DIMS[0], DIMS[0]
    for i, layer in enumerate(model.linears):
        if all_square:
            # residual: projection already mapped 8 -> 16, every hidden is 16x16
            assert layer.weight.shape == (DIMS[0], DIMS[0])
        else:
            # plain: first hidden is 8 -> 16, the rest are 16x16
            in_dim = IN if i == 0 else DIMS[0]
            assert layer.weight.shape == (DIMS[0], in_dim)
    assert model.output.weight.shape == (OUT, DIMS[-1])
    x = torch.randn(N, IN)
    logits = model(x)
    assert logits.shape == (N, OUT)
    logits2, acts = model.forward_with_activations(x)
    assert torch.equal(logits, logits2)
    assert len(acts) == len(DIMS)
    assert all(a.shape == (N, DIMS[0]) for a in acts)


def test_simple_mlp_shapes():
    _check_shapes(SimpleMLP(input_dim=IN, hidden_dims=DIMS, num_classes=OUT),
                  all_square=False)


def test_residual_mlp_shapes():
    _check_shapes(ResidualMLP(input_dim=IN, hidden_dims=DIMS, num_classes=OUT,
                              branch_scale=0.1), all_square=True)


def test_simple_mlp_param_count():
    m = SimpleMLP(input_dim=IN, hidden_dims=DIMS, num_classes=OUT)
    assert m.count_parameters() == param_count(IN, DIMS, OUT)


def test_toy_param_count_matches_hand_formula():
    # hidden (16,16,16): 8->16, 16->16, 16->16, 16->3
    m = SimpleMLP(input_dim=8, hidden_dims=(16, 16, 16), num_classes=3)
    assert m.count_parameters() == (8 * 16 + 16) + 2 * (16 * 16 + 16) + (16 * 3 + 3)


def test_insert_and_remove_linears(toy_model):
    n0 = toy_model.num_layers
    toy_model.remove_linears([1, 0])
    assert toy_model.num_layers == n0 - 2
    from torch import nn
    toy_model.insert_linear(0, nn.Linear(16, 16))
    assert toy_model.num_layers == n0 - 1


def test_reset_is_deterministic():
    torch.manual_seed(0)
    m1 = SimpleMLP(input_dim=IN, hidden_dims=DIMS, num_classes=OUT)
    torch.manual_seed(0)
    m2 = SimpleMLP(input_dim=IN, hidden_dims=DIMS, num_classes=OUT)
    w1 = [p.detach().clone() for p in m1.parameters()]
    w2 = [p.detach().clone() for p in m2.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(w1, w2))


def test_reset_with_unknown_init_raises(toy_model):
    import pytest
    with pytest.raises(ValueError):
        toy_model.reset(init_name="bogus")


def test_constructor_requires_dataset_params():
    # The "no hardcoding" contract: input_dim/hidden/classes must be explicit.
    import pytest
    with pytest.raises(TypeError):
        SimpleMLP()  # missing required arguments