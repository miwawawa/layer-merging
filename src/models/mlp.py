"""SimpleMLP: the 'dummy' neural network used as the non-residual baseline.

Symmetric hidden dims so that all hidden layers are shape-identical and
therefore mergeable (exactly the condition that makes MKA work on transformers,
whose stacked layers share ``d_model``).

Dataset-specific values (``input_dim``, ``hidden_dims``, ``num_classes``) are
REQUIRED constructor arguments — the experiments always pass them from
``Config`` (see ``experiments/common.py::build_model``). No magic defaults here.
"""

import torch
import torch.nn as nn

from models.base import BaseNN


class SimpleMLP(BaseNN, nn.Module):
    def __init__(self, input_dim: int, hidden_dims, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        dims = [input_dim, *hidden_dims]
        self._linears = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self._linears.append(nn.Linear(dims[i], dims[i + 1]))
        self._output = nn.Linear(hidden_dims[-1], num_classes)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.reset()

    @property
    def linears(self):
        return self._linears

    @property
    def output(self):
        return self._output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_activations(x)[0]

    def forward_with_activations(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        h = x
        acts = []
        for layer in self._linears:
            h = self.drop(self.act(layer(h)))
            acts.append(h)
        return self._output(h), acts