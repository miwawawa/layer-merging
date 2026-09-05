"""ResidualMLP: the architecturally-faithful reproduction of the MKA paper.

The paper's success on transformers relies on residual connections + layer
normalization — information flows around each layer through the residual
stream, so fusing layers is much less destructive. This model mirrors that:
a projection layer maps input into a symmetric hidden space, then each hidden
layer is a residual block ``h = h + act(W h)`` (all dims 256 = "d_model").

The branch-init gain ``branch_scale`` (required, supplied by the experiments
as ``Config.branch_scale``) is deliberately *small*: scaling each residual
branch down keeps the hidden states of distinct layers from collapsing into
identical copies, which would make merging trivial and meaningless (a pitfall
we hit and fixed during development — see docs/architecture.md).

Contrast with SimpleMLP (no residuals): merging destroys accuracy there —
an honest finding we report, not hide.
"""

import torch
import torch.nn as nn

from models.base import BaseNN


class ResidualMLP(BaseNN, nn.Module):
    def __init__(self, input_dim: int, hidden_dims, num_classes: int,
                 dropout: float = 0.0, branch_scale: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dims[0])
        self._linears = nn.ModuleList(
            [nn.Linear(hidden_dims[0], hidden_dims[0]) for _ in hidden_dims])
        self._output = nn.Linear(hidden_dims[-1], num_classes)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            for layer in self._linears:      # distinct, small branches break the
                nn.init.xavier_normal_(layer.weight, gain=branch_scale)
                nn.init.zeros_(layer.bias)

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
        h = self.act(self.proj(x))
        acts = []
        for layer in self._linears:
            h = h + self.drop(self.act(layer(h)))     # residual block
            acts.append(h)
        return self._output(h), acts