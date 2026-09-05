"""BaseNN: the single contract every trainable model in this project follows."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn


class BaseNN(ABC):
    """A feed-forward network whose hidden layers are exposed for merging/pruning.

    Hides nothing: subclasses must define ``linears`` (the list of hidden layers),
    ``output`` (final classifier head), and ``forward``.
    """

    @property
    @abstractmethod
    def linears(self) -> List[nn.Module]:
        """Hidden layers, in order. All of these are merge/prune candidates."""

    @property
    @abstractmethod
    def output(self) -> nn.Module:
        """Final layer mapping last hidden dim -> logits."""

    @property
    def num_layers(self) -> int:
        return len(self.linears)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full pass: flatten input -> hidden layers -> logits."""

    def remove_linears(self, indices: List[int]) -> None:
        """Delete hidden layers by index (deletes highest first)."""
        for idx in sorted(indices, reverse=True):
            del self.linears[idx]

    def insert_linear(self, index: int, module: nn.Module) -> None:
        self.linears.insert(index, module)

    def forward_with_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run forward and also return every hidden activation (for similarity)."""
        acts: List[torch.Tensor] = []
        h = x
        for layer in self.linears:
            h = layer(h)
            acts.append(h)
        return self.output(h), acts

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset(self, init_name: str = "xavier_uniform") -> None:
        """Re-seed parameter init so every (re)train starts identically.

        ``init_name`` selects the initialization used for all Linear layers
        (a named registry kept here so the choice lives in exactly one place).
        """
        inits = {
            "xavier_uniform": nn.init.xavier_uniform_,
            "xavier_normal": nn.init.xavier_normal_,
        }
        if init_name not in inits:
            raise ValueError(f"Unknown init {init_name!r}; choose {list(inits)}")
        init_fn = inits[init_name]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_fn(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()