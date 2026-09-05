"""LRP: Layer-wise Relevance Propagation (paper-3, SparC3).

Backward rule for a linear layer with inputs a, weights W (out,in), bias b:
    z_ij = a_i * W_ij ,   z_j = sum_i z_ij + b_j
    R_{i<-j} = (z_ij / z_j) * R_j ,   R_i = sum_j R_{i<-j}

We use the epsilon-stabilized denominator (classic rule):
    z_j -> z_j + sign(z_j) * eps * mean(|z_j|)

Relevance starts at the output neurons (one-hot on the predicted class weighted
by its softmax probability) and flows back through every linear, giving
per-input and per-weight attributions.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.base import BaseNN


def _stabilized(z: torch.Tensor, eps: float) -> torch.Tensor:
    scale = eps * z.abs().mean(dim=-1, keepdim=True)
    return z + torch.sign(z) * scale


class LRPAnalyzer:
    def __init__(self, model: BaseNN, eps: float = 1e-2):
        self.model = model
        self.eps = eps

    def _forward_trace(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, nn.Linear]]:
        """Record (input-to-layer, layer) for every linear incl. the output."""
        a = x.view(x.size(0), -1) if x.dim() == 4 else x
        h = a
        trace: List[Tuple[torch.Tensor, nn.Linear]] = []
        for layer in self.model.linears:
            trace.append((h, layer))
            h = torch.relu(layer(h))
        trace.append((h, self.model.output))
        return trace

    def _backward(self, trace, R_out) -> torch.Tensor:
        for a, layer in reversed(trace):
            z = layer(a)
            C = a.unsqueeze(2) * layer.weight.t().unsqueeze(0)      # (b, in, out)
            zs = _stabilized(z, self.eps)
            R_out = (C / zs.unsqueeze(1) * R_out.unsqueeze(1)).sum(dim=2)
        return R_out

    def input_relevance(self, x: torch.Tensor, target: Optional[torch.Tensor] = None,
                        aggregate: bool = True) -> torch.Tensor:
        """Relevance of every input pixel for the (predicted) class at `target`."""
        trace = self._forward_trace(x)
        logits = self.model.output(trace[-1][0])
        R = self._backward(trace, self._output_relevance(logits, target))
        return R.mean(dim=0).detach() if aggregate else R.detach()

    def weight_relevance(self, x: torch.Tensor, target: Optional[torch.Tensor] = None,
                         aggregate: bool = True) -> List[torch.Tensor]:
        """Per hidden layer, the (in, out) relevance R_{i<-j}, mean over batch."""
        trace = self._forward_trace(x)
        logits = self.model.output(trace[-1][0])
        R_out = self._output_relevance(logits, target)
        n_hidden = self.model.num_layers
        per_layer = []
        for idx, (a, layer) in reversed(list(enumerate(trace))):
            z = layer(a)
            C = a.unsqueeze(2) * layer.weight.t().unsqueeze(0)      # (b, in, out)
            zs = _stabilized(z, self.eps)
            contrib = (C / zs.unsqueeze(1)) * R_out.unsqueeze(1)    # R_{i<-j}
            if idx < n_hidden:          # hidden layer -> keep its attributions
                per_layer.append(contrib)
            R_out = contrib.sum(dim=2)   # relevance of this layer's inputs
        per_layer.reverse()
        if aggregate:
            return [t.mean(dim=0).detach() for t in per_layer]
        return [t.detach() for t in per_layer]

    @staticmethod
    def _output_relevance(logits: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
        """Full relevance on the target output neuron, weighted by its probability."""
        probs = torch.softmax(logits, dim=1)
        if target is None:
            target = logits.argmax(dim=1)
        R = torch.zeros_like(probs)
        R.scatter_(1, target.unsqueeze(1), probs.gather(1, target.unsqueeze(1)))
        return R