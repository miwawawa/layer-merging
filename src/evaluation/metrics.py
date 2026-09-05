"""Metrics: the small set of numbers every experiment reports.

We distinguish three jobs:

* classification quality  — :func:`evaluate` (accuracy, the headline metric)
* compression cost        — parameter count, FLOPs, on-disk size
* operational cost        — inference latency (wall-clock, machine-dependent
                            but perfectly fine for *relative* before/after
                            speedups inside the same run)
* sparsity                — zeroed-parameter share after pruning

:func:`summarize` bundles everything into one dict so phases and the final
evidence table all talk the same language.
"""

from typing import Dict, List, Optional

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import BaseNN


# --------------------------------------------------------------------------
# classification quality
# --------------------------------------------------------------------------

def evaluate(model: BaseNN, loader: DataLoader) -> float:
    """Mean accuracy over a loader (no gradients needed)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            if x.dim() == 4:
                x = x.view(x.size(0), -1)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def accuracy_delta(acc_before: float, acc_after: float) -> Dict[str, float]:
    """How much the operation cost / kept.

    * ``drop`` is the *absolute* accuracy loss (percentage points, 0..1 scale);
    * ``retention`` is the share of accuracy preserved (0..1).

    Used to "show the reduction side" of merging/pruning in the final table.
    """
    return {
        "acc_drop": round(acc_before - acc_after, 6),
        "retention": round(acc_after / acc_before, 6) if acc_before else 0.0,
    }


# --------------------------------------------------------------------------
# compression cost
# --------------------------------------------------------------------------

def count_parameters(model: BaseNN) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: BaseNN) -> int:
    """Multiply-add FLOPs of one forward pass (a Linear layer is 2*in*out +
    out bias adds). Independent of the actual input count — a single sample."""
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features + 1
    return int(flops)


def model_size_bytes(model: BaseNN) -> int:
    """RAM/disk footprint of the parameters (float32 = 4 B/param)."""
    return count_parameters(model) * 4


def inference_latency_ms(model: BaseNN, loader: DataLoader,
                         batches: int = 20, warmup: int = 2,
                         device: str = "cpu") -> float:
    """Median wall-clock time (ms) of one forward batch.

    Wall-clock is machine-dependent, but the *ratio* before/after on the same
    machine is a fair operational speedup. Warmup batches are discarded and the
    median (not mean) resists GC / OS spikes.
    """
    model.eval()
    x0, _ = next(iter(loader))
    if x0.dim() == 4:
        x0 = x0.view(x0.size(0), -1)
    with torch.no_grad():
        for _ in range(warmup):
            model(x0)
        times: List[float] = []
        for _ in range(batches):
            t = time.perf_counter()
            model(x0)
            times.append((time.perf_counter() - t) * 1e3)
    return float(np.median(times))


def nonzero_parameters(model: BaseNN) -> int:
    """Parameters that are not zeroed by pruning (sparsity metric)."""
    return sum(int((p != 0).sum().item()) for p in model.parameters())


def sparsity(model: BaseNN) -> float:
    total = count_parameters(model)
    return 1.0 - nonzero_parameters(model) / max(total, 1)


def weight_sparsity(model: BaseNN) -> float:
    """Share of zeroed *weights* (biases excluded — they are never masked)."""
    nz = sum(int((l.weight != 0).sum().item()) for l in model.linears)
    nz += int((model.output.weight != 0).sum().item())
    tot = sum(l.weight.numel() for l in model.linears)
    tot += model.output.weight.numel()
    return 1.0 - nz / max(tot, 1)


def compression_ratio(n_before: int, n_after: int) -> float:
    """How many times smaller the model became."""
    return n_before / max(n_after, 1)


# --------------------------------------------------------------------------
# one dict to rule them all
# --------------------------------------------------------------------------

def summarize(model: BaseNN, loader: Optional[DataLoader] = None,
              **overrides) -> Dict:
    """A complete snapshot of a model used by every results table.

    ``overrides`` lets callers add contextual fields (e.g. ``stage``,
    ``arch``, ``merges``) without mutating the model.
    """
    n_params = count_parameters(model)
    out: Dict = {
        "params": n_params,
        "flops": count_flops(model),
        "size_bytes": model_size_bytes(model),
        "size_mb": round(model_size_bytes(model) / 1e6, 4),
        "sparsity": round(sparsity(model), 6),
        "weight_sparsity": round(weight_sparsity(model), 6),
        "nonzero": nonzero_parameters(model),
        "layers": model.num_layers,
    }
    if loader is not None:
        out["acc"] = round(evaluate(model, loader), 6)
        out["latency_ms"] = round(inference_latency_ms(model, loader), 4)
    out.update(overrides)
    return out