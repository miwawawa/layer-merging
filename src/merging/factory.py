"""Merging factory: build a MergeStrategy by name (strategy/DRY)."""

from merging.protocol import MergeStrategy
from merging.slerp import SlerpMerge
from merging.uniform import UniformMerge
from merging.weighted import WeightedMerge


def get_merger(name: str, **kwargs) -> MergeStrategy:
    if name == "uniform":
        return UniformMerge()
    if name == "weighted":
        return WeightedMerge(**kwargs)
    if name == "slerp":
        return SlerpMerge(**kwargs)
    raise ValueError(f"Unknown merger '{name}'. Choose from ['uniform', 'weighted', 'slerp'].")