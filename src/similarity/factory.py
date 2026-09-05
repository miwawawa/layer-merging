"""Similarity factory: build a LayerSimilarity by name (strategy/DRY)."""

from similarity.cosine import CosineSimilarity
from similarity.nmi import NmiSimilarity
from similarity.protocol import LayerSimilarity


def get_similarity(name: str, **kwargs) -> LayerSimilarity:
    if name == "nmi":
        return NmiSimilarity(**kwargs)
    if name == "cosine":
        return CosineSimilarity()
    raise ValueError(f"Unknown similarity '{name}'. Choose from ['nmi', 'cosine'].")