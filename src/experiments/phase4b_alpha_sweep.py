"""Phase 4b — α-ablation: how sensitive is MKA merging to the mixing weight?

The weighted strategy picks α = S_lm (clamped). Here we fix the *same* best
pair for a grid of α values, so we can see the accuracy/parameter/α trade-off
directly. Rows are comparable: every α starts from the same cached residual
baseline and fuses the same pair; only the convex-combination weight changes.
"""

import json
from copy import deepcopy

from experiments.common import setup, make_similarity
from evaluation.metrics import count_parameters, evaluate
from merging.merger import LayerMerger
from merging.weighted import WeightedMerge
from utils.viz import alpha_sweep_plot


class FixedAlphaMerge(WeightedMerge):
    """Mix with a caller-chosen alpha instead of S_lm (the ablation control).

    Bounds are [0, 1] so the clip is a no-op; ``weight`` is overridden to
    return exactly the fixed value.
    """

    def __init__(self, alpha: float):
        super().__init__(min_alpha=0.0, max_alpha=1.0)
        self._fixed = alpha

    def weight(self, similarity: float) -> float:
        return self._fixed


def main():
    model, loaders, cfg = setup()
    baseline_acc = evaluate(model, loaders["test"])
    sim = make_similarity(cfg)
    rows = []
    for alpha in cfg.alpha_sweep:
        clone = deepcopy(model)                        # same starting weights
        merger = LayerMerger(clone, sim, FixedAlphaMerge(alpha),
                             threshold=cfg.merge_threshold, cfg=cfg)
        merger.set_calibration(loaders["val"], cfg.n_calibration_samples)
        done = merger.merge_once()
        acc = evaluate(clone, loaders["test"])
        rows.append({
            "alpha": float(alpha),
            "merged": done,
            "acc": acc,
            "acc_delta": acc - baseline_acc,
            "params": count_parameters(clone),
            "pair": merger.last_merge["pair"] if done else None,
            "similarity": merger.last_merge["similarity"] if done else None,
        })
        print(f"[phase4b] alpha={alpha:.2f} merged={done} "
              f"acc={acc:.4f} (delta={acc - baseline_acc:+.4f}) "
              f"params={rows[-1]['params']:,}")

    with open(cfg.artifact("alpha_sweep_results"), "w") as f:
        json.dump({"baseline_acc": baseline_acc, "sweep": rows}, f, indent=2)
    alpha_sweep_plot([r["alpha"] for r in rows], [r["acc"] for r in rows],
                     baseline_acc, cfg.artifact("alpha_sweep_fig"))
    print(f"[phase4b] figure -> {cfg.artifact('alpha_sweep_fig')}")


if __name__ == "__main__":
    main()