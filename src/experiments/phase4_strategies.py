"""Phase 4 — Merge strategies: weighted (MKA) vs uniform vs SLERP (CoM).

Each strategy merges *one* identical pair from the same baseline with the same
NMI similarity, so results are directly comparable. We record the alpha each
strategy picked (evidence: SLERP and uniform pick the same pair but mix it
differently) and draw the comparison figure against the no-merge baseline.
"""

import json
from copy import deepcopy

import torch

from experiments.common import setup, make_similarity, make_merger
from evaluation.metrics import (accuracy_delta, evaluate, count_parameters)
from merging.merger import LayerMerger
from utils.viz import strategies_bars


def run_strategy(name: str, model, loaders, cfg):
    clone = deepcopy(model)                              # fair: same starting weights
    merger = LayerMerger(clone, make_similarity(cfg), make_merger(name, cfg),
                         threshold=cfg.merge_threshold)
    merger.set_calibration(loaders["val"], cfg.n_calibration_samples)
    done = merger.merge_once()
    acc = evaluate(clone, loaders["test"])
    row = {
        "strategy": name,
        "merged": done,
        "acc": acc,
        "params": count_parameters(clone),
        **accuracy_delta(evaluate(model, loaders["test"]), acc),
    }
    if done:
        row["pair"] = merger.last_merge["pair"]
        row["similarity"] = merger.last_merge["similarity"]
        row["alpha"] = merger.last_merge["alpha"]
    return row


def main():
    model, loaders, cfg = setup()
    baseline_acc = evaluate(model, loaders["test"])
    rows = [run_strategy(n, model, loaders, cfg) for n in cfg.merge_strategies]
    for r in rows:
        print(f"[phase4] {r['strategy']:<10} merged={r['merged']} "
              f"acc={r['acc']:.4f} (drop={r['acc_drop']:+.4f}) params={r['params']:,}"
              + (f" alpha={r.get('alpha', float('nan')):.3f} pair={r.get('pair')}"
                 if r["merged"] else ""))

    with open(cfg.artifact("phase4_results"), "w") as f:
        json.dump(rows, f, indent=2)
    torch.save(model.state_dict(), cfg.artifact("baseline_strategies"))

    strategies_bars([r["strategy"] for r in rows], [r["acc"] for r in rows],
                    [r["params"] for r in rows], baseline_acc,
                    cfg.artifact("strategies_fig"))
    print(f"[phase4] figure -> {cfg.artifact('strategies_fig')}")


if __name__ == "__main__":
    main()