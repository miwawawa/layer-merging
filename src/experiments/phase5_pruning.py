"""Phase 5 — Pruning: LRP relevance (paper-3) vs magnitude vs random.

Sweeps sparsity on the *unpruned* baseline and records accuracy + retention,
showing that LRP-based attribution pruning preserves task accuracy much better
than random (or magnitude) at the same sparsity. Also saves the input
relevance map and an accuracy-vs-sparsity figure for the report.
"""

import json
from copy import deepcopy

import numpy as np

from experiments.common import setup
from evaluation.metrics import accuracy_delta, evaluate, sparsity
from pruning.importance import LRPImportance, MagnitudeImportance
from pruning.lrp import LRPAnalyzer
from pruning.pruner import prune_model_global
from utils.viz import pruning_curves, relevance_map


def random_importances(model, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for layer in list(model.linears) + [model.output]:
        out.append(np.concatenate([rng.random(layer.weight.numel()),
                                   rng.random(layer.bias.numel())]))
    return out


def main():
    model, loaders, cfg = setup(arch="plain")   # plain graph keeps LRP exact
    baseline_acc = evaluate(model, loaders["test"])

    lrp = LRPImportance(eps=cfg.lrp_eps)
    imp_lrp = lrp.score(model, loaders["val"], cfg.n_calibration_samples)
    imp_mag = MagnitudeImportance().score(model, loaders["val"], cfg.n_calibration_samples)
    imp_rand = random_importances(model, cfg.random_seed)

    rows = []
    for keep in cfg.prune_sweep:
        for name in cfg.prune_methods:
            imp = {"lrp": imp_lrp, "magnitude": imp_mag, "random": imp_rand}[name]
            clone = deepcopy(model)
            prune_model_global(clone, imp, keep_ratio=keep)
            acc = evaluate(clone, loaders["test"])
            rows.append({"keep_ratio": keep, "method": name, "acc": acc,
                         "sparsity": sparsity(clone),
                         **accuracy_delta(baseline_acc, acc)})
            print(f"[phase5] keep={keep:<4} {name:<9} acc={acc:.4f} "
                  f"drop={baseline_acc - acc:+.4f} sparsity={sparsity(clone):.3f}")

    # How different is LRP attribution from plain magnitude? (findings)
    layers = list(model.linears) + [model.output]
    flat_lrp = np.concatenate([imp[:l.weight.numel()]
                               for imp, l in zip(imp_lrp, layers)])
    flat_mag = np.concatenate([imp[:l.weight.numel()]
                               for imp, l in zip(imp_mag, layers)])
    corr = np.corrcoef(flat_lrp, flat_mag)[0, 1]
    print(f"[phase5] corr(LRP-attribution, magnitude) = {corr:.3f}")

    # Accuracy-vs-sparsity figure (lines per method), the SparC3 study.
    sparsities = {m: [] for m in cfg.prune_methods}
    accs = {m: [] for m in cfg.prune_methods}
    for r in rows:
        sparsities[r["method"]].append(r["sparsity"])
        accs[r["method"]].append(r["acc"])
    op_sp = next((r["sparsity"] for r in rows
                  if r["method"] == "lrp" and r["keep_ratio"] == cfg.prune_keep_ratio),
                 sparsities["lrp"][-1])
    pruning_curves(sparsities, accs, cfg.artifact("pruning_curves"),
                   operating_pts={"lrp": op_sp})
    print(f"[phase5] figure -> {cfg.artifact('pruning_curves')}")

    # Input relevance map for one sample (nice figure + sanity check)
    analyzer = LRPAnalyzer(model, eps=cfg.lrp_eps)
    x, y = next(iter(loaders["val"]))
    r_input = analyzer.input_relevance(x[:1], target=y[:1]).numpy()
    relevance_map(r_input / (np.abs(r_input).max() + 1e-12),
                  cfg.artifact("lrp_input_relevance"))

    with open(cfg.artifact("phase5_results"), "w") as f:
        json.dump(rows, f, indent=2)
        print(f"[phase5] wrote {cfg.artifact('phase5_results')}")


if __name__ == "__main__":
    main()