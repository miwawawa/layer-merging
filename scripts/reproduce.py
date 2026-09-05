"""Multi-seed reproduction: aggregate the two tracks' headline numbers.

Trains the residual + plain baselines for every ``Config.seeds`` seed, runs the
full merging (MKA, α=S_lm) and LRP-pruning tracks, and reports MEAN +/- STD
per metric — the honest, robust numbers for the report (single-seed results can
be lucky/unlucky). Baselines are cached per seed, so re-runs are cheap.

Outputs (under the configured artifacts dir):
  reproduce_summary.csv   one row per seed/track + the aggregate rows
  reproduce_summary.json  machine-readable version
  reproduce_table.png     evidence table image (mean +/- std)
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config.default import Config
from evaluation.metrics import accuracy_delta, count_parameters, summarize
from experiments.common import make_merger, make_similarity, setup
from merging.merger import LayerMerger
from pruning.importance import LRPImportance
from pruning.pruner import prune_model_global


def run_seed(seed: int, cfg: Config, keep_ratio: float) -> dict:
    """One seed: both tracks, full before/after metric snapshots."""
    out = {"seed": seed}

    # ---- track A: layer merging on the residual model --------------------
    model, loaders, cfg = setup(seed=seed)
    base = summarize(model, loaders["test"])
    merger = LayerMerger(model, make_similarity(cfg), make_merger("weighted", cfg),
                         threshold=cfg.merge_threshold)
    merger.set_calibration(loaders["val"], cfg.n_calibration_samples)
    n_merges = merger.merge_all()
    merged = summarize(model, loaders["test"], merges=n_merges)
    out["merge_merges"] = n_merges
    out.update(_pair(base, merged, "merge"))
    print(f"[reproduce] seed={seed} merge: {base['acc']:.4f} -> {merged['acc']:.4f} "
          f"params {base['params']:,} -> {merged['params']:,} "
          f"({base['params'] / merged['params']:.2f}x) retention="
          f"{merged['acc'] / base['acc']:.2%}")

    # ---- track B: LRP pruning on the plain model --------------------------
    plain, plain_loaders, cfg = setup(arch="plain", seed=seed)
    p_base = summarize(plain, plain_loaders["test"])
    imp = LRPImportance(eps=cfg.lrp_eps).score(plain, plain_loaders["val"],
                                               cfg.n_calibration_samples)
    prune_model_global(plain, imp, keep_ratio=keep_ratio)
    pruned = summarize(plain, plain_loaders["test"])
    out["prune_keep_ratio"] = keep_ratio
    out.update(_pair(p_base, pruned, "prune", latency=False))
    print(f"[reproduce] seed={seed} prune: {p_base['acc']:.4f} -> {pruned['acc']:.4f} "
          f"sparsity={pruned['sparsity']:.2f} retention={pruned['acc'] / p_base['acc']:.2%}")
    return out


def _pair(before: dict, after: dict, tag: str, latency: bool = True) -> dict:
    """Metric deltas + savings for one track (kept in one place)."""
    d = {
        f"{tag}_params": after["params"],
        f"{tag}_params_savings": 1 - after["params"] / before["params"],
        f"{tag}_flops": after["flops"],
        f"{tag}_flops_savings": 1 - after["flops"] / before["flops"],
        f"{tag}_size_mb": after["size_mb"],
        f"{tag}_sparsity": after["sparsity"],
        f"{tag}_layers": after["layers"],
        f"{tag}_retention": after["acc"] / before["acc"],
        f"{tag}_acc_delta": after["acc"] - before["acc"],
        f"{tag}_acc_after": after["acc"],
        f"{tag}_acc_before": before["acc"],
    }
    if latency:
        d[f"{tag}_latency_ms"] = after["latency_ms"]
        d[f"{tag}_speedup"] = before["latency_ms"] / after["latency_ms"]
    return d


def _aggregate(rows: list) -> dict:
    """Mean +/- std across seeds for every numeric key."""
    merged = {}
    keys = [k for k in rows[0] if k not in ("seed", "prune_keep_ratio")]
    for k in keys:
        vals = np.array([r[k] for r in rows])
        merged[f"{k}_mean"] = round(float(vals.mean()), 6)
        merged[f"{k}_std"] = round(float(vals.std(ddof=1)), 6)
    merged["n_seeds"] = len(rows)
    return merged


def main():
    cfg = Config()
    cfg.ensure_artifacts()
    keep = cfg.prune_keep_ratio
    rows = [run_seed(int(s), cfg, keep) for s in cfg.seeds]

    with open(cfg.artifact("reproduce_json"), "w") as f:
        json.dump({"per_seed": rows, "aggregate": _aggregate(rows)}, f, indent=2)

    agg = _aggregate(rows)
    with open(cfg.artifact("reproduce_csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["per_seed"] + list(rows[0]) + list(agg))
        w.writeheader()
        for r in rows:
            w.writerow({"per_seed": "seed", **r})
        w.writerow({"per_seed": "aggregate", **agg})

    # console summary (mean +/- std) — the numbers you paste into the report
    print()
    print("=== reproduction summary (mean +/- std over seeds "
          f"{tuple(cfg.seeds)}) ===")
    for k in ("merge_retention", "merge_params_savings", "merge_flops_savings",
              "merge_speedup", "merge_merges_mean", "prune_retention",
              "prune_sparsity", "prune_acc_delta"):
        if f"{k}_mean" in agg:
            print(f"  {k:<24} {agg[f'{k}_mean']:.4f} +/- {agg[f'{k}_std']:.4f}")
    for s in rows:
        print(f"  seed {s['seed']:<5} merge acc "
              f"{s['merge_acc_before']:.4f}->{s['merge_acc_after']:.4f} "
              f"params_saved {s['merge_params_savings']:.1%} | prune acc "
              f"{s['prune_acc_before']:.4f}->{s['prune_acc_after']:.4f} "
              f"sparsity {s['prune_sparsity']:.2f}")

    # evidence table image of the aggregates
    from utils.viz import results_table
    table_rows = [{
        "metric": k.replace("_", " "),
        "mean": agg[f"{k}_mean"],
        "std": agg[f"{k}_std"],
    } for k in ("merge_retention", "merge_params_savings", "merge_flops_savings",
                "merge_speedup", "prune_retention", "prune_sparsity")
      if f"{k}_mean" in agg]
    rows_fmt = [{"metric": r["metric"], "mean (std)":
                 f"{r['mean']:.4f} (+/- {r['std']:.4f})"} for r in table_rows]
    results_table(rows_fmt, cfg.artifact("reproduce_table"),
                  order=["metric", "mean (std)"],
                  title="Multi-seed reproduction — mean +/- std")
    print(f"[reproduce] wrote {cfg.artifact('reproduce_csv')} and "
          f"{cfg.artifact('reproduce_table')}")


if __name__ == "__main__":
    main()