"""Phase 6b — cross-track feasibility: WHY merging and pruning stay separate.

Two compression tracks (phase 6) exist for a structural reason:
  * MKA merging is only safe on a residual graph (redundancy to fuse);
  * SparC3 LRP pruning is only *defined* on a plain chain (its backward rule
    walks a plain linear graph — a residual stream has no clean z_j denom).

Phase 6b makes both negatives explicit and honest:
  track D  plain    -> greedy merge  -> LRP prune 0.5
  track C  residual -> LRP scoring attempt (expected: structurally undefined)

The plain track collapses at the merge step (no residual redundancy) and the
follow-up pruning cannot recover it; the residual track shows LRP is not
applicable (skip connections break the relevance-backward rule). Both findings
are evidence, not noise — they justify keeping the two pipelines separate.
"""

import json
from copy import deepcopy

from experiments.common import setup, make_similarity, make_merger
from evaluation.metrics import count_parameters, evaluate
from merging.merger import LayerMerger
from pruning.importance import LRPImportance
from pruning.pruner import prune_model_global
from utils.viz import hybrid_bars

LRP_UNSUPPORTED_REASON = (
    "LRP traces a plain Linear chain; the residual stream (h <- h + act(W h)) "
    "has no plain forward path, so weight relevance is undefined here."
)


def _row(stage: str, arch: str, acc, params):
    return {"stage": stage, "arch": arch,
            "acc": None if acc is None else round(float(acc), 6),
            "params": params}


def _merge(model, loaders, cfg):
    merger = LayerMerger(model, make_similarity(cfg), make_merger("weighted", cfg),
                         threshold=cfg.merge_threshold, cfg=cfg)
    merger.set_calibration(loaders["val"], cfg.n_calibration_samples)
    return merger.merge_all()


def track_d(loaders, cfg):
    """Plain graph: merge-then-prune should fail loudly (negative #1)."""
    model, loaders, cfg = setup(arch="plain")
    base_params = count_parameters(model)
    base = evaluate(model, loaders["test"])
    merged = _merge(model, loaders, cfg)
    merged_params = count_parameters(model)
    acc_merged = evaluate(model, loaders["test"])
    pruned = deepcopy(model)
    imp = LRPImportance(eps=cfg.lrp_eps).score(pruned, loaders["val"],
                                               cfg.n_calibration_samples)
    prune_model_global(pruned, imp, keep_ratio=cfg.prune_keep_ratio)
    acc_pruned = evaluate(pruned, loaders["test"])
    print(f"[phase6b][D] plain: {base:.4f} -> merged {merged} merges {acc_merged:.4f} "
          f"-> pruned {acc_pruned:.4f}")
    return [_row("baseline", "plain", base, base_params),
            _row(f"merged ({merged} pairs)", "plain", acc_merged, merged_params),
            _row(f"merged + LRP{cfg.prune_keep_ratio:.0%}", "plain", acc_pruned,
                 count_parameters(pruned))] + [LRP_UNSUPPORTED_REASON]


def track_c(loaders, cfg):
    """Residual graph: LRP is structurally undefined (negative #2, shown honestly)."""
    model, loaders, cfg = setup()
    base_params = count_parameters(model)
    base = evaluate(model, loaders["test"])
    merged = _merge(model, loaders, cfg)
    merged_params = count_parameters(model)
    acc_merged = evaluate(model, loaders["test"])
    print(f"[phase6b][C] residual: {base:.4f} -> merged {merged} merges {acc_merged:.4f}")

    reason = LRP_UNSUPPORTED_REASON
    try:
        LRPImportance(eps=cfg.lrp_eps).score(model, loaders["val"],
                                             cfg.n_calibration_samples)
        reason = None                                  # would be a surprise
    except (RuntimeError, ValueError):
        pass                                          # expected: structurally undefined
    return [_row("baseline", "residual", base, base_params),
            _row(f"merged ({merged} pairs)", "residual", acc_merged,
                 merged_params),
            reason]


def main():
    _, loaders, cfg = setup()
    d_result, c_result = track_d(loaders, cfg), track_c(loaders, cfg)
    d_rows, d_reason = d_result[:3], d_result[3]
    c_rows, c_reason = c_result[:2], c_result[-1]
    with open(cfg.artifact("hybrid_results"), "w") as f:
        json.dump({
            "prune_keep_ratio": cfg.prune_keep_ratio,
            "rows": d_rows + c_rows,
            "notes": {
                "plain": ("merging a plain graph collapses it; LRP-pruning the "
                          "collapsed model cannot recover accuracy"),
                "residual": d_reason,
                "residual_lrp": c_reason,
            },
        }, f, indent=2)
    hybrid_bars(d_rows + c_rows, cfg.artifact("hybrid_fig"))
    print(f"[phase6b] evidence -> {cfg.artifact('hybrid_results')} "
          f"| figure -> {cfg.artifact('hybrid_fig')}")
    for r in d_rows + c_rows:
        acc = "n/a" if r["acc"] is None else f"{r['acc']:.4f}"
        print(f"[phase6b] {r['stage']:<28} {r['arch']:<8} "
              f"acc={acc} params={r['params']:,}")
    print(f"[phase6b] note (residual lrp): {c_reason}")


if __name__ == "__main__":
    main()