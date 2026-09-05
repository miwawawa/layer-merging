"""Phase 6 — Full pipeline: two complementary compression tracks.

MKA-style layer merging with the α=S_lm weighted strategy needs the residual
architecture (its redundancy is what makes merging safe) — track A.
SparC3-style LRP pruning is exact only on a plain feed-forward graph — track B.
Each track reports a FULL metric snapshot before/after (accuracy, params,
FLOPs, footprint, latency, sparsity) so the *reduction side* of merging is
quantified, not just accuracy. (Merging-then-pruning in a single hybrid is not
reproducible in one MLP: merging collapses a plain graph before LRP can help —
an honest documented limitation.)
"""

import csv
import json
from copy import deepcopy

import torch

from experiments.common import setup, make_similarity, make_merger
from evaluation.metrics import nonzero_parameters, summarize
from merging.merger import LayerMerger
from pruning.importance import LRPImportance
from pruning.pruner import prune_model_global
from utils.viz import gantt, results_table


def track_merge(cfg, loaders):
    """Residual model, greedy α=S_lm merging (MKA)."""
    model, loaders, cfg = setup()
    before = summarize(model, loaders["test"], stage="Residual baseline",
                       arch="residual", merges=0)
    merger = LayerMerger(model, make_similarity(cfg), make_merger("weighted", cfg),
                         threshold=cfg.merge_threshold, cfg=cfg)
    merger.set_calibration(loaders["val"], cfg.n_calibration_samples)
    n_merges = merger.merge_all()

    after = summarize(model, loaders["test"], stage="Layer-merging track (MKA)",
                      arch="residual", merges=n_merges)
    alpha_used = round(merger.last_merge["alpha"], 4) if n_merges else None
    print(f"[phase6][merge] baseline acc={before['acc']:.4f} params={before['params']:,} "
          f"-> merged acc={after['acc']:.4f} params={after['params']:,} "
          f"({before['params'] / after['params']:.2f}x, {n_merges} merges, "
          f"last alpha={alpha_used})")
    torch.save(model.state_dict(), cfg.artifact("pipeline_merged"))
    return _evidence_rows(before, after, "Layer merging (MKA)"), after


def track_prune(cfg, loaders):
    """Plain model, global LRP pruning (SparC3)."""
    model, loaders, cfg = setup(arch="plain")
    before = summarize(model, loaders["test"], stage="Plain baseline",
                       arch="plain", merges=0)
    imp = LRPImportance(eps=cfg.lrp_eps).score(model, loaders["val"],
                                               cfg.n_calibration_samples)
    clone = deepcopy(model)
    prune_model_global(clone, imp, keep_ratio=cfg.prune_keep_ratio)
    after = summarize(clone, loaders["test"],
                      stage=f"LRP-pruning track (SparC3 @{cfg.prune_keep_ratio:.0%})",
                      arch="plain", merges=0, nonzero=nonzero_parameters(clone))
    print(f"[phase6][prune] baseline acc={before['acc']:.4f} params={before['params']:,} "
          f"-> pruned@{cfg.prune_keep_ratio:.2f} acc={after['acc']:.4f} "
          f"sparsity={after['sparsity']:.2f} retention={after['acc'] / before['acc']:.2%}")
    torch.save(clone.state_dict(), cfg.artifact("pipeline_pruned"))
    return _evidence_rows(before, after, "LRP pruning (SparC3)"), after


def _evidence_rows(before: dict, after: dict, title: str):
    """before/after rows incl. savings, for the evidence table + summary JSON."""
    b, a = before, after
    drop = a["acc"] - b["acc"]                                   # sign: - is loss
    base_row = {"title": f"{title} — before", "acc": b["acc"], "params": b["params"],
                "flops": b["flops"], "size_mb": b["size_mb"], "latency_ms": b["latency_ms"],
                "sparsity": b["sparsity"], "layers": b["layers"]}
    after_row = {"title": f"{title} — after", "acc": a["acc"], "params": a["params"],
                 "flops": a["flops"], "size_mb": a["size_mb"], "latency_ms": a["latency_ms"],
                 "sparsity": a["sparsity"], "layers": a["layers"],
                 "acc_delta": drop, "retention": a["acc"] / b["acc"],
                 "params_savings": 1 - a["params"] / b["params"],
                 "flops_savings": 1 - a["flops"] / b["flops"],
                 "size_savings": 1 - a["size_mb"] / b["size_mb"],
                 "speedup": b["latency_ms"] / a["latency_ms"]}
    return [base_row, after_row]


def main():
    model, loaders, cfg = setup()
    merge_rows, merged = track_merge(cfg, loaders)
    prune_rows, pruned = track_prune(cfg, loaders)
    rows = [merge_rows[0], merge_rows[1], prune_rows[0], prune_rows[1]]

    # every metric snapshot as machine-readable records (CSV + JSON)
    def snap(track: str, stage: str, row: dict, note: str = "") -> dict:
        return {"track": track, "stage": stage, "note": note,
                **{k: row[k] for k in ("acc", "params", "flops", "size_mb",
                                       "latency_ms", "sparsity")}}

    snapshots = [
        snap("merge (MKA)", "baseline", rows[0]),
        snap("merge (MKA)", "merged", rows[1],
             f"{merged['merges']} layer pairs fused"),
        snap("prune (SparC3)", "baseline", rows[2]),
        snap("prune (SparC3)", "pruned", rows[3],
             f"keep {cfg.prune_keep_ratio:.0%}, zeroed weights"),
    ]
    with open(cfg.artifact("final_json"), "w") as f:
        json.dump(snapshots, f, indent=2)
    with open(cfg.artifact("final_csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(snapshots[0]))
        w.writeheader()
        w.writerows(snapshots)

    results_table(rows, cfg.artifact("results_table"),
                  title="Compression evidence — each track against its own baseline")
    print(f"[phase6] evidence table -> {cfg.artifact('results_table')}")

    for r in rows:
        print(f"[phase6] {r['title'][:40]:<40} acc={r['acc']:.4f} "
              f"params={r['params']:,} sparsity={r['sparsity']:.2f}")
    print(f"[phase6] merge track: params {merged['params']:,} "
          f"({rows[1]['params_savings']:.1%} saved) "
          f"flops {merged['flops']:,} ({rows[1]['flops_savings']:.1%}) "
          f"latency {rows[0]['latency_ms']:.3f}->{rows[1]['latency_ms']:.3f} ms "
          f"({rows[1]['speedup']:.2f}x) retention {rows[1]['retention']:.2%}")
    print(f"[phase6] prune track: sparsity {pruned['sparsity']:.2f} "
          f"retention {rows[3]['retention']:.2%} "
          f"(params/FLOPs unchanged — zeroing needs sparse kernels)")

    # Gantt for the project-management section of the report (data in cfg).
    gantt(list(cfg.gantt_tasks), list(cfg.gantt_starts), list(cfg.gantt_durations),
          cfg.artifact("gantt"))


if __name__ == "__main__":
    main()