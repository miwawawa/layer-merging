"""Phase 3 — Merging: compress the model by fusing the most-similar layer pairs.

Uses the paper-1 adaptive weight (alpha = S_lm). Records WHICH pair is fused
with WHAT affinity at every step (the evidence of *why* merging is safe), and
reports the compression-vs-accuracy trade-off as pairs are merged one at a
time, so we can pick the operating point for the final pipeline.
"""

import json

import torch

from experiments.common import setup, make_similarity
from evaluation.metrics import evaluate, count_parameters
from merging.factory import get_merger
from merging.merger import LayerMerger
from utils.viz import merge_curve


def main():
    model, loaders, cfg = setup()
    baseline_acc = evaluate(model, loaders["test"])
    n0 = count_parameters(model)

    merger = LayerMerger(model, make_similarity(cfg), get_merger("weighted"),
                         threshold=cfg.merge_threshold)
    merger.set_calibration(loaders["val"], cfg.n_calibration_samples)

    curve = [{"merges": 0, "acc": baseline_acc, "params": n0, "drop": 0.0,
              "pair": None, "alpha": None}]
    print(f"[phase3] merges=0    acc={baseline_acc:.4f} params={n0:,}")
    while merger.merge_once():
        acc = evaluate(model, loaders["test"])
        lm = merger.last_merge
        curve.append({
            "merges": len(curve),
            "acc": acc,
            "params": lm["params_out"],
            "drop": round(baseline_acc - acc, 6),
            "pair": lm["pair"],
            "alpha": lm["alpha"],
        })
        c = curve[-1]
        print(f"[phase3] merges={c['merges']} acc={acc:.4f} "
              f"drop={baseline_acc - acc:+.4f} params={lm['params_out']:,} "
              f"pair={lm['pair']} alpha={lm['alpha']:.3f} (S={lm['similarity']:.3f})")

    torch.save(model.state_dict(), cfg.artifact("merged_model"))
    with open(cfg.artifact("phase3_curve"), "w") as f:
        json.dump(curve, f, indent=2)
        print(f"[phase3] wrote {cfg.artifact('phase3_curve')}")

    merge_curve([c["merges"] for c in curve], [c["acc"] for c in curve],
                [c["params"] for c in curve], cfg.artifact("merge_curve"))
    print(f"[phase3] figure -> {cfg.artifact('merge_curve')}")


if __name__ == "__main__":
    main()