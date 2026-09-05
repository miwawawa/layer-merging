"""Phase 1 — Baseline: train (or load) the dummy MLP on MNIST.

Goal: a correct, reproducible reference point that every later phase
(merging, pruning, similarity) is measured against — including its footprint
(FLOPs, size, latency) for the reduction story.
"""

import json

from experiments.common import setup
from evaluation.metrics import summarize


def main():
    model, loaders, cfg = setup()
    snap = summarize(model, loaders["test"])
    results = {"test_acc": round(snap["acc"], 6), "params": snap["params"],
               "layers": snap["layers"], "flops": snap["flops"],
               "size_mb": snap["size_mb"], "latency_ms": snap["latency_ms"]}
    with open(cfg.artifact("phase1_results"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[phase1] test_acc={snap['acc']:.4f} params={snap['params']:,} "
          f"layers={snap['layers']} flops={snap['flops']:,} "
          f"size={snap['size_mb']:.2f}MB latency={snap['latency_ms']:.3f}ms")


if __name__ == "__main__":
    main()