"""Run all six experiment phases in order (the full reproduction).

Usage:  python scripts/run_phases.py            (from anywhere in the repo)

Reads Configuration from .env/environment, downloads/eagerly uses MNIST under
the configured data dir, and writes every artifact under the configured
artifacts dir — no CWD assumptions.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

PHASES = [
    "phase1_baseline",
    "phase2_similarity",
    "phase3_merging",
    "phase4_strategies",
    "phase5_pruning",
    "phase6_pipeline",
]


def run(name: str) -> None:
    print(f"\n=== {name} ===")
    mod = __import__(f"experiments.{name}", fromlist=["main"])
    mod.main()


if __name__ == "__main__":
    only = sys.argv[1:]                      # optional: limit to given phases
    for name in only or PHASES:
        if name not in PHASES:
            print(f"unknown phase {name!r}; choices: {PHASES}")
            sys.exit(1)
        run(name)
    print("\n=== all phases done ===")