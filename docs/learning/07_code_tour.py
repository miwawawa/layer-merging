#!/usr/bin/env python3
"""docs/learning/07_code_tour.py — a guided walk through the repository.

Run from anywhere in the repo:  python docs/learning/07_code_tour.py

Prints a file-by-file narrative (no side effects, no imports beyond stdlib) so
a newcomer can follow the exact order the codebase builds up: data -> models ->
training -> similarity -> merging -> pruning -> experiments -> evidence.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SECTIONS = [
    ("0. Where things live", [
        ("src/", "everything importable lives here; scripts/ and notebooks/ only drive it"),
        ("src/config/", "single source of truth (Config) — run everything from here"),
        ("src/data/", "MNIST loader + train/val split (val = 6000 of 60k)"),
        ("src/models/", "SimpleMLP (plain) and ResidualMLP (residual stream)"),
        ("src/training.py", "the single trainer used by every experiment (adam, 1e-3, 6 epochs)"),
        ("src/similarity/", "diffusion map + Gaussian NMI + cosine (paper-1 metric)"),
        ("src/merging/", "pair search, alpha, and the 3 fuse strategies (paper 1+2)"),
        ("src/pruning/", "LRP analyzer + importance scorers + pruner (paper 3)"),
        ("src/evaluation/", "params/FLOPs/footprint/latency/sparsity + summarize()"),
        ("src/utils/", "seeding, activation capture, all matplotlib figures"),
        ("src/experiments/", "phase1..6b run scripts, each writes artifacts + a figure"),
        ("tests/", "mirrors src one-to-one; tiny models so the suite runs in seconds"),
    ]),
    ("1. Data + model + training (the baseline everything references)", [
        ("src/data/mnist.py", "download once under data/, 28x28 flatten inside loaders"),
        ("src/models/base.py", "BaseNN: linears/output/num_layers + remove/insert linears"),
        ("src/models/residual_mlp.py", "h <- h + act(W h); branch_scale=0.1 keeps layers distinct"),
        ("src/training.py", "the single trainer used by every experiment (adam, 1e-3, 6 epochs)"),
        ("src/experiments/phase1_baseline.py", "trains+caches both baselines, prints evidence table"),
    ]),
    ("2. Similarity (phase 2: is layer k redundant with layer m?)", [
        ("src/utils/activations.py", "extract_activations: the SAME 512 samples every phase"),
        ("src/similarity/diffusion.py", "diffusion embedding over the activation point-cloud"),
        ("src/similarity/gaussian_mi.py", "closed-form Gaussian MI + NMI on the embeddings"),
        ("src/similarity/nmi.py", "composes DiffusionMap + gaussian_nmi into a similarity"),
        ("src/similarity/cosine.py", "second view on the same activations (contrast evidence)"),
        ("src/experiments/phase2_similarity.py", "writes similarity_all.png + phase2 JSON"),
    ]),
    ("3. Merging (phase 3+4: fusing the redundant layers)", [
        ("src/merging/merger.py", "greedy argmax pair over S, threshold gate, fuse, merge_log"),
        ("src/merging/weighted.py", "alpha = clip(S_lm, 0.1, 0.9); convex combination (MKA)"),
        ("src/merging/uniform.py", "alpha pinned to 0.5 (the naive baseline)"),
        ("src/merging/slerp.py", "spherical interpolation with near-collinear fallback (CoM)"),
        ("src/experiments/phase3_merging.py", "merge curve: acc/params vs number of merges"),
        ("src/experiments/phase4_strategies.py", "same pair, three mixers, apples-to-apples"),
        ("src/experiments/phase4b_alpha_sweep.py", "alpha on a grid; pair choice dominates alpha"),
    ]),
    ("4. Pruning (phase 5: SparC3 on the plain graph)", [
        ("src/pruning/lrp.py", "LRP: eps-stabilized backwards relevance (R_{i<-j})"),
        ("src/pruning/importance.py", "aggregate R over reference samples (Eq.1) -> per-weight"),
        ("src/pruning/pruner.py", "rank + boolean mask + apply_prune; row-wise variant"),
        ("src/experiments/phase5_pruning.py", "lrp vs magnitude vs random over the keep-ratio sweep"),
    ]),
    ("5. The two-track pipeline + honest boundaries (phase 6 + 6b)", [
        ("src/experiments/phase6_pipeline.py", "track A residual-merge, track B plain-prune, evidence table"),
        ("src/experiments/phase6b_hybrid.py", "why one model can't do both: merge collapses plain; LRP undefined on residual"),
    ]),
    ("6. Evidence you can reproduce (scripts)", [
        ("scripts/run_phases.py", "runs phase1..6b in order (make phases)"),
        ("scripts/reproduce.py", "per-seed reruns (42,7,2026) -> mean+std (make reproduce)"),
        ("scripts/build_notebooks.py", "regenerates every notebook from the SAME artifacts"),
        ("scripts/run_notebooks.py", "executes every notebook (verifies the Colab path)"),
        ("scripts/smoke_test.py", "module-level import/API smoke (make smoke)"),
        ("Makefile", "make verify = pytest + smoke, the pre-push gate"),
    ]),
]


def main() -> int:
    for title, items in SECTIONS:
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        for rel, note in items:
            full = os.path.join(ROOT, rel)
            exists = os.path.exists(full)
            print(f"  {rel:<42} {('✓' if exists else '?!')}  {note}")
    print("\n" + "=" * 72)
    print("Then read docs/learning/ in order: 00 roadmap, 01 primer,")
    print("02/03/04 per-paper numbers, 05 related work, 06 code guide.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())