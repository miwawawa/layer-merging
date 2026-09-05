"""Phase 2 — Similarity: how redundant are the trained layers?

Paper-1 pipeline (activations -> diffusion map -> Gaussian NMI) vs the cosine
baseline, computed for BOTH model architectures. Activations are the post-
non-linearity outputs of each hidden layer (what MKA calls H^l). The finding:
both nets show exploitable-looking NMI redundancy in their hidden units, yet
only the residual net survives merging — high similarity is NOT sufficient for
mergeability, residual structure is what makes fusion information-preserving.
"""

import json
import os

import numpy as np

from experiments.common import setup, make_similarity
from similarity.cosine import CosineSimilarity
from utils.activations import extract_activations
from utils.viz import similarity_grid, similarity_heatmap


def show(model, tag: str, loaders, cfg) -> dict:
    acts = extract_activations(model, loaders["val"], cfg.n_calibration_samples)
    matrices = {
        "nmi": make_similarity(cfg).compute(acts),
        "cosine": CosineSimilarity().compute(acts),
    }
    for name, S in matrices.items():
        S_off = S - np.eye(S.shape[0])
        stem = cfg.files[f"similarity_{tag}_{name}"]   # e.g. similarity_residual_nmi
        np.savez(os.path.join(cfg.artifacts_dir, f"{stem}.npz"), S=S)
        similarity_heatmap(S, os.path.join(cfg.artifacts_dir, f"{stem}.png"),
                           title=f"Layer similarity — {tag} ({name})")
        print(f"[phase2] {stem:<24} off-diag mean={S_off.mean():.3f} max={S_off.max():.3f}")
    return {f"{tag}_{name}": np.round(S.tolist(), 4) for name, S in matrices.items()}


def main():
    model, loaders, cfg = setup()          # default = residual
    grid = {}
    grid.update(show(model, "residual", loaders, cfg))

    plain_path = os.path.join(cfg.artifacts_dir, cfg.files["baseline_plain"])
    if os.path.exists(plain_path):
        plain, plain_loaders, _ = setup(arch="plain")
        grid.update(show(plain, "plain", plain_loaders, cfg))

        # one 2x2 dashboard figure (residual/plain x NMI/cosine)
        grids_np = {}
        for key, S in grid.items():
            grids_np[key] = np.asarray(S)
        similarity_grid(grids_np, cfg.artifact("similarity_grid"))
        print(f"[phase2] dashboard -> {cfg.artifact('similarity_grid')}")

    with open(cfg.artifact("phase2_json"), "w") as f:
        json.dump({"mean_offdiag": {k: float(np.mean(
            np.asarray(v)[~np.eye(len(v), dtype=bool)])) for k, v in grid.items()}},
            f, indent=2)


if __name__ == "__main__":
    main()