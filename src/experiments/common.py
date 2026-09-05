"""Shared setup/run helpers for the phase scripts (DRY).

Every phase entry point goes through :func:`setup`, which is the ONLY place
that reads ``Config`` and touches the filesystem, so the phases stay small
and consistent. Models are built by :func:`build_model` from ``cfg`` — no
model-construction defaults are duplicated anywhere else.
"""

import os

import torch

from config.default import Config
from data.mnist import get_mnist
from evaluation.metrics import count_parameters
from merging.factory import get_merger
from models.mlp import SimpleMLP
from similarity.factory import get_similarity
from utils.seed import set_seed


def make_similarity(cfg: Config):
    """Paper-1 similarity with the tuned config parameters (single source)."""
    return get_similarity("nmi", sigma=cfg.kernel_sigma,
                          n_dims=cfg.n_manifold_dims, time=cfg.diffusion_time)


def make_merger(name: str, cfg: Config):
    """Merge strategy built from the configured alpha bounds."""
    return get_merger(name, min_alpha=cfg.merge_min_alpha, max_alpha=cfg.merge_max_alpha)


def build_model(cfg: Config):
    """Construct the configured architecture from ``cfg`` (both archs share shape).

    'residual' reproduces the merging story; 'plain' is the non-residual graph
    used for the LRP/pruning studies where a plain feed-forward makes LRP exact.
    """
    if cfg.model_arch == "residual":
        from models.residual_mlp import ResidualMLP
        return ResidualMLP(input_dim=cfg.input_dim, hidden_dims=cfg.hidden_dims,
                           num_classes=cfg.num_classes, dropout=cfg.dropout,
                           branch_scale=cfg.branch_scale)
    from models.mlp import SimpleMLP
    return SimpleMLP(input_dim=cfg.input_dim, hidden_dims=cfg.hidden_dims,
                     num_classes=cfg.num_classes, dropout=cfg.dropout)


def setup(model_save: str | None = None, arch: str | None = None,
          seed: int | None = None) -> "tuple[BaseNN, dict, Config]":
    """Build config, seed, loaders, and a baseline model (trained if needed).

    ``model_save`` overrides the artifact name; ``arch`` overrides
    ``cfg.model_arch``; ``seed`` retrains with that seed (cached separately).
    Baselines are cached under ``cfg.artifacts_dir`` so re-runs skip training.
    All paths come from ``cfg`` (absolute under the project root), so CWD
    never matters.
    """
    cfg = Config()
    if arch:
        cfg.model_arch = arch
    if seed is not None:
        cfg.seed = seed
    set_seed(cfg.seed)
    cfg.ensure_artifacts()
    loaders = get_mnist(cfg.data_dir, cfg.batch_size, cfg.val_size,
                        mirror=cfg.mnist_mirror)
    if model_save is None:
        arch = "plain" if cfg.model_arch == "plain" else "residual"
        default_seed = Config().seed
        # canonical baselines keep the plain name; other seeds get a tag
        if seed is None or seed == default_seed:
            model_save = (cfg.files["baseline_plain"] if arch == "plain"
                          else cfg.files["baseline_residual"])
        else:
            model_save = f"baseline_{arch}_seed{seed}.pt"
    model_path = os.path.join(cfg.artifacts_dir, model_save)

    model = build_model(cfg)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"[setup] loaded baseline from {model_path} ({count_parameters(model):,} params)")
    else:
        print("[setup] training baseline...")
        from training import train_model
        train_model(model, loaders, cfg)
        torch.save(model.state_dict(), model_path)
        print(f"[setup] saved baseline to {model_path}")
    return model, loaders, cfg