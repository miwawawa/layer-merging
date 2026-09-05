"""Single source of truth for every tunable in the project.

Policy (see docs/guidelines_checklist.md, "no hardcoding" checks):

* Every experiment-relevant parameter lives HERE as a dataclass field.
* Dataclass defaults are the documented baseline values used in the report.
* Every field can be overridden through the environment / a ``.env`` file:
  the env var name is ``LAYER_MERGING_<FIELD_NAME.upper()>`` (values are parsed
  with ``ast.literal_eval``, so tuples/floats/ints/strs all work).
* ``.env`` lives at the project root next to ``.env.example`` (gitignored).
* ``data_dir`` and ``artifacts_dir`` are anchored to the project root so that
  runs work identically no matter which directory they are launched from.

Usage
-----
    from config.default import Config
    cfg = Config()                 # reads defaults + .env + os.environ

The experiments/faces never build their own magic numbers; they read ``cfg``."""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path

#: Project root — always the folder two levels above this file (src/config/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: Prefix for environment-variable overrides (see module docstring).
ENV_PREFIX = "LAYER_MERGING_"
#: Optional dotenv file (created from ``.env.example`` by ``make env``).
ENV_FILE = PROJECT_ROOT / ".env"


@dataclass
class Config:
    # --- Model -----------------------------------------------------------
    hidden_dims: tuple = (256, 256, 256, 256)  # symmetric -> layers are mergeable
    num_classes: int = 10
    input_dim: int = 28 * 28                   # MNIST image size (28, 28)
    dropout: float = 0.0                       # 0 = deterministic (merging math)
    model_arch: str = "residual"               # 'residual' (merging) | 'plain' (pruning)
    branch_scale: float = 0.1                  # residual-branch init gain (paper pitfall)

    # --- Data --------------------------------------------------------------
    data_dir: str = "data"                     # resolved under the project root
    mnist_mirror: str = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    batch_size: int = 128
    val_size: int = 6000                       # out of 60k train images

    # --- Training -----------------------------------------------------------
    seed: int = 42                      # canonical reproduction seed
    seeds: tuple = (42, 7, 2026)        # multi-seed aggregation for the report
    epochs: int = 6
    lr: float = 1e-3
    optimizer: str = "adam"                    # 'adam' | 'adamw' | 'sgd'
    loss: str = "cross_entropy"                # 'cross_entropy' | 'mse'

    # --- Similarity / manifold (MKA paper) -----------------------------------
    n_manifold_dims: int = 8                   # diffusion-map dimension k
    diffusion_time: int = 1                    # t in Phi_t
    kernel_sigma: float = 3.0                  # Gaussian bandwidth sigma
    n_calibration_samples: int = 512           # reference samples for similarity

    # --- Merging --------------------------------------------------------------
    merge_min_alpha: float = 0.1
    merge_max_alpha: float = 0.9
    merge_threshold: float = 0.05              # pairs below this similarity are skipped
    merge_strategies: tuple = ("weighted", "uniform", "slerp")   # phase-4 comparison
    alpha_sweep: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)  # phase-4b ablation: α on the grid

    # --- Pruning (SparC3 paper) -------------------------------------------------
    lrp_eps: float = 1e-2                      # epsilon-stabilized LRP denominator
    prune_keep_ratio: float = 0.5              # phase-6 pipeline operating point
    prune_sweep: tuple = (1.0, 0.9, 0.75, 0.5, 0.3, 0.1)  # phase-5 keep-ratio sweep
    prune_methods: tuple = ("lrp", "magnitude", "random")  # names, order in phase-5
    random_seed: int = 0                       # RNG for the random-pruning baseline

    # --- Project paths -----------------------------------------------------------
    artifacts_dir: str = "artifacts"           # resolved under the project root

    # --- Artifact filenames (one place; notebooks reference a few of these) ------
    files: dict = field(default_factory=lambda: {
        "baseline_plain": "baseline_model.pt",
        "baseline_residual": "residual_baseline.pt",
        "merged_model": "merged_model.pt",
        "baseline_strategies": "baseline_for_strategies.pt",
        "pipeline_merged": "pipeline_merged.pt",
        "pipeline_pruned": "pipeline_pruned.pt",
        "baseline_plain_seed": "baseline_plain_seed{seed}.pt",
        "baseline_residual_seed": "baseline_residual_seed{seed}.pt",
        "similarity_residual_nmi": "similarity_residual_nmi",
        "similarity_residual_cosine": "similarity_residual_cosine",
        "similarity_plain_nmi": "similarity_plain_nmi",
        "similarity_plain_cosine": "similarity_plain_cosine",
        "similarity_grid": "similarity_all.png",
        "phase1_results": "phase1_baseline.json",
        "phase2_json": "phase2_similarity.json",
        "phase3_curve": "phase3_merge_curve.json",
        "merge_curve": "merge_curve.png",
        "phase4_results": "phase4_strategies.json",
        "strategies_fig": "strategies.png",
        "alpha_sweep_results": "phase4b_alpha_sweep.json",
        "alpha_sweep_fig": "alpha_sweep.png",
        "hybrid_results": "phase6b_hybrid.json",
        "hybrid_fig": "hybrid_tracks.png",
        "phase5_results": "phase5_pruning.json",
        "pruning_curves": "pruning_curves.png",
        "final_csv": "final_results.csv",
        "final_json": "final_results.json",
        "results_table": "results_table.png",
        "lrp_input_relevance": "lrp_input_relevance.png",
        "reproduce_csv": "reproduce_summary.csv",
        "reproduce_json": "reproduce_summary.json",
        "reproduce_table": "reproduce_table.png",
        "gantt": "gantt.png",
    })

    # --- Gantt chart (Project-Management section of the report) ---------------
    gantt_tasks: tuple = ("Read papers", "Baseline + data", "Similarity",
                          "Merging", "Strategies", "Pruning", "Report + viva")
    gantt_starts: tuple = (0, 1, 2, 3, 4, 5, 6)
    gantt_durations: tuple = (1, 1, 1, 1, 1, 1, 2)

    # ------------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Apply ``.env``/environment overrides, then anchor the two paths."""
        self._apply_env()
        self._resolve_paths()

    def _apply_env(self) -> None:
        """Overlay values from ``.env`` (if python-dotenv is installed) and
        from real environment variables with the ``LAYER_MERGING_`` prefix.

        An explicitly-passed constructor argument always wins over the env:
        we only consult the env when the field still holds its class default.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv(ENV_FILE)  # does not overwrite already-set env vars
        except ImportError:
            pass  # .env support is optional; os.environ still works
        for name in self.__dataclass_fields__:
            key = ENV_PREFIX + name.upper()
            raw = os.environ.get(key)
            if raw is None:
                continue
            f = next(fd for fd in fields(self) if fd.name == name)
            default = f.default_factory() if f.default is MISSING else f.default
            if getattr(self, name) != default:
                continue  # explicitly set by the caller -> keep their value
            setattr(self, name, self._coerce(key, raw, default))

    @staticmethod
    def _coerce(key: str, raw: str, default) -> object:
        """Parse an env value into the right python type.

        Prefers ``ast.literal_eval`` (handles tuples / ints / floats). A bare
        unquoted word (e.g. ``residual``, ``data``) is NOT a valid literal, so
        we fall back to the field's default type: strings stay strings,
        numbers get cast. Anything else raises a clear error.
        """
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            if isinstance(default, str):
                return raw
            if isinstance(default, (int, float)):
                try:
                    return type(default)(raw)
                except ValueError as exc:
                    raise ValueError(f"{key}={raw!r} is not a number") from exc
            raise ValueError(
                f"{key}={raw!r} is not valid config (use python literals, "
                f'e.g. LAYER_MERGING_HIDDEN_DIMS="(256, 256)").'
            )

    def _resolve_paths(self) -> None:
        """Absolute paths under PROJECT_ROOT so CWD never matters."""
        self.data_dir = self._absolute(self.data_dir)
        self.artifacts_dir = self._absolute(self.artifacts_dir)

    @staticmethod
    def _absolute(path: str) -> str:
        p = Path(path)
        return str(p if p.is_absolute() else PROJECT_ROOT / p)

    # --- tiny helpers used all over the experiments ---------------------------
    def ensure_artifacts(self) -> str:
        """Create the artifacts directory and return its absolute path."""
        os.makedirs(self.artifacts_dir, exist_ok=True)
        return self.artifacts_dir

    def artifact(self, key: str) -> str:
        """Absolute path of a named artifact (see ``files``)."""
        return os.path.join(self.artifacts_dir, self.files[key])


if __name__ == "__main__":
    cfg = Config()
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"data_dir     = {cfg.data_dir}")
    print(f"artifacts_dir= {cfg.artifacts_dir}")
    print(f"model_arch   = {cfg.model_arch}")
    print(f"hidden_dims  = {cfg.hidden_dims}")