"""Tests for the single source of configuration (env-driven, no hardcoding)."""

import os

import pytest

from config.default import PROJECT_ROOT, Config, ENV_PREFIX


def test_defaults(cfg: Config):
    assert cfg.hidden_dims == (256, 256, 256, 256)
    assert cfg.num_classes == 10
    assert cfg.input_dim == 28 * 28
    assert cfg.model_arch == "residual"
    assert cfg.branch_scale == 0.1
    assert cfg.epochs == 6
    assert cfg.merge_threshold == 0.05
    assert cfg.prune_keep_ratio == 0.5
    assert cfg.mnist_mirror.startswith("https://")


def test_paths_are_absolute_and_anchored(cfg: Config):
    # Runs must not depend on the current working directory.
    assert os.path.isabs(cfg.data_dir)
    assert os.path.isabs(cfg.artifacts_dir)
    assert cfg.data_dir == str(PROJECT_ROOT / "data")
    assert cfg.artifacts_dir == str(PROJECT_ROOT / "artifacts")


def test_env_override_integer(cfg: Config, monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "EPOCHS", "7")
    assert Config().epochs == 7


def test_env_override_tuple(cfg: Config, monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "PRUNE_SWEEP", "(1.0, 0.9, 0.5)")
    assert Config().prune_sweep == (1.0, 0.9, 0.5)


def test_env_override_string_word(cfg: Config, monkeypatch):
    # "bad literal" (bare word) must still parse as the string it is.
    monkeypatch.setenv(ENV_PREFIX + "MODEL_ARCH", "plain")
    assert Config().model_arch == "plain"


def test_env_override_float(cfg: Config, monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "LR", "0.0007")
    assert Config().lr == 0.0007


def test_invalid_env_raises(cfg: Config, monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "EPOCHS", "not-a-number")
    with pytest.raises(ValueError):
        Config()


def test_files_map_has_expected_keys(cfg: Config):
    for key in ("baseline_plain", "baseline_residual", "similarity_residual_nmi",
                "phase5_results", "final_csv", "gantt"):
        assert cfg.files[key]
    assert cfg.files["baseline_residual"].endswith(".pt")


def test_artifact_joins_artifacts_dir(cfg: Config):
    assert cfg.artifact("gantt") == os.path.join(cfg.artifacts_dir, cfg.files["gantt"])


def test_ensure_artifacts_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "ARTIFACTS_DIR", str(tmp_path / "out"))
    cfg = Config()
    assert os.path.isdir(cfg.ensure_artifacts())


def test_gantt_data_vector_sizes_agree(cfg: Config):
    assert len(cfg.gantt_tasks) == len(cfg.gantt_starts) == len(cfg.gantt_durations)