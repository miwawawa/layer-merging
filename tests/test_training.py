"""Tests for the shared training loop (optimizer / loss configurability)."""

import pytest

from config.default import Config
from training import train_model
from evaluation.metrics import evaluate


def test_train_model_returns_history(trained_toy_model, loaders):
    cfg = Config(epochs=2, lr=1e-2)
    history = train_model(trained_toy_model, loaders, cfg)
    assert set(history) == {"train_loss", "val_acc"}
    assert len(history["train_loss"]) == 2
    assert all(0.0 <= v <= 1.0 for v in history["val_acc"])


def test_train_model_epochs_override(trained_toy_model, loaders):
    cfg = Config(epochs=5)
    history = train_model(trained_toy_model, loaders, cfg, epochs=2)
    assert len(history["train_loss"]) == 2       # explicit arg wins over cfg


def test_train_improves_accuracy(toy_model, loaders, cfg):
    from models.mlp import SimpleMLP
    before = evaluate(toy_model, loaders["test"])
    train_model(toy_model, loaders, Config(epochs=8, lr=5e-2), epochs=8)
    after = evaluate(toy_model, loaders["test"])
    assert after >= before                       # deterministic train on toy data


def test_unknown_optimizer_raises(toy_model, loaders):
    with pytest.raises(ValueError):
        train_model(toy_model, loaders, Config(), optimizer="not_an_optimizer")


def test_unknown_loss_raises(toy_model, loaders):
    with pytest.raises(ValueError):
        train_model(toy_model, loaders, Config(), loss="not_a_loss")


def test_sgd_and_mse_constructible(toy_model, loaders):
    cfg = Config(epochs=1, lr=1e-2, optimizer="sgd", loss="cross_entropy")
    h = train_model(toy_model, loaders, cfg)
    assert len(h["train_loss"]) == 1