"""Training: the one training loop every phase reuses (DRY).

The optimizer and loss function come from ``Config`` (``cfg.optimizer``,
``cfg.loss``) so no algorithm is hardcoded here. Anything else (batch count,
device) is inferred or also read from ``cfg``.
"""

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from config.default import Config
from models.base import BaseNN

#: Name -> constructor map for the optimizers we support (extend as needed).
_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

#: Name -> constructor map for the loss functions we support.
_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
}


def train_model(model: BaseNN, loaders: Dict[str, DataLoader],
                cfg: Config, epochs: int | None = None, lr: float | None = None,
                optimizer: str | None = None, loss: str | None = None) -> Dict:
    """Train ``model`` with the configured optimizer/loss; returns history.

    Simple override rule: an explicit argument wins over ``cfg``, which wins
    over the module-level ``_OPTIMIZERS``/``_LOSSES`` names.
    """
    epochs = epochs or cfg.epochs
    lr = lr if lr is not None else cfg.lr
    optimizer = optimizer or cfg.optimizer
    loss = loss or cfg.loss

    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {optimizer!r}; choose {list(_OPTIMIZERS)}")
    if loss not in _LOSSES:
        raise ValueError(f"Unknown loss {loss!r}; choose {list(_LOSSES)}")

    optim = _OPTIMIZERS[optimizer](model.parameters(), lr=lr)
    loss_fn = _LOSSES[loss]()
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in loaders["train"]:
            if x.dim() == 4:
                x = x.view(x.size(0), -1)
            optim.zero_grad()
            logits = model(x)
            if loss == "mse":  # one-hot targets so MSE compares equal-shaped tensors
                y_ohe = torch.nn.functional.one_hot(y, num_classes=logits.size(-1)).float()
                loss_val = loss_fn(logits, y_ohe)
            else:
                loss_val = loss_fn(logits, y)
            loss_val.backward()
            optim.step()
            total_loss += loss_val.item()
            n_batches += 1
        from evaluation.metrics import evaluate
        val_acc = evaluate(model, loaders["val"])
        history["train_loss"].append(total_loss / max(n_batches, 1))
        history["val_acc"].append(val_acc)
        print(f"  epoch {epoch + 1}/{epochs}  loss={history['train_loss'][-1]:.4f}  val_acc={val_acc:.4f}")
    return history