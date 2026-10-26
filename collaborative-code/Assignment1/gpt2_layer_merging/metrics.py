"""Evaluation metrics."""

import math
import time

import torch

from .config import DEVICE, EVAL_BLOCK_SIZE, EVAL_TOKENS, LATENCY_REPEATS


@torch.no_grad()
def calculate_ppl(model, text, tokenizer, block_size=EVAL_BLOCK_SIZE,
                  max_tokens=EVAL_TOKENS):
    """
    Approximate perplexity using the same non-overlapping chunks for every model.
    """
    model.eval()

    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    if max_tokens is not None:
        ids = ids[:max_tokens]

    n_blocks = len(ids) // block_size

    if n_blocks == 0:
        raise ValueError("Not enough tokens for one evaluation block.")

    total_nll = 0.0
    total_pred_tokens = 0

    for i in range(n_blocks):
        x = ids[
            i * block_size:(i + 1) * block_size
        ].unsqueeze(0).to(DEVICE)

        outputs = model(
            input_ids=x,
            labels=x,
            use_cache=False,
        )

        count = x.numel() - 1
        total_nll += outputs.loss.item() * count
        total_pred_tokens += count

    return math.exp(total_nll / total_pred_tokens)


def parameter_count(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_latency(model, text, tokenizer, seq_len=256,
                    repeats=LATENCY_REPEATS):
    """Measure average forward-pass latency in milliseconds."""
    model.eval()

    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][:, :seq_len].to(DEVICE)

    for _ in range(5):
        _ = model(ids, use_cache=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(repeats):
        _ = model(ids, use_cache=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    return 1000.0 * (end - start) / repeats


def evaluate_model(model, text, tokenizer):
    """Calculate PPL, parameter count, and latency for a model."""
    model = model.to(DEVICE).eval()

    return {
        "PPL": calculate_ppl(model, text, tokenizer),
        "Parameters": parameter_count(model),
        "Latency_ms": measure_latency(model, text, tokenizer),
    }
