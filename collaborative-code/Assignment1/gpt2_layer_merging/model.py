"""GPT-2 model/tokenizer loading and block manipulation."""

import copy

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_ID


def load_tokenizer():
    """Load GPT-2 tokenizer and disable the tokenizer length warning."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.model_max_length = int(1e12)
    return tokenizer


def load_fresh_model():
    """Load a fresh GPT-2 model in evaluation mode."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.config.use_cache = False
    model.eval()
    return model


def remove_real_block(L):
    """Remove Transformer block L from a fresh GPT-2 model."""
    model = load_fresh_model()

    blocks = list(model.transformer.h)
    new_blocks = blocks[:L] + blocks[L + 1:]

    model.transformer.h = nn.ModuleList(new_blocks)
    model.config.n_layer = len(new_blocks)

    return model


def average_real_blocks(L, alpha=0.5):
    """Average corresponding parameters of GPT-2 blocks L and L+1."""
    model = load_fresh_model()

    block1 = model.transformer.h[L]
    block2 = model.transformer.h[L + 1]
    merged = copy.deepcopy(block2)

    p1 = dict(block1.named_parameters())
    p2 = dict(block2.named_parameters())

    with torch.no_grad():
        for name, p in merged.named_parameters():
            if name not in p1 or name not in p2:
                raise KeyError(f"{name} is missing")
            if p1[name].shape != p2[name].shape:
                raise ValueError(f"Shape mismatch for {name}")

            p.copy_(alpha * p1[name] + (1.0 - alpha) * p2[name])

    blocks = list(model.transformer.h)
    new_blocks = blocks[:L] + [merged] + blocks[L + 2:]

    model.transformer.h = nn.ModuleList(new_blocks)
    model.config.n_layer = len(new_blocks)

    return model


def random_weight_average_real_block(L, seed):
    """
    Shuffle every parameter tensor of block L+1, average it 50/50 with the
    original, then remove block L.
    """
    model = load_fresh_model()

    blocks = list(model.transformer.h)
    block2 = blocks[L + 1]
    merged = copy.deepcopy(block2)
    p2 = dict(block2.named_parameters())

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    from .utils import shuffle_same_distribution

    with torch.no_grad():
        for name, p in merged.named_parameters():
            if name not in p2:
                raise KeyError(f"{name} is missing")

            original_parameter = p2[name]
            random_parameter = shuffle_same_distribution(
                original_parameter,
                generator,
            )

            p.copy_(0.5 * original_parameter + 0.5 * random_parameter)

    new_blocks = blocks[:L] + [merged] + blocks[L + 2:]

    model.transformer.h = nn.ModuleList(new_blocks)
    model.config.n_layer = len(new_blocks)

    return model
