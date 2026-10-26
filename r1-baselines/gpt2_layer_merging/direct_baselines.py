"""Direct GPT-2 block-merging baselines."""

import numpy as np
import pandas as pd

from .config import RANDOM_RUNS, SEED
from .metrics import evaluate_model
from .model import (
    average_real_blocks,
    random_weight_average_real_block,
    remove_real_block,
)
from .utils import free_model


def run_direct_pair(L, validation_text, tokenizer):
    """Run the three direct GPT-2 conditions for one adjacent pair."""
    rows = []

    print(f"\n[Direct] pair {L}-{L+1}")

    model = remove_real_block(L)
    result = evaluate_model(model, validation_text, tokenizer)
    rows.append({
        "Family": "Direct GPT-2",
        "Pair_Start": L,
        "Pair": f"{L}-{L+1}",
        "Method": "Remove",
        "Run": np.nan,
        "Alpha": np.nan,
        "Random_Seed": np.nan,
        **result,
    })
    free_model(model)

    model = average_real_blocks(L, alpha=0.5)
    result = evaluate_model(model, validation_text, tokenizer)
    rows.append({
        "Family": "Direct GPT-2",
        "Pair_Start": L,
        "Pair": f"{L}-{L+1}",
        "Method": "Average",
        "Run": np.nan,
        "Alpha": 0.5,
        "Random_Seed": np.nan,
        **result,
    })
    free_model(model)

    for run in range(RANDOM_RUNS):
        random_seed = SEED + 1000 * L + run

        model = random_weight_average_real_block(
            L,
            seed=random_seed,
        )
        result = evaluate_model(model, validation_text, tokenizer)

        rows.append({
            "Family": "Direct GPT-2",
            "Pair_Start": L,
            "Pair": f"{L}-{L+1}",
            "Method": "Random-Weight Average",
            "Run": run,
            "Alpha": np.nan,
            "Random_Seed": random_seed,
            **result,
        })
        free_model(model)

    return pd.DataFrame(rows)
