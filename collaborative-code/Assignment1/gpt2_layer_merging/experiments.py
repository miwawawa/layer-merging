"""Experiment runners for affine layer merging."""

import numpy as np
import pandas as pd

from .affine import (
    build_affine_merged_model,
    fit_affine,
    hidden_mse,
    make_affine_conditions,
    make_random_weight_affine_condition,
)
from .config import DEVICE, EVAL_TOKENS, FIT_TOKENS, RANDOM_RUNS, SEED
from .metrics import calculate_ppl, evaluate_model, measure_latency, parameter_count
from .state_collection import collect_pair_states
from .utils import free_model


def run_affine_pair(L, original_model, train_text, validation_text, tokenizer):
    """Fit and evaluate all four affine conditions for one pair."""
    print(f"\n[Affine] pair {L}-{L+1}")

    X_train, Y1_train, Y2_train = collect_pair_states(
        original_model,
        train_text,
        tokenizer,
        L,
        total_tokens=FIT_TOKENS,
    )

    A1, b1 = fit_affine(X_train, Y1_train)
    A2, b2 = fit_affine(Y1_train, Y2_train)

    X_val, Y1_val, Y2_val = collect_pair_states(
        original_model,
        validation_text,
        tokenizer,
        L,
        total_tokens=min(FIT_TOKENS, EVAL_TOKENS),
    )

    rows = []

    fixed_conditions = make_affine_conditions(A1, b1, A2, b2)

    for method in ["Remove", "Average", "Multiply"]:
        A, b = fixed_conditions[method]

        local_mse = hidden_mse(X_val, Y2_val, A, b)

        model = build_affine_merged_model(L, A, b).to(DEVICE)

        rows.append({
            "Family": "Affine fair comparison",
            "Pair_Start": L,
            "Pair": f"{L}-{L+1}",
            "Method": method,
            "Run": np.nan,
            "Alpha": 0.5 if method == "Average" else np.nan,
            "Random_Seed": np.nan,
            "Hidden_MSE": local_mse,
            "PPL": calculate_ppl(model, validation_text, tokenizer),
            "Parameters": parameter_count(model),
            "Latency_ms": measure_latency(model, validation_text, tokenizer),
        })

        free_model(model)

    for run in range(RANDOM_RUNS):
        random_seed = SEED + 1000 * L + run

        A, b = make_random_weight_affine_condition(
            A2,
            b2,
            seed=random_seed,
        )

        local_mse = hidden_mse(X_val, Y2_val, A, b)

        model = build_affine_merged_model(L, A, b).to(DEVICE)

        rows.append({
            "Family": "Affine fair comparison",
            "Pair_Start": L,
            "Pair": f"{L}-{L+1}",
            "Method": "Random-Weight Average",
            "Run": run,
            "Alpha": np.nan,
            "Random_Seed": random_seed,
            "Hidden_MSE": local_mse,
            "PPL": calculate_ppl(model, validation_text, tokenizer),
            "Parameters": parameter_count(model),
            "Latency_ms": measure_latency(model, validation_text, tokenizer),
        })

        free_model(model)

    fitted = {
        "A1": A1,
        "b1": b1,
        "A2": A2,
        "b2": b2,
    }

    return pd.DataFrame(rows), fitted


def run_all_affine_pairs(pair_starts, original_model, train_text,
                         validation_text, tokenizer):
    """Run the affine experiment over all adjacent layer pairs."""
    all_results = []
    all_fitted_maps = {}

    for idx, L in enumerate(pair_starts, start=1):
        print("\n" + "=" * 70)
        print(f"AFFINE PAIR {idx}/{len(pair_starts)}: {L}-{L+1}")
        print("=" * 70)

        df_pair, fitted = run_affine_pair(
            L,
            original_model,
            train_text,
            validation_text,
            tokenizer,
        )

        all_results.append(df_pair)
        all_fitted_maps[L] = fitted

    return pd.concat(all_results, ignore_index=True), all_fitted_maps


def summarize_affine_results(affine_raw_df, baseline_params, baseline_ppl):
    """Aggregate affine results by pair and method."""
    summary_rows = []

    for (pair_start, pair, method), group in affine_raw_df.groupby(
        ["Pair_Start", "Pair", "Method"]
    ):
        summary_rows.append({
            "Pair_Start": pair_start,
            "Pair": pair,
            "Method": method,
            "PPL_mean": group["PPL"].mean(),
            "PPL_std": group["PPL"].std(ddof=1) if len(group) > 1 else 0.0,
            "Hidden_MSE_mean": group["Hidden_MSE"].mean(),
            "Hidden_MSE_std": (
                group["Hidden_MSE"].std(ddof=1)
                if len(group) > 1 else 0.0
            ),
            "Alpha_mean": group["Alpha"].mean(),
            "Parameters": int(group["Parameters"].iloc[0]),
            "Latency_ms_mean": group["Latency_ms"].mean(),
        })

    summary = pd.DataFrame(summary_rows)

    summary["Parameter_Reduction_%"] = (
        100.0 * (1.0 - summary["Parameters"] / baseline_params)
    )

    summary["PPL_Change_vs_Original_%"] = (
        100.0 * (summary["PPL_mean"] / baseline_ppl - 1.0)
    )

    return summary


def summarize_direct_results(direct_raw_df, baseline_params):
    """Aggregate direct GPT-2 results by pair and method."""
    rows = []

    for (pair_start, pair, method), group in direct_raw_df.groupby(
        ["Pair_Start", "Pair", "Method"]
    ):
        rows.append({
            "Pair_Start": pair_start,
            "Pair": pair,
            "Method": method,
            "PPL_mean": group["PPL"].mean(),
            "PPL_std": group["PPL"].std(ddof=1) if len(group) > 1 else 0.0,
            "Alpha_mean": group["Alpha"].mean(),
            "Parameters": int(group["Parameters"].iloc[0]),
            "Latency_ms_mean": group["Latency_ms"].mean(),
        })

    summary = pd.DataFrame(rows)
    summary["Parameter_Reduction_%"] = (
        100.0 * (1.0 - summary["Parameters"] / baseline_params)
    )

    return summary
