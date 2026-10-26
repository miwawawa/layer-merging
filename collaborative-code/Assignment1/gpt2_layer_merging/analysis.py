"""Result analysis and final test selection."""

import numpy as np
import pandas as pd

from .affine import (
    build_affine_merged_model,
    make_affine_conditions,
    make_random_weight_affine_condition,
)
from .config import DEVICE, RANDOM_RUNS, SEED
from .metrics import calculate_ppl, measure_latency, parameter_count
from .utils import free_model


def best_method_per_pair(affine_summary_df):
    """Select the lowest-PPL affine method for each pair."""
    return (
        affine_summary_df
        .sort_values("PPL_mean")
        .groupby("Pair_Start", as_index=False)
        .first()
        .sort_values("Pair_Start")
    )


def global_best_configuration(affine_summary_df):
    """Select the globally best validation configuration."""
    return affine_summary_df.sort_values("PPL_mean").iloc[0]


def compare_multiply_vs_average(affine_summary_df, pair_starts):
    """Compare Multiply and Average on every layer pair."""
    rows = []

    for L in pair_starts:
        pair_name = f"{L}-{L+1}"
        group = affine_summary_df[affine_summary_df["Pair"] == pair_name]

        avg_row = group[group["Method"] == "Average"].iloc[0]
        mul_row = group[group["Method"] == "Multiply"].iloc[0]

        rows.append({
            "Pair": pair_name,
            "Average_PPL": avg_row["PPL_mean"],
            "Multiply_PPL": mul_row["PPL_mean"],
            "Multiply_minus_Average_PPL": (
                mul_row["PPL_mean"] - avg_row["PPL_mean"]
            ),
            "Multiply_Better": (
                mul_row["PPL_mean"] < avg_row["PPL_mean"]
            ),
            "Average_MSE": avg_row["Hidden_MSE_mean"],
            "Multiply_MSE": mul_row["Hidden_MSE_mean"],
        })

    return pd.DataFrame(rows)


def build_selected_affine_model(L, method, all_fitted_maps, random_seed=None):
    """Rebuild the selected affine model from stored fitted maps."""
    fitted = all_fitted_maps[L]

    A1, b1 = fitted["A1"], fitted["b1"]
    A2, b2 = fitted["A2"], fitted["b2"]

    if method == "Random-Weight Average":
        if random_seed is None:
            raise ValueError(
                "random_seed is required for Random-Weight Average"
            )

        A, b = make_random_weight_affine_condition(
            A2, b2, seed=random_seed
        )
    else:
        A, b = make_affine_conditions(A1, b1, A2, b2)[method]

    return build_affine_merged_model(L, A, b)


def select_random_seed(affine_raw_df, pair_start):
    """Choose the validation-best random run for a selected pair."""
    candidates = affine_raw_df[
        (affine_raw_df["Pair_Start"] == pair_start)
        & (affine_raw_df["Method"] == "Random-Weight Average")
    ]

    best_run = candidates.sort_values("PPL").iloc[0]
    return int(best_run["Random_Seed"])


def evaluate_final_test(best_pair_start, best_method, all_fitted_maps,
                         affine_raw_df, original_model, test_text, tokenizer,
                         baseline_params):
    """Evaluate the validation-selected configuration on the test set."""
    original_test_ppl = calculate_ppl(
        original_model,
        test_text,
        tokenizer,
    )

    selected_random_seed = None

    if best_method == "Random-Weight Average":
        selected_random_seed = select_random_seed(
            affine_raw_df,
            best_pair_start,
        )

    best_test_model = build_selected_affine_model(
        best_pair_start,
        best_method,
        all_fitted_maps,
        random_seed=selected_random_seed,
    ).to(DEVICE)

    best_test_ppl = calculate_ppl(
        best_test_model,
        test_text,
        tokenizer,
    )

    best_test_latency = measure_latency(
        best_test_model,
        test_text,
        tokenizer,
    )

    best_test_params = parameter_count(best_test_model)

    result = {
        "Original_Test_PPL": original_test_ppl,
        "Compressed_Pair": f"{best_pair_start}-{best_pair_start + 1}",
        "Method": best_method,
        "Random_Seed": selected_random_seed,
        "Compressed_Test_PPL": best_test_ppl,
        "Parameters": best_test_params,
        "Parameter_Reduction_%": (
            100.0 * (1.0 - best_test_params / baseline_params)
        ),
        "Latency_ms": best_test_latency,
    }

    free_model(best_test_model)
    return result


def build_ppl_pivot(affine_summary_df, pair_starts):
    """Build PPL pivot table in natural layer-pair order."""
    pivot = affine_summary_df.pivot(
        index="Pair",
        columns="Method",
        values="PPL_mean",
    )

    return pivot.reindex([f"{L}-{L+1}" for L in pair_starts])


def build_mse_pivot(affine_summary_df, pair_starts):
    """Build hidden-MSE pivot table in natural layer-pair order."""
    pivot = affine_summary_df.pivot(
        index="Pair",
        columns="Method",
        values="Hidden_MSE_mean",
    )

    return pivot.reindex([f"{L}-{L+1}" for L in pair_starts])


def run_full_test_table(pair_starts, all_fitted_maps, random_runs,
                        test_text, tokenizer):
    """Optionally evaluate every pair/method on the test set."""
    rows = []

    for L in pair_starts:
        fitted = all_fitted_maps[L]
        A1, b1 = fitted["A1"], fitted["b1"]
        A2, b2 = fitted["A2"], fitted["b2"]

        fixed_conditions = make_affine_conditions(A1, b1, A2, b2)

        for method in ["Remove", "Average", "Multiply"]:
            A, b = fixed_conditions[method]
            model = build_affine_merged_model(L, A, b).to(DEVICE)

            rows.append({
                "Pair": f"{L}-{L+1}",
                "Method": method,
                "Alpha": 0.5 if method == "Average" else np.nan,
                "Random_Seed": np.nan,
                "Test_PPL": calculate_ppl(model, test_text, tokenizer),
                "Parameters": parameter_count(model),
                "Latency_ms": measure_latency(
                    model, test_text, tokenizer
                ),
            })
            free_model(model)

        for run in range(random_runs):
            random_seed = SEED + 1000 * L + run
            A, b = make_random_weight_affine_condition(
                A2, b2, seed=random_seed
            )

            model = build_affine_merged_model(L, A, b).to(DEVICE)

            rows.append({
                "Pair": f"{L}-{L+1}",
                "Method": "Random-Weight Average",
                "Run": run,
                "Alpha": np.nan,
                "Random_Seed": random_seed,
                "Test_PPL": calculate_ppl(model, test_text, tokenizer),
                "Parameters": parameter_count(model),
                "Latency_ms": measure_latency(
                    model, test_text, tokenizer
                ),
            })
            free_model(model)

    return pd.DataFrame(rows)
