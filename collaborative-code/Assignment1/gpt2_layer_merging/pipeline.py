"""Top-level experiment pipeline."""

from pathlib import Path

import pandas as pd

from .analysis import (
    best_method_per_pair,
    build_mse_pivot,
    build_ppl_pivot,
    compare_multiply_vs_average,
    evaluate_final_test,
    global_best_configuration,
    run_full_test_table,
)
from .config import (
    DEVICE,
    MODE,
    PAIR_STARTS,
    RESULTS_DIR,
    RANDOM_RUNS,
    RUN_DIRECT_SWEEP,
    TEST_ALL_PAIRS,
)
from .data import load_wikitext
from .direct_baselines import run_direct_pair
from .experiments import (
    run_all_affine_pairs,
    summarize_affine_results,
    summarize_direct_results,
)
from .metrics import calculate_ppl, measure_latency, parameter_count
from .model import load_fresh_model, load_tokenizer
from .plots import plot_hidden_mse, plot_latency, plot_ppl
from .utils import set_seed


def save_results(affine_raw_df, affine_summary_df, best_per_pair,
                 mul_vs_avg_df, direct_raw_df, direct_summary_df,
                 test_all_df):
    """Save experiment tables to CSV."""
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "affine_all_pairs_raw.csv": affine_raw_df,
        "affine_all_pairs_summary.csv": affine_summary_df,
        "best_method_per_pair.csv": best_per_pair,
        "multiply_vs_average.csv": mul_vs_avg_df,
    }

    if RUN_DIRECT_SWEEP:
        files["direct_all_pairs_raw.csv"] = direct_raw_df
        files["direct_all_pairs_summary.csv"] = direct_summary_df

    if TEST_ALL_PAIRS:
        files["test_all_pairs.csv"] = test_all_df

    for filename, dataframe in files.items():
        dataframe.to_csv(output_dir / filename, index=False)

    print("\n===== SAVED FILES =====")
    for filename in files:
        print(output_dir / filename)


def run_pipeline():
    """Run the complete experiment."""
    set_seed()

    print("device:", DEVICE)
    print("mode:", MODE)
    print("pairs:", [(L, L + 1) for L in PAIR_STARTS])
    print("direct sweep:", RUN_DIRECT_SWEEP)

    # ------------------------------------------------------------------
    # Data and model setup
    # ------------------------------------------------------------------
    train_text, validation_text, test_text = load_wikitext()
    tokenizer = load_tokenizer()
    original_model = load_fresh_model().to(DEVICE)

    print("Number of Transformer blocks:",
          len(original_model.transformer.h))
    print("Hidden size:", original_model.config.n_embd)

    # ------------------------------------------------------------------
    # Original GPT-2 baseline
    # ------------------------------------------------------------------
    baseline_ppl = calculate_ppl(
        original_model,
        validation_text,
        tokenizer,
    )
    baseline_params = parameter_count(original_model)
    baseline_latency = measure_latency(
        original_model,
        validation_text,
        tokenizer,
    )

    print("\n===== ORIGINAL GPT-2 =====")
    print("Validation PPL :", baseline_ppl)
    print("Parameters     :", f"{baseline_params:,}")
    print("Latency (ms)   :", baseline_latency)

    # ------------------------------------------------------------------
    # Linear sanity check
    # ------------------------------------------------------------------
    import torch

    torch.manual_seed(42)
    d, n = 16, 128
    X = torch.randn(n, d)
    W1 = torch.randn(d, d)
    W2 = torch.randn(d, d)

    sequential = (X @ W1) @ W2
    composed = X @ (W1 @ W2)

    sanity = {
        "max_abs_error":
            (sequential - composed).abs().max().item(),
        "mse":
            torch.mean((sequential - composed) ** 2).item(),
    }

    print("\n===== LINEAR SANITY CHECK =====")
    print(sanity)

    # ------------------------------------------------------------------
    # Affine experiment over all pairs
    # ------------------------------------------------------------------
    affine_raw_df, all_fitted_maps = run_all_affine_pairs(
        PAIR_STARTS,
        original_model,
        train_text,
        validation_text,
        tokenizer,
    )

    print("\n===== ALL AFFINE RAW RESULTS =====")
    display(affine_raw_df)

    affine_summary_df = summarize_affine_results(
        affine_raw_df,
        baseline_params,
        baseline_ppl,
    )

    print("\n===== ALL-PAIR AFFINE SUMMARY =====")
    display(
        affine_summary_df.sort_values(
            ["Pair_Start", "PPL_mean"]
        )
    )

    best_per_pair = best_method_per_pair(affine_summary_df)

    print("\n===== BEST METHOD FOR EACH PAIR =====")
    display(
        best_per_pair[
            [
                "Pair",
                "Method",
                "PPL_mean",
                "Hidden_MSE_mean",
                "Latency_ms_mean",
                "Parameter_Reduction_%",
            ]
        ]
    )

    global_best = global_best_configuration(affine_summary_df)

    print("\n===== BEST VALIDATION CONFIGURATION =====")
    print(global_best)

    # ------------------------------------------------------------------
    # Direct GPT-2 sweep
    # ------------------------------------------------------------------
    if RUN_DIRECT_SWEEP:
        direct_results = []

        for idx, L in enumerate(PAIR_STARTS, start=1):
            print("\n" + "=" * 70)
            print(f"DIRECT PAIR {idx}/{len(PAIR_STARTS)}: {L}-{L+1}")
            print("=" * 70)

            direct_results.append(
                run_direct_pair(L, validation_text, tokenizer)
            )

        direct_raw_df = pd.concat(
            direct_results,
            ignore_index=True,
        )

        direct_summary_df = summarize_direct_results(
            direct_raw_df,
            baseline_params,
        )

        print("\n===== DIRECT GPT-2 ALL-PAIR SUMMARY =====")
        display(
            direct_summary_df.sort_values(
                ["Pair_Start", "PPL_mean"]
            )
        )
    else:
        direct_raw_df = pd.DataFrame()
        direct_summary_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Tables and plots
    # ------------------------------------------------------------------
    ppl_pivot = build_ppl_pivot(affine_summary_df, PAIR_STARTS)
    print("\n===== PPL PIVOT TABLE =====")
    display(ppl_pivot)

    mse_pivot = build_mse_pivot(affine_summary_df, PAIR_STARTS)
    print("\n===== HIDDEN MSE PIVOT TABLE =====")
    display(mse_pivot)

    plot_ppl(affine_summary_df, PAIR_STARTS, baseline_ppl)
    plot_hidden_mse(affine_summary_df, PAIR_STARTS)
    plot_latency(affine_summary_df, PAIR_STARTS, baseline_latency)

    # ------------------------------------------------------------------
    # Multiply vs Average
    # ------------------------------------------------------------------
    mul_vs_avg_df = compare_multiply_vs_average(
        affine_summary_df,
        PAIR_STARTS,
    )

    print("\n===== MULTIPLY VS AVERAGE =====")
    display(mul_vs_avg_df)

    n_mul_wins = int(mul_vs_avg_df["Multiply_Better"].sum())

    print(
        f"\nMultiply beats Average on "
        f"{n_mul_wins}/{len(mul_vs_avg_df)} adjacent layer pairs."
    )

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    best_pair_start = int(global_best["Pair_Start"])
    best_method = str(global_best["Method"])

    print("\nSelected by validation:")
    print("Pair:", f"{best_pair_start}-{best_pair_start + 1}")
    print("Method:", best_method)

    final_test = evaluate_final_test(
        best_pair_start,
        best_method,
        all_fitted_maps,
        affine_raw_df,
        original_model,
        test_text,
        tokenizer,
        baseline_params,
    )

    print("\n===== FINAL SELECTED TEST RESULT =====")
    for key, value in final_test.items():
        print(f"{key}:", value)

    # ------------------------------------------------------------------
    # Optional full test table
    # ------------------------------------------------------------------
    if TEST_ALL_PAIRS:
        test_all_df = run_full_test_table(
            PAIR_STARTS,
            all_fitted_maps,
            RANDOM_RUNS,
            test_text,
            tokenizer,
        )

        print("\n===== FULL TEST TABLE =====")
        display(test_all_df)
    else:
        test_all_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Save and interpret
    # ------------------------------------------------------------------
    save_results(
        affine_raw_df,
        affine_summary_df,
        best_per_pair,
        mul_vs_avg_df,
        direct_raw_df,
        direct_summary_df,
        test_all_df,
    )

    best_avg = (
        affine_summary_df[
            affine_summary_df["Method"] == "Average"
        ]
        .sort_values("PPL_mean")
        .iloc[0]
    )

    best_mul = (
        affine_summary_df[
            affine_summary_df["Method"] == "Multiply"
        ]
        .sort_values("PPL_mean")
        .iloc[0]
    )

    print("\n===== AUTOMATIC INTERPRETATION =====")
    print(f"Original validation PPL: {baseline_ppl:.4f}")
    print(
        f"Multiply beat Average on "
        f"{n_mul_wins}/{len(PAIR_STARTS)} pairs."
    )
    print(
        "\nBest Average:",
        "Pair =", best_avg["Pair"],
        ", PPL =", round(best_avg["PPL_mean"], 4),
    )
    print(
        "\nBest Multiply:",
        "Pair =", best_mul["Pair"],
        ", PPL =", round(best_mul["PPL_mean"], 4),
    )
    print(
        "\nBest compressed validation configuration:",
        "Pair =", global_best["Pair"],
        ", Method =", global_best["Method"],
        ", PPL =", round(global_best["PPL_mean"], 4),
    )

    return {
        "affine_raw_df": affine_raw_df,
        "affine_summary_df": affine_summary_df,
        "best_per_pair": best_per_pair,
        "mul_vs_avg_df": mul_vs_avg_df,
        "direct_raw_df": direct_raw_df,
        "direct_summary_df": direct_summary_df,
        "test_all_df": test_all_df,
        "final_test": final_test,
        "all_fitted_maps": all_fitted_maps,
    }
