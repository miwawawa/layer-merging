"""Visualization functions."""

import matplotlib.pyplot as plt


METHODS = [
    "Remove",
    "Average",
    "Random-Weight Average",
    "Multiply",
]


def _plot_metric(summary_df, pair_starts, metric, ylabel, title,
                 baseline=None, baseline_label=None):
    plt.figure(figsize=(10, 5))

    for method in METHODS:
        group = (
            summary_df[summary_df["Method"] == method]
            .sort_values("Pair_Start")
        )

        plt.plot(
            group["Pair_Start"],
            group[metric],
            marker="o",
            label=method,
        )

    if baseline is not None:
        plt.axhline(
            baseline,
            linestyle="--",
            label=baseline_label,
        )

    plt.xticks(
        pair_starts,
        [f"{L}-{L+1}" for L in pair_starts],
        rotation=45,
    )
    plt.xlabel("Merged layer pair")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ppl(summary_df, pair_starts, baseline_ppl):
    """Plot validation PPL across layer pairs."""
    _plot_metric(
        summary_df,
        pair_starts,
        "PPL_mean",
        "Validation Perplexity (lower is better)",
        "GPT-2 Layer Merging: PPL Across All Adjacent Pairs",
        baseline=baseline_ppl,
        baseline_label="Original GPT-2",
    )


def plot_hidden_mse(summary_df, pair_starts):
    """Plot hidden-state reconstruction MSE."""
    _plot_metric(
        summary_df,
        pair_starts,
        "Hidden_MSE_mean",
        "Hidden-state MSE (lower is better)",
        "GPT-2 Layer Merging: Reconstruction Error Across All Pairs",
    )


def plot_latency(summary_df, pair_starts, baseline_latency):
    """Plot inference latency across layer pairs."""
    _plot_metric(
        summary_df,
        pair_starts,
        "Latency_ms_mean",
        "Latency (ms)",
        "Inference Latency Across All Adjacent Pairs",
        baseline=baseline_latency,
        baseline_label="Original GPT-2",
    )
