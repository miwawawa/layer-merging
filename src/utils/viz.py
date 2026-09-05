"""All plotting helpers. Every figure the experiments/report needs lives here.

Figures deliberately have a *consistent* look (matplotlib "seaborn-v0_8" palette,
150 dpi, tight boxes) so report screenshots look professional and uniform.
"""

import os
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

#: colour-blind-friendly palette (matplotlib v0 "seaborn" qualitative Set1-like)
PALETTE = ["#4C72B0", "#C44E52", "#55A868", "#8172B3", "#CCB974", "#64B5CD"]


def save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------

def similarity_heatmap(S: np.ndarray, path: str, title: str = "Layer similarity") -> str:
    """S is (num_layers, num_layers) — colour-code how redundant layers are."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(S, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(S.shape[0]))
    ax.set_yticks(range(S.shape[0]))
    ax.set_xticklabels([f"L{i}" for i in range(S.shape[0])])
    ax.set_yticklabels([f"L{i}" for i in range(S.shape[0])])
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return save(fig, path)


def similarity_grid(matrices: Dict[str, np.ndarray], path: str) -> str:
    """One 2x2 figure for the four similarity matrices (residual/plain x NMI/cos).

    ``matrices`` = {"residual_nmi": S, "residual_cosine": S, "plain_nmi": S,
                    "plain_cosine": S}. Keys are rendered as titles.
    """
    n = len(matrices)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, S) in zip(axes, matrices.items()):
        im = ax.imshow(S, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(S.shape[0]))
        ax.set_yticks(range(S.shape[0]))
        ax.set_xticklabels([f"L{i}" for i in range(S.shape[0])], fontsize=7)
        ax.set_yticklabels([f"L{i}" for i in range(S.shape[0])], fontsize=7)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title(name.replace("_", " ").title(), fontsize=10)
        fig.colorbar(im, ax=ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Layer similarity — NMI (MKA) vs cosine, both architectures")
    plt.tight_layout()
    return save(fig, path)


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------

def merge_curve(merges: Sequence[int], acc: Sequence[float],
                params: Sequence[int], path: str) -> str:
    """Two-panel merge story: accuracy-to-merges and the accuracy-vs-params curve."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(merges, acc, "o-", color=PALETTE[0], lw=1.8)
    for m, a in zip(merges, acc):
        ax.annotate(f"{a:.3f}", (m, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Merged layer pairs")
    ax.set_ylabel("Test accuracy")
    ax.set_title("(a) Accuracy vs #merges")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(params, acc, "s-", color=PALETTE[1], lw=1.8)
    for m, p, a in zip(merges, params, acc):
        ax.annotate(f"{m} merges", (p, a), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8)
    ax.set_xlabel("Model parameters")
    ax.set_ylabel("Test accuracy")
    ax.set_title("(b) Accuracy vs parameters (shrinkage)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return save(fig, path)


def strategies_bars(labels: List[str], acc: List[float], params: List[int],
                    baseline_acc: float, path: str) -> str:
    """MKA-vs-uniform-vs-SLERP bars against the *no-merge* baseline reference."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    idx = np.arange(len(labels))
    bars = ax.bar(idx, acc, color=PALETTE[: len(labels)], width=0.6)
    ax.axhline(baseline_acc, color="k", ls="--", lw=1.2,
               label=f"baseline {baseline_acc:.4f}")
    for b, a, p in zip(bars, acc, params):
        ax.annotate(f"{a:.4f}\n({p:,} params)",
                    (b.get_x() + b.get_width() / 2, a),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Merge strategies — one merge from the same baseline")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return save(fig, path)


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------

def pruning_curves(sparsity_by_method: Dict[str, List[float]],
                   acc_by_method: Dict[str, List[float]], path: str,
                   operating_pts: Optional[Dict[str, float]] = None) -> str:
    """Accuracy-vs-sparsity lines for lrp / magnitude / random (SparC3 study)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (method, spars) in enumerate(sparsity_by_method.items()):
        ax.plot(spars, acc_by_method[method], "o-", color=PALETTE[i % len(PALETTE)],
                lw=1.8, label=method)
    if operating_pts:
        for method, sp in operating_pts.items():
            ax.axvline(sp, color="grey", ls=":", lw=1.0, alpha=0.8)
        ax.plot([0], [0], color="grey", ls=":", lw=1.0,
                label="pipeline operating point")
    ax.set_xlabel("Sparsity (fraction of parameters zeroed)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Pruning path: LRP attribution vs magnitude vs random")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return save(fig, path)


# --------------------------------------------------------------------------
# generic pieces
# --------------------------------------------------------------------------

def curve(x: np.ndarray, y: np.ndarray, path: str, xlabel: str, ylabel: str,
          title: str = "", marker: str = "o-") -> str:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(x, y, marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return save(fig, path)


def grouped_bars(labels, values, path, ylabel: str, title: str = "", colors=None,
                 baseline: Optional[float] = None, value_labels: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    idx = np.arange(len(labels))
    bars = ax.bar(idx, values, color=colors if colors else "steelblue")
    if baseline is not None:
        ax.axhline(baseline, color="k", ls="--", lw=1.2, label=f"baseline {baseline:.4f}")
    if value_labels:
        for b, v in zip(bars, values):
            ax.annotate(f"{v:.4g}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if baseline is not None:
        ax.legend(loc="best")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return save(fig, path)


def results_table(rows: List[dict], path: str, order: Optional[List[str]] = None,
                  title: str = "Results — before vs after compression") -> str:
    """Render a list of metric dicts as an embeddable table image.

    Values are formatted nicely (params with separators, percentages %,
    savings x). This is the *evidence table*: the before/after story at a gance.
    """
    rows = [dict(r) for r in rows]                      # don't mutate caller data
    order = order or (list(rows[0]) if rows else [])
    headers = [_pretty(key) for key in order]
    _format_rows(rows, order)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.35 * len(order)), 0.55 * (len(rows) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=[[str(r.get(k, "")) for k in order] for r in rows],
                   colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef2f7")
    ax.set_title(title, pad=18)
    plt.tight_layout()
    return save(fig, path)


def relevance_map(flat_vals: np.ndarray, path: str, shape=(28, 28), title: str = "Relevance") -> str:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(flat_vals.reshape(shape).max(axis=-1) if flat_vals.ndim > 1 else
              flat_vals.reshape(shape), cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return save(fig, path)


def gantt(tasks, starts, durations, path: str) -> str:
    """Tasks of a single-track plan (e.g. the 6 phases over the assignment week)."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    yy = np.arange(len(tasks))[::-1]
    ax.barh(yy, durations, left=starts, height=0.5, color="#4C72B0")
    ax.set_yticks(yy)
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Day of plan")
    ax.set_title("Project Gantt — layer merging assignment")
    plt.tight_layout()
    return save(fig, path)


# --------------------------------------------------------------------------
# ablations (phase 4b alpha sweep, phase 6b hybrid) and reduce-to-table helpers
# --------------------------------------------------------------------------

def alpha_sweep_plot(alphas, accs, baseline_acc: float, path: str) -> str:
    """α vs post-merge accuracy, with the no-merge baseline as a dashed line."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(alphas, accs, marker="o", color="#4C72B0", linewidth=2)
    ax.axhline(baseline_acc, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"no-merge baseline {100 * baseline_acc:.2f}%")
    for x, y in zip(alphas, accs):
        ax.annotate(f"{100 * y:.2f}%", (x, y), textcoords="offset points",
                    xytext=(4, 8), fontsize=7)
    ax.set_xlabel("Mixing weight $\\alpha$")
    ax.set_ylabel("Test accuracy")
    ax.set_title("α-ablation — one merged pair at each α")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return save(fig, path)


def hybrid_bars(rows: List[dict], path: str) -> str:
    """Grouped bars for the cross-track feasibility ablation (phase 6b).

    The residual group has no 'merged + LRP' bar: run-time LRP on a residual
    stream is structurally undefined (the honest finding), so that entry has
    ``acc=None`` and is rendered as a hatched 'n/a' column instead.
    """
    plain, residual = rows[:3], rows[3:5]
    stages = [r["stage"] for r in plain]
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    width = 0.38
    groups = [("plain", plain, "#4C72B0", 0.0),
              ("residual", residual + [None], "#55A868", width)]
    for label, slice_, color, off in groups:
        xs = np.arange(len(stages)) + off
        accs = [r["acc"] if r else None for r in slice_]
        bars = ax.bar(xs, [a if a is not None else 0.0 for a in accs], width,
                      label=label, color=color)
        for b, a in zip(bars, accs):
            if a is None:
                b.set_hatch("///")
                ax.text(b.get_x() + b.get_width() / 2, 0.35, "n/a",
                        ha="center", va="center", fontsize=7, rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, a + 0.003,
                        f"{100 * a:.1f}%", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(stages)) + width / 2)
    ax.set_xticklabels(stages, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cross-track feasibility — merging vs pruning on each graph")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return save(fig, path)


# --------------------------------------------------------------------------
# small formatters shared by the table figure
# --------------------------------------------------------------------------

def _pretty(key: str) -> str:
    return key.replace("_", " ").title()


def _format_rows(rows: List[dict], order: List[str]) -> None:
    """Format common metric values in place (kept separate for testability)."""
    fraction_cols = ("acc_drop", "retention", "sparsity", "weight_sparsity")
    for r in rows:
        for k in list(order):
            v = r.get(k)
            if v is None or isinstance(v, str):
                continue
            if k in ("params", "flops", "nonzero"):
                r[k] = f"{v:,}"
            elif k.endswith("_mb"):
                r[k] = f"{v:.1f} MB"
            elif k == "acc":
                r[k] = f"{100 * v:.2f}%"
            elif k.endswith("_savings") or k in fraction_cols:
                r[k] = f"{100 * v:.1f}%"               # fractions shown as %
            elif k.endswith("_ms"):
                r[k] = f"{v:.3f} ms"
            elif k == "speedup":
                r[k] = f"{v:.2f}x"
            elif isinstance(v, float):
                r[k] = f"{v:.4f}"