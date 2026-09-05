"""Builds one small Colab-compatible notebook per experiment phase.

Notebook cells import the real `src/` modules (DRY - nothing reimplemented),
run the phase, then display the saved intermediate results inline (heatmaps,
tables, curves) for the report screenshots.
"""

import json
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(ROOT, "notebooks")
ARTIFACTS = "artifacts"  # relative to notebook dir; os.path.abspath below
SYSPATH = ('import sys, os\n'
           'os.makedirs(os.path.abspath("../artifacts"), exist_ok=True)\n'
           'sys.path.insert(0, os.path.abspath("../src"))\n'
           'ARTIFACTS = os.path.abspath("../artifacts")')

os.makedirs(NOTEBOOK_DIR, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


def _fig(name, caption=None, w="65%"):
    c = f'from IPython.display import Image, display\ndisplay(Image("{ARTIFACTS}/{name}", width="{w}"))'
    cells = [code(c)]
    if caption:
        cells.insert(0, md(caption))
    return cells


def write(phase: str, title: str, body: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.cells = [md(f"# Phase {phase} — {title}"),
                md(f"> {body}"), code(SYSPATH), *cells]
    path = os.path.join(NOTEBOOK_DIR, f"phase{phase}.ipynb")
    nbf.write(nb, path)
    print(f"wrote {path}")


def build_all():
    write("1", "Baseline — train the dummy MLP on MNIST",
          "Trains (or loads) the residual-base classifier on MNIST and records "
          "test accuracy and parameter count.",
          [code("from experiments.phase1_baseline import main\nmain()"),
           md("**Result (from artifacts/phase1_baseline.json):**"),
           code('import json\nprint(json.load(open(f"{ARTIFACTS}/phase1_baseline.json")))')])

    write("2", "Similarity — layer redundancy via diffusion map + NMI",
          "Extracts per-layer activations, embeds them with the diffusion map, "
          "then scores pair-wise redundancy with the joint-Gaussian NMI and the "
          "cosine baseline. High off-diagonal similarity = exploitable "
          "redundancy — yet phase 3 shows it is NOT sufficient: merging still "
          "collapses the plain net, only the residual survives.",
          [code("from experiments.phase2_similarity import main\nmain()"),
           md("**Layer-similarity heatmaps** (residual model, MKA's NMI):"),
           *_fig("similarity_residual_nmi.png"),
           md("The plain (non-residual) MLP's NMI also looks high, but its "
              "layers still cannot be merged safely — redundancy alone is not "
              "enough, residual connections are what make fusion lossless."),
           *_fig("similarity_plain_nmi.png")])

    write("3", "Merging — MKA adaptive fusion of similar layers",
          "Greedily fuses the most similar compatible pair using alpha = S_lm "
          "(paper-1). Accuracy vs number of merges shows the compression-loss "
          "trade-off, and the JSON records WHICH pair (with what similarity and "
          "alpha) was fused at every step.",
          [code("from experiments.phase3_merging import main\nmain()"),
           *_fig("merge_curve.png",
                 "**Merge curve** — accuracy vs #merges (left) and vs parameters "
                 "(right). The first merge even improves accuracy (+0.0006)."),
           code('import json\n'
                'for r in json.load(open(f"{ARTIFACTS}/phase3_merge_curve.json")):\n'
                '    print(r)')])

    write("4", "Merge strategies — weighted vs uniform vs SLERP",
          "The same baseline, the same similarity, one pair merged — but three "
          "different fusion rules (paper-1 adaptive alpha, 50/50, paper-2 SLERP).",
          [code("from experiments.phase4_strategies import main\nmain()"),
           *_fig("strategies.png",
                 "**Strategy comparison** — all three preserve accuracy on the "
                 "redundant residual pair; alpha values are recorded in the JSON."),
           code('import json\nfor r in json.load(open(f"{ARTIFACTS}/phase4_strategies.json")):\n'
                '    print(f\'{r["strategy"]:<9} acc={r["acc"]:.4f} params={r["params"]:,} '
                'alpha={r.get("alpha", float("nan")):.3f} pair={r.get("pair")}\')')])

    write("5", "Pruning — LRP attribution vs magnitude vs random (SparC3)",
          "Scores every weight by its LRP attribution aggregated over reference "
          "samples, then globally prunes the least relevant `q` elements at "
          "several sparsity levels, comparing against magnitude and random.",
          [code("from experiments.phase5_pruning import main\nmain()"),
           *_fig("pruning_curves.png",
                 "**Pruning paths** — accuracy vs sparsity per method. LRP beats "
                 "random at every sparsity; magnitude stays ahead on clean MNIST "
                 "(an honest nuance)."),
           code('import json, numpy as np, matplotlib.pyplot as plt\n'
                'rows = json.load(open(f"{ARTIFACTS}/phase5_pruning.json"))\n'
                'for m in ("lrp", "magnitude", "random"):\n'
                '    xs = [r["sparsity"] for r in rows if r["method"]==m]\n'
                '    ys = [r["acc"] for r in rows if r["method"]==m]\n'
                '    plt.plot(xs, ys, "o-", label=m)\n'
                'plt.xlabel("sparsity"); plt.ylabel("test accuracy")\n'
                'plt.legend(); plt.title("Attribution-guided pruning"); plt.show()\n'),
           *_fig("lrp_input_relevance.png",
                 "**LRP input-relevance map** — hot pixels are the evidence the "
                 "network uses for its prediction (interpretability side of SparC3).")])

    write("6", "End-to-end pipeline — merging + pruning, final report table",
          "Runs both compression tracks (MKA merging on the residual net, SparC3 "
          "LRP pruning on the plain net), writes the full before/after evidence "
          "table and the Gantt chart used in the Project-Management section.",
          [code("from experiments.phase6_pipeline import main\nmain()"),
           *_fig("results_table.png",
                 "**Compression evidence** — every track vs its own baseline, "
                 "with savings (params/FLOPs/size/latency) and retention."),
           md("**Machine-readable snapshots (artifacts/final_results.csv):**"),
           code('import csv\nprint("\\n".join(open(f"{ARTIFACTS}/final_results.csv").read().splitlines()))'),
           *_fig("gantt.png", "**Gantt chart for the report.**")])


if __name__ == "__main__":
    build_all()