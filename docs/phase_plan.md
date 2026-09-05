# Phase-Wise Plan

Assignment deadline: **Sunday, Sep 06, 2026, 11:59 PM**

> Two tracks run in parallel: **(R)** report writing and **(E)** experiments.
> Each `docs/papers/*` file feeds a report section; each `experiments/phase_*.py`
> feeds a report figure/table.

## Track E — Experiments (each phase = 1 notebook + 1 script)

| Phase | Focus | Source module(s) | Deliverable |
|---|---|---|---|
| **1. Baseline** | Train `ResidualMLP`/`SimpleMLP(256⁴)` on MNIST; snapshot footprint | `models/` `data/` `evaluation/` | trained model + accuracy/FLOPs/size/latency |
| **2. Similarity** | Layer redundancy (Diffusion map + NMI vs cosine) | `similarity/` | heatmaps + 2×2 grid (`similarity_all.png`) |
| **3. Merging** | Greedy α=S_lm merging, per-merge pair+α evidence | `merging/` | `phase3_merge_curve.json` + `merge_curve.png` |
| **4. Strategies** | Weighted vs uniform vs SLERP (same pair, α recorded) | `merging/` | `phase4_strategies.json` + `strategies.png` |
| **5. Pruning** | LRP vs magnitude vs random sweep | `pruning/` | `phase5_pruning.json` + `pruning_curves.png` |
| **6. Pipeline** | Two tracks end-to-end: full-metric evidence table | everything | `final_results.csv/json` + `results_table.png` + Gantt |
| **Reproduce** | Multi-seed aggregation (mean±std) | `scripts/reproduce.py` | `reproduce_summary.csv/json` + `reproduce_table.png` |

## Track R — Report (maps to the 50% rubric)

| Rubric section | Material already prepared |
|---|---|
| Introduction | `docs/papers/*` motivation + NLP view (write 3 paragraphs) |
| Related Work | `docs/paper_comparison.md` + results tables in `01_mka_summary.md` |
| Datasets | `data/mnist.py` (balanced, labelled, 70k examples) |
| Experimental Plan | `docs/phase_plan.md` + `experiments/phase_*.py` |
| Project Management | Gantt in `notebooks/06` + resources/risk sections |
| Reproduce results | `experiments/phase2-6` |
| Findings | written from `experiments/phase6_pipeline.py` outputs |
| Team contribution | add in report |

## Workflow per phase

1. Read the matching paper section (see `docs/papers/`).
2. Run `python scripts/run_phases.py` (or `make phases`) at the repo root.
3. Open `notebooks/phaseN.ipynb` (regenerate with `python scripts/build_notebooks.py`)
   in Colab to see the same steps + intermediate results. The notebooks call the
   very same `experiments/phase*` entry points the CLI runs.
4. Note down numbers + figures → paste into report (artifacts live in `artifacts/`).

Every tunable is a Config field (`src/config/default.py`), overridable via `.env`
(`make env`). The full suite of tests runs with `make test`; the module-level
sanity check is `make smoke`. On a fresh clone: `make setup && make test`.

## Resource estimate

- CPU-only (Colab free tier) is enough; each run trains a small MLP in < 2 min.
- Disk < 500 MB (MNIST + artifacts).
- 3 seeds per experiment for robustness (mean ± std).

## Risk register

| Risk | Mitigation |
|---|---|
| Merging collapses accuracy | Start with 1 merge; use α=S_lm (paper's own heuristic); test SLERP |
| NMI unstable on small calibration sets | Diffusion-map embeddings (dimensionality reduction) keep it stable; use ≥256 samples |
| LRP conservation off | Use ε-relaxed rule + softmax initialization; document as approximation |
| Colab session resets | Scripts are re-runnable; save artifacts to disk after every phase |
| Different layer shapes block merges | Symmetric architecture + shape-filter in merger |

## Success criteria

- [x] Baseline ≥ 96% test accuracy (97.8%)
- [x] ≥ 2 layer-pairs merged with < 2% accuracy drop (3 merges → 97.06%)
- [x] ≥ 20% parameter reduction (1.73× on the residual model)
- [x] All three paper ideas reproduced on a small open model
- [ ] Report covers every rubric section; Gantt + team contribution included
- [ ] Code + screenshots pushed to GitHub, PDF submitted