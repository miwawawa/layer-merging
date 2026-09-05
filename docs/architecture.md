# Architecture — Module Map & Data Flow

## Package layout (`src/`)

```
src/
├── config/default.py         # ALL parameters; .env-aware, PROJECT_ROOT-anchored paths
├── data/
│   └── mnist.py              # MNIST IDX loader (no torchvision); mirror URL from Config
├── models/
│   ├── base.py               # BaseNN: contract every trainable model follows
│   ├── mlp.py                # SimpleMLP (non-residual baseline; required ctor args)
│   └── residual_mlp.py       # ResidualMLP: transformer-like residual blocks
├── similarity/               # "how similar are two layers?"  (Strategy pattern)
│   ├── protocol.py           # LayerSimilarity ABC → returns S matrix
│   ├── diffusion.py          # DiffusionMap projector (paper-1 manifold learning)
│   ├── gaussian_mi.py        # NMI similarity under joint-Gaussian assumption (paper-1 core)
│   ├── cosine.py             # Cosine similarity (baseline / paper-2 structural)
│   └── factory.py            # similarity = get_similarity('nmi' | 'cosine')
├── merging/                  # "how should two layers fuse?" (Strategy pattern)
│   ├── protocol.py           # MergeStrategy ABC (weight + fuse)
│   ├── weighted.py           # WeightedMerge: α=S_lm (paper-1), softmax option
│   ├── uniform.py            # UniformMerge: α=0.5
│   ├── slerp.py              # SlerpMerge: spherical interpolation (paper-2)
│   ├── factory.py            # merge = get_merger('weighted' | 'uniform' | 'slerp')
│   └── merger.py             # LayerMerger: greedy pair-selection + rewiring (Facade)
├── pruning/                  # "which weights to drop?" (paper-3)
│   ├── lrp.py                # LRP relevance propagation (explanations + scoring)
│   ├── importance.py         # weight-importance scoring on reference samples
│   └── pruner.py             # masks: prune_mask / apply_prune / prune_model(…global)
├── evaluation/
│   └── metrics.py            # accuracy, param count, compression, non-zero params
├── training.py               # the single train loop (optimizer/loss from Config)
├── utils/
│   ├── seed.py               # reproducibility
│   ├── activations.py        # extract per-layer activations from a data loader
│   └── viz.py                # all plotting helpers
└── experiments/              # one script per phase (orchestration only, ~DRY)
    ├── common.py             # setup()/build_model()/make_similarity()/make_merger()
    ├── phase1_baseline.py    #   load/train baseline  + test accuracy
    ├── phase2_similarity.py  #   layer redundancy heatmaps (NMI + cosine)
    ├── phase3_merging.py     #   greedy α=S_lm merge curve (acc vs compression)
    ├── phase4_strategies.py  #   weighted vs uniform vs SLERP, one merge each
    ├── phase4b_alpha_sweep.py#   α-ablation (pair > weight evidence)
    ├── phase5_pruning.py     #   LRP vs magnitude vs random sparsity sweep
    └── phase6_pipeline.py    #   merging track + pruning track → final_results.csv
    └── phase6b_hybrid.py     #   why the two tracks can't live on one model

scripts/   smoke_test.py · run_phases.py · reproduce.py · build_notebooks.py · run_notebooks.py
tests/     pytest suite (unit per module + toy integration)
notebooks/ phase1..6.ipynb (Colab-compatible, generated)
docs/      paper summaries · survey · rubric · guidelines · plan · architecture
```

## Data flow (who calls whom)

```
data/mnist.py ─────────────► experiments/common.setup  (loaders)
models/* ──────────────────► experiments/common.build_model (from Config)
config/default.py ─────────► any module needing a tunable (single source)

utils/activations.py ──────► similarity/*   (feeds LayerSimilarity.compute)
similarity/diffusion.py ───► similarity/gaussian_mi.py   (embed → score)
similarity/factory.py ─────► merging/merger.py           (S matrix)
merging/merger.py ─────────► merging/*strategies         (fuse params)
merging/merger.py ─────────► models/base.py              (rewire .linears)

pruning/importance.py ─────► pruning/pruner.py
pruning/lrp.py ────────────► pruning/importance.py (optionally)
evaluation/metrics.py ─────► ALL experiments          (score everything)
utils/viz.py ──────────────► ALL experiments          (figures)
utils/seed.py ─────────────► ALL experiments          (reproducibility)
```

## Design decisions (SOLID / DRY / YAGNI)

- **S — Single Responsibility:** each file does exactly one thing
  (`SlerpMerge` only knows how to interpolate on a sphere).
- **O — Open/Closed:** add a new similarity metric or merge strategy by
  subclassing the ABC and registering it in the factory — no changes to `merger.py`.
- **L — Liskov:** strategies are interchangeable via the ABCs; `LayerMerger` never
  subclasses.
- **I — Interfaces:** `LayerSimilarity.compute(activations) -> S`
  and `MergeStrategy.weight(...) + fuse(...)` are tiny, focused contracts.
- **D — Dependency Inversion:** `LayerMerger` depends on the *abstractions*
  (`LayerSimilarity`, `MergeStrategy`), not concrete classes.
- **DRY:** shared logic lives in `utils/` and the ABCs; experiments only orchestrate —
  every tunable is a Config field, so `experiments/common.py` is the single place that
  touches Config and the filesystem.
- **No hardcoding policy:** nothing is baked into models or phase scripts — dataset
  shape, optimizer, loss, merge threshold, pruning ratios, mirror URL all come from
  `Config` (override via `.env`). Constructor arguments beat env overrides so tests and
  quick scripts never get clobbered by `.env`.
- **YAGNI:** no CLI framework or persistence layer — `scripts/` + the `Config` dataclass
  + a `Makefile` cover it.

## Config + environment (the "any machine" story)

`src/config/default.py` defines every tunable as a dataclass field. `.env` (or exported
`LAYER_MERGING_<NAME>` vars, parsed with Python literals) overrides any field. Paths to
`data/` and `artifacts/` are anchored to `PROJECT_ROOT` (discovered from the file
location), so notebooks, scripts, and pytest all resolve the same files regardless of
the launching directory.

- `make setup`  → `pip install -r requirements.txt`
- `make env`    → copy `.env.example` to `.env`
- `make test`   → pytest suite
- `make smoke`  → toy-scale end-to-end check
- `make phases` → full reproduction into `artifacts/`
- `make run-notebooks` → proves the Colab notebooks execute

## Why all hidden-layer sizes are equal

Merging fuses `θ_l` and `θ_m` *element-wise*, so both layers must have identical
weight shapes. Transformer layers share `d_model`, so MKA works there directly.
Our `SimpleMLP` uses **symmetric** hidden dims (`256→256→256→256`) to reproduce
the same condition because shallow networks with different-width layers would
otherwise be unmergable. The first/last layers touch input/output dims and may be
non-mergable; the merger filters pairs whose shapes differ.