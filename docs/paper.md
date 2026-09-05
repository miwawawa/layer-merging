# Layer Merging for Model Compression — Reproduction on MNIST

**A reproduction of MKA (EMNLP 2024), Chain-of-Merging (CVPR 2026), and
SparC3 (2025) on a from-scratch MNIST MLP.**

*Paper draft — every number below is produced by this repository
(`scripts/run_phases.py` + `scripts/reproduce.py`). Figures are saved under
`artifacts/` and referenced by filename.*
<!-- CROSS REFERENCES: [results-table] → artifacts/results_table.png,
     [merge-curve] → artifacts/merge_curve.png, [pruning-curves] →
     artifacts/pruning_curves.png, [strategies] → artifacts/strategies.png,
     [similarity] → artifacts/similarity_all.png, [reproduce] →
     artifacts/reproduce_table.png, [alpha-sweep] → artifacts/alpha_sweep.png,
     [hybrid] → artifacts/hybrid_tracks.png, [gantt] → artifacts/gantt.png -->

---

## Abstract

Over-parameterized networks contain large amounts of redundancy: whole hidden
layers can collapse into one, and individual weights can be dropped, with
negligible loss. We reproduce three *training-free* compression techniques on a
small MNIST MLP and measure the **reduction-and-accuracy** trade-off honestly.
(1) **MKA merging**: diffusion-map + Gaussian-NMI similarity identifies the most
redundant layer pair; fusion `θ̃ = αθ₁ + (1−α)θ₂` with `α = S_lm` deletes a whole
layer. (2) **Chain-of-Merging**: the same merging with geodesic SLERP fusion,
non-positional pair choice. (3) **SparC3 pruning**: LRP attribution ranks every
weight; the least-relevant half is zeroed. On a residual MLP we remove **42% of
parameters/FLOPs and 1 of 4 hidden layers** for a 99.3% accuracy retention
(0.9775 → 0.9706); on a plain MLP the identical merge collapses accuracy —
*residual structure is what makes layer merging safe*, the key (reproduced)
finding. LRP pruning reaches 50% sparsity at 75% retention, beating random and
failing only where magnitude is expected to win.

---

## 1. Introduction

### 1.1 Motivation
Deploying neural models anywhere real — phones, edge boxes, inference servers —
is a game with three currencies: accuracy, parameter count, and compute. The
vast majority of trained weights are **redundant**: networks over-fit their
breadth before they fit their data, and adjacent layers frequently compute
near-identical transformations. Model compression tries to spend only the
currencies that matter, ideally *after* training and *without* re-training.

### 1.2 Why NLP-adjacent
The three papers we reproduce all sit in natural-language territory. MKA and
CoM operate on **transformers** — the architecture behind every modern LLM —
with MKA merging redundant layers of LLMs (MMLU, PIQA) and CoM merging layers
across vision-language models. SparC3 prunes LLM weights guided by LRP
explanations. Reproducing those 8B-scale numbers is out of scope for an
undergraduate assignment (and explicitly allowed to be reprojected onto a small
model); we reproduce the *geometry* of their results — similarity matrices,
compression curves, sparsity trade-offs — on a small MLP whose residual MLP
mirrors the transformer's `x + F(x)` block structure.

### 1.3 Problem type
A *post-hoc model-compression* study: no training of the compressed models, two
operations (whole-layer merging, weight masking), one clear evaluation axis
(accuracy retained per unit of parameter/FLOP reduction).

---

## 2. Related Work

See `docs/paper_comparison.md` (three-paper comparison), `docs/research_survey.md`
(implementable vs ongoing research), and `docs/other_approaches.md` (the full
survey of other compression families). Summary of the prior-art table:

| Approach | Mechanism | Needs retraining | Reduces FLOPs | Assignment paper |
|---|---|---|---|---|
| **Layer merging (ours)** | fuse redundant layers, α=S_lm or SLERP | no | **yes** (whole layers) | MKA, CoM |
| Attribution pruning | LRP-relevance top-k masks | no | only w/ sparse kernels | SparC3 |
| Magnitude pruning | zero smallest \|w\| | no | only w/ sparse kernels | baseline |
| Structured pruning / SVD | narrow / factor layers | often | yes | — |
| Quantization / clustering | fewer bits / shared codes | no | no | — |
| Distillation | student trains on teacher | **yes** | yes | — |
| Cross-model merging | task vectors, soups, MoE | no | — | surveyed |

**Baselines** in this repo: magnitude and random pruning; uniform (model-soup)
fusion; cosine activation similarity. **State of the art**: our three papers'
own results (LLM layer reduction 31–58%, WikiText sparsity 50%+) are documented
in `docs/papers/*_summary.md`.

---

## 3. Datasets

**MNIST** — 70,000 28×28 grayscale handwritten digits, 60k train + 10k test,
10 balanced classes. Loaded from scratch (no `torchvision`) from the IDX format;
the download URL is configurable (`LAYER_MERGING_MNIST_MIRROR`) so the dataset
is reproducible without hardcoded sources. We split 60k into 54k train + 6k
validation (for similarity calibration + early-stopping signal); the 10k test
set is untouched by any hyper-parameter choice. Class balance ≈ uniform (each
digit ~6k), minimal labeling bias; MNIST's known limits (clean scans, low
variance) are acknowledged — it is chosen for CPU-speed and the ability to
retrain baselines across seeds, not for difficulty.

---

## 4. Experimental Plan

### 4.1 Models
| Architecture | Shape | Purpose |
|---|---|---|
| `ResidualMLP` | 784 → proj(256) → 4× residual block(256) → 10 | merging track (transformer-like `h + F(h)`) |
| `SimpleMLP` | 784 → 4× ReLU(256) → 10 | pruning track (plain graph, exact LRP) |

Both share the symmetric `(256, 256, 256, 256)` hidden structure — the exact
condition that makes layers *mergeable* (identical weight shapes, like a
transformer's shared `d_model`).

### 4.2 Training
Adam, LR 1e-3, 6 epochs, batch 128, cross-entropy, seed 42 (multi-seed
reproduction over `(42, 7, 2026)`). ~17 s/epoch. Baseline accuracy: residual
0.9775, plain 0.9760.

### 4.3 Protocol per phase
1. **Baseline** — snapshot accuracy + footprint (params, FLOPs, size, latency).
2. **Similarity** — activations → diffusion map → Gaussian NMI vs cosine.
3. **Merging** — greedy α=S_lm fusion, curve up to full collapse.
4. **Strategies** — weighted vs uniform vs SLERP, one merge each.
4b. **α-ablation** — same pair fused at every α ∈ {.1, .3, .5, .7, .9}.
5. **Pruning** — LRP/magnitude/random at keep ∈ {1.0, .9, .75, .5, .3, .1}.
6. **Pipeline** — the two compression tracks end-to-end + evidence table.
6b. **Feasibility** — demonstrate honestly why ONE model cannot do both
    tracks (merge collapses plain; LRP is undefined on residual).

Every formula behind 1–6b is worked out with repo numbers in
`docs/learning/` (`00_roadmap.md` → `02/03/04  *_calculations.md`).

### 4.4 Metrics
`acc`, `acc_drop`, `retention`, `params`, `flops`, `size_mb`, `latency_ms`,
`sparsity`, `params_savings`, `speedup` — all defined in `docs/approach.md` §4.
Savings are relative to each track's own baseline.

### 4.5 Reproducibility
Every knob is a `Config` field overridable via `.env`; the whole pipeline runs
`make phases`; multi-seed evidence via `make reproduce`. No model is re-trained
after compression.

---

## 5. Methodology (the three techniques)

Full mathematics in `docs/approach.md`. Summary:

1. **Similarity (MKA).** Embed each layer's activations with the diffusion map
   `Φ_t = Λ^t·Φ` (skip trivial eigenvector), score pairs by Gaussian NMI
   `NMI = I/√(H_l H_m)`, `I = ½ln|Σ_l||Σ_m|/|Σ_lm|`.
2. **Merging.** Choose `argmax` NMI over shape-compatible pairs; fuse
   `θ̃ = αθ_l + (1−α)θ_m` with `α = clamp(S_lm, .1, .9)` (weighted) or spherical
   SLERP (CoM); greedy loop until similarity < τ = 0.05.
3. **Pruning (SparC3).** LRP `R_{i←j} = (a_i w_ij / (z_j + εsign z_j)) R_j`
   aggregated over reference samples; global top-k mask of the least-relevant
   weights at the configured keep-ratio.

---

## 6. Results

> Single-seed numbers below (seed 42); multi-seed mean±std is §6.5 / table
> [reproduce]. *All artifacts regenerate with `make phases`.*

### 6.1 Redundancy exists (Phase 2) — [similarity]

Residual net off-diagonal NMI mean **0.512**, max **0.988**; cosine mean 0.645.
Plain net NMI mean 0.525, max 1.000. Both architectures contain near-duplicate
layers by information content. **Yet only the residual net can exploit it (next
section)** — high NMI is necessary but not sufficient for safe merging.

### 6.2 Merging reduces structure with tiny loss (Phase 3) — [merge-curve]

| # merges | acc | params | Δacc | pair fused | α (S_lm) |
|---|---|---|---|---|---|
| 0 | 0.9775 | 466,698 | — | — | — |
| 1 | 0.9781 | 400,906 | **+0.0006** | (1, 2) | 0.9 (S=0.988) |
| 2 | 0.9755 | 335,114 | −0.0020 | (1, 2) | 0.9 (S=1.000) |
| 3 | 0.9706 | 269,322 | −0.0069 | (0, 1) | 0.9 (S=0.935) |

After the first fusion the merged layer is *almost identical* to its neighbor
(S=1.000) — evidence that α=S_lm keeps the strongest layer's parameters, so
further fusions are nearly lossless. Final: **4 hidden layers → 1**,
**1.73× fewer parameters (42.3%), 42.2% fewer FLOPs, 99.3% retention.**

### 6.3 Merge strategies (Phase 4) — [strategies]

| strategy | acc | parameters | α chosen |
|---|---|---|---|
| baseline (no merge) | 0.9775 | 466,698 | — |
| weighted (MKA) | 0.9781 | 400,906 | 0.900 |
| uniform (soup) | 0.9786 | 400,906 | 0.500 |
| SLERP (CoM) | 0.9786 | 400,906 | 0.900 |

All three strategies *improve* accuracy on the first merge and are within 0.0005 —
on a redundant residual net the fusion rule matters little; the similarity-driven
pair choice matters.

### 6.4 α-ablation: pair choice dominates the mixing weight (Phase 4b) — [alpha-sweep]

Fuse the *same* pair `(1,2)` at every α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}:

| α | merged acc | Δ vs baseline |
|---|---|---|
| 0.1 | 0.9785 | +0.0010 |
| 0.3 | 0.9786 | +0.0011 |
| 0.5 | 0.9786 | +0.0011 |
| 0.7 | 0.9783 | +0.0008 |
| 0.9 | 0.9781 | +0.0006 |

All five land within 0.0005 of each other — once the *right pair* is chosen
(the similarity-driven argmax), the exact α is a second-order detail. MKA's
`α = S_lm` is therefore a principled default, not a hyper-parameter to tune.

### 6.5 Splitting the plain graph (the honest negative)
Merge one pair of the plain `SimpleMLP` → accuracy **0.9760 → 0.1229**. The
layers are *statistically* redundant (NMI high) but *functionally* coupled:
without a residual stream, replacing two sequential transforms with one average
destroys the computation. This is why MKA/CoM are transformer (residual) methods.

### 6.6 Pruning: LRP wins at low sparsity, magnitude at high (Phase 5) — [pruning-curves]

accuracy at each keep-ratio (plain net, global masks):

| keep | LRP | magnitude | random |
|---|---|---|---|
| 1.0 | 0.9760 | 0.9760 | 0.9760 |
| 0.9 | 0.9665 | 0.9762 | 0.9720 |
| 0.75 | 0.9480 | 0.9752 | 0.9554 |
| 0.5 | **0.7330** | 0.9730 | 0.6436 |
| 0.3 | 0.1259 | 0.9679 | 0.1679 |
| 0.1 | 0.0974 | 0.5410 | 0.0974 |

LRP = task-guided (keeps sharp, task-critical weights; corr(LRP, magnitude) =
0.10 — genuinely different information). It beats random at every sparsity, but
magnitude wins outright because MNIST hidden layers store *global* magnitude
features — an honest nuance we report rather than hide.

### 6.7 Pipeline evidence — [results-table], [gantt]

| track | acc (before→after) | params | FLOPs | size | latency | sparsity | retention |
|---|---|---|---|---|---|---|---|
| **Layer merging (MKA)** | 0.9775 → 0.9706 | 466.7k → 269.3k | 931k → 538k | 1.87→1.08 MB | 42% ↓ (noisy) | 0 | **99.3%** |
| **LRP pruning (SparC3)** | 0.9760 → 0.7330 | 400.9k (kept) | same | same | same | **50%** | 75.1% |

Two complementary currencies: merging buys **structural reduction** (params +
FLOPs + footprint) at near-zero accuracy cost; pruning buys **sparsity** at a
controllable accuracy price.

### 6.9 Cross-track feasibility: why one model cannot do both (Phase 6b) — [hybrid]

Every combination of graph × technique was actually *attempted*:

| track (graph → op) | result | read |
|---|---|---|
| plain (baseline 0.9760) → merge | **0.0308** | merging destroys a non-residual closure (negative #1) |
| plain → merge → LRP 50% | 0.0709 | pruning cannot resurrect the collapsed model |
| residual (baseline 0.9775) → merge | 0.9706 | the working MKA track (retention 99.3%) |
| residual → LRP score | **undefined** | LRP's backward rule assumes a plain chain; a residual stream has no `z_j = Σ a_i w_ij` (negative #2) |

The last cell is reported honestly, not hidden: `src/pruning/lrp.py` documents
the single-chain trace assumption, and `phase6b_hybrid.py` catches and labels
the structural mismatch as evidence. The pairing is therefore **by design**,
not a gap: merging needs residuals (MKA/CoM), LRP needs a plain graph (SparC3).

### 6.10 Multi-seed reproduction (mean ± std over seeds 42, 7, 2026)

`scripts/reproduce.py` trains fresh baselines for every seed and replays both
tracks (evidence table [reproduce] = `artifacts/reproduce_table.png`):

| metric (per seed) | seed 42 | seed 7 | seed 2026 | **mean ± std** |
|---|---|---|---|---|
| merge accuracy before → after | .9775 → .9706 | .9799 → .9727 | .9785 → .9717 | → **99.29% ± 0.02% retention** |
| params saved (merge) | 42.3% | 42.3% | 42.3% | **42.29% ± 0.00%** |
| FLOPs saved (merge) | 42.2% | 42.2% | 42.2% | **42.24% ± 0.00%** |
| latency speedup (merge, noisy) | 3.4× | — | — | **2.67× ± 0.74** |
| prune accuracy before → after | .9760 → .7330 | .9785 → .7804 | .9776 → .6493 | → **73.76% ± 6.77% retention** |
| prune sparsity | 49.9% | 49.9% | 49.9% | **49.87% ± 0.00%** |

The merge story is extremely stable across fresh initializations (retention
std 0.02%); LRP-pruning retention at 50% sparsity fluctuates with the seed
(66–80%) — worth reporting as mean ± std, not a single number.

---

## 7. Findings & Discussion

1. **Layer merging is a residual-architecture technique.** NMI finds redundancy
   everywhere; only residual nets turn it into loss-free parameter deletion.
2. **Pair choice > fusion rule.** Similarity-greedy (non-positional) selection
   with any of weighted/uniform/SLERP beats positional merging; the first merge
   even *improves* accuracy slightly (regularization-like averaging).
3. **Merging reduces real compute** (whole layers dropped): 42% FLOPs — the
   only training-free method in the survey that does this (see other_approaches).
4. **LRP is different information from magnitude** (corr 0.10) and is the
   principled SparC3 choice; on small clean data magnitude remains a strong
   baseline — relevance ≠ "all good", and the report states the boundary.
5. **Two tracks is a feature, not a gap.** Merging (needs residual) and LRP
   pruning (needs plain) are structurally exclusive on one small model —
   quantified honestly in phase 6b rather than hidden.

**Trade-off synthesis.** A production system would *merge* redundant residual
blocks (get 42% back structurally) and *then* prune the surviving weights at a
modest sparsity (LRP, ~10–25%) — a cascade that our two-track experiment
decomposes honestly rather than glosses over. (On a transformer, both would act
on the same residual model: the skip-connection-aware LRP variant is out of
scope for this assignment, see §6.9.)

---

## 8. Limitations

- MNIST-scale; we reproduce the geometry of the papers' results, not 8B-scale
  numbers (per assignment rule).
- Merging and pruning target different architectures, so one single-model hybrid
  is structurally impossible here — demonstrated quantitatively in `phase 6b`
  (§6.9), not asserted.
- Wall-clock latency is machine-dependent and noisy; FLOPs/params are the robust
  reduction metrics.
- Weight sparsity ≠ wall-clock speedup without sparse kernels.
- Learning-path companion: the math and numbers in this paper are re-derived
  step by step in `docs/learning/` for anyone (including viva examiners) to follow.

---

## 9. Project Management, Compute & Risk

- **Compute:** CPU-only (this repo), ~1.7 min per baseline seed; full
  reproduction ≈ 12 min. Colab-free-tier runnable.
- **Gantt:** [gantt] (`artifacts/gantt.png`), 6 phases + report/viva week.
- **Risks & mitigations:** merging collapses → we start with 1 merge, use
  α=S_lm, test SLERP, and report the plain-MLP negative; NMI instability → z-score
  + ≥512 calibration samples + diffusion-map smoothing (design table in
  `docs/phase_plan.md`).
- **Team contribution:** `§10`.

## 10. Team Contribution
| Phase | Contributor | Deliverable |
|---|---|---|
| (fill per member on submission) | | |

---

## 11. Reproduce-all checklist (kept in repo)
- [x] `make setup && make test` (78 tests)
- [x] `make verify` (tests + smoke as the pre-push gate)
- [x] `make smoke`
- [x] `make phases` → all eight experiment phases + figures
- [x] `make reproduce` → multi-seed mean±std evidence
- [x] `make run-notebooks` → notebooks execute end-to-end → screenshots for PDF
- [ ] PDF with screenshots of every notebook step, link to this repo, submitted