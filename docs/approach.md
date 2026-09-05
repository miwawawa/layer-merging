# Approach & Techniques — How We Reproduce Layer Merging

This document is the *methodology* half of the report: it explains, step by
step, the maths and algorithms behind the three reproduced techniques, why each
works, how it is implemented in this repo, and what parameters control it. Read
`paper_comparison.md` for the papers' context and `other_approaches.md` for the
alternatives we surveyed.

---

## 0. Problem statement

Modern MLPs are **over-parameterized**: many hidden layers compute nearly the
same function (redundancy), and many individual weights carry little
information. We study two *training-free, post-hoc* ways to shrink a trained
model with minimal accuracy loss:

1. **Layer merging** — fuse entire redundant layers into one, deleting
   parameters and FLOPs at once (MKA, CoM);
2. **Attribution-guided pruning** — zero out the least-relevant individual
   weights (SparC3).

Both are *compression without retraining*: the model is used exactly once to
decide what to remove, then compressed. The headline question is the
**reduction vs accuracy** trade-off, and — for merging — *under which
architectural conditions the operation is even safe*.

---

## 1. Technique 1 — Measuring layer redundancy (MKA §3.2–3.3)

Before merging anything we must decide *which* layers are redundant. MKA does
this in three steps.

### 1.1 Activations

For a calibration set `𝒟` of `n` samples we run the model and keep, for each
hidden layer `l`, its output activations

```
H_l = [a_l(x_1), ..., a_l(x_n)]^T   ∈  ℝ^{n × d_l}
```

(We use post-non-linearity activations — what the next layer actually receives.)
Implementation: `utils/activations.py::extract_activations`.

### 1.2 Diffusion map (manifold learning)

The raw coordinates are redundant; MKA instead embeds each `H_l` into a
low-dimensional manifold whose geometry captures *functional* similarity.

1. Gaussian affinity over samples:
   `W_ij = exp(-‖H_i − H_j‖² / σ²)`, with zero diagonal.
2. Degree matrix `D_ii = Σ_j W_ij` and the *symmetric diffusion operator*
   `P̃ = D^(−1/2) W D^(−1/2)`.
3. Eigen-decomposition `P̃ = Φ Λ Φᵀ`; discard the trivial `λ₁ = 1` eigenvector.
4. The diffusion map is
   `Φ_t(H_l) = (λ_2ᵗ φ_2, …, λ_{k+1}ᵗ φ_{k+1})`, i.e. each sample gets `k`
   geometric coordinates.

Intuition: two layers whose *point-cloud geometry* is similar will have similar
`Φ_t` coordinates, which is much more meaningful than comparing raw Euclidean
positions. Implementation: `similarity/diffusion.py::DiffusionMap`.

### 1.3 Gaussian-NMI similarity

For two embedded layers `Ψ_l = Φ_t(H_l)` and `Ψ_m = Φ_t(H_m)`, both viewed as
`k`-dim Gaussian random vectors, mutual information has a closed form:

```
I(Ψ_l; Ψ_m) = ½ · ln( |Σ_l| · |Σ_m| / |Σ_lm| )
H(Ψ)        = ½ · ln( (2πe)^k · |Σ| )           (differential entropy)
NMI_lm      = I(Ψ_l; Ψ_m) / √( H(Ψ_l) H(Ψ_m) )   ∈ [0, 1]
```

`NMI = 1` means the layers carry identical information (maximally redundant);
`0` means independent. Features are z-scored first so the covariance
determinants stay well-conditioned (`similarity/gaussian_mi.py`). We also keep
a plain **cosine** similarity between mean activations as a cheap baseline
(`similarity/cosine.py`).

**Result.** The similarity matrix `S` (`S_lm = NMI_lm`) is the redundancy map —
the higher an off-diagonal entry, the more the two layers can share a single
successor.

---

## 2. Technique 2 — Merging layers (MKA Alg. 1 + CoM)

### 2.1 Pair selection

We never assume a *positional* pairing (`layer_1↔layer_2`, `layer_2↔layer_3`,
…), CoM's central lesson: merges must be **non-positional**. Instead we pick the
globally most-similar *compatible* pair:

```
(i, j) = argmax_{i<j, shape(W_i)=shape(W_j)} S_ij
```

compatible = identical weight shape (the transformer analogy: all layers share
`d_model`, so all pairs are compatible there).

Repeat greedily: merge, re-measure, merge again — until no pair beats the
similarity threshold `τ` (`Config.merge_threshold`). Each merge removes one
layer, so a 4-hidden-layer net needs 3 merges to collapse to 1 hidden layer.

Implementation: `merging/merger.py::LayerMerger` (the Facade), Algorithm 1.

### 2.2 Fusion rules (the strategies)

Given the winning pair `(l, m)` with similarity `S_lm`, the merged parameters
are a combination of both:

| Strategy | Rule | Source | When it wins |
|---|---|---|---|
| **Weighted (MKA)** | `θ̃ = α·θ_l + (1−α)·θ_m`, `α = clamp(S_lm, α_min, α_max)` | Paper 1 | Trusts the more redundant/stronger layer proportionally to similarity |
| **Uniform** | `θ̃ = 0.5·θ_l + 0.5·θ_m` | model soups | Baseline; works when layers are near-identical |
| **SLERP (CoM)** | spherical interpolation along the geodesic between `θ_l`, `θ_m` (below) | Paper 2 | Keeps weight *magnitudes* natural — good when vectors are far apart |

**SLERP formula.** For flattened `a = θ_l`, `b = θ_m`:

```
Ω  = arccos( ⟨a,b⟩ / (‖a‖‖b‖) )
θ̃ = sin((1−α)Ω)/sin Ω · a  +  sin(αΩ)/sin Ω · b
```

When `sin Ω ≈ 0` (parallel vectors) SLERP degenerates to the plain average —
the linear fallback that keeps it stable. The interpolation is applied
**per-tensor** (each `weight`/`bias`) — never globally flattened, per CoM.
Implementation: `merging/{weighted,uniform,slerp}.py` + `merging/factory.py`.

**The old strategy in the new model.** After fusion the two layers are replaced
by one `nn.Linear` with the fused weights, inserted back at `min(i, j)`, and
the next layer consumes the (merged) hidden activation — no affine mismatch
because every candidate layer shares `(in, out) = (d_model, d_model)`.

### 2.3 When is merging safe? (our key architectural finding)

Merging replaces *two reversible paths* with one. In a **plain feed-forward**
MLP, `layer_m` consumes `layer_l`'s output directly; averaging the two weights
either double-counts or cancels the transformation — accuracy collapses
(0.976 → 0.123 after ONE merge in our measurements). In a **residual** MLP,
`h ← h + F_l(h)`, each layer is a *perturbation on a stream*; fusing two similar
blocks barely moves the stream, so `residual 0.9775 → 0.9706` after the full
collapse (99.3% retention). This mirrors exactly why MKA/CoM work on
transformers (residual blocks + shared `d_model`) and why they would fail on
history-free deep straight-line networks.

---

## 3. Technique 3 — Attribution-guided pruning (SparC3)

### 3.1 LRP (Layer-wise Relevance Propagation)

For each sample we propagate the class prediction *backwards* to weight space.
Given layer output `z_j = Σ_i a_i w_ij` (plus bias), the ε-stabilized LRP rule
distributes relevance of neuron `j` to its inputs:

```
R_{i←j} = ( a_i · w_ij / (z_j + ε·sign(z_j)) ) · R_j
```

with `ε = lrp_eps` guarding the denominator (`pruning/lrp.py`). Every weight
`w_ij` accumulates relevance `R_ij = Σ_x R_{i←j}(x)` over the reference
samples — a *per-parameter attribution* of the task to each connection.

### 3.2 Importance → global pruning

SparC3 aggregates over references and ranks **all weights together**:

```
R̄_ψk = 1/n · Σ_x R_ψk(x)          (mean attribution per weight)
keep top-q:  mask the (1−q) lowest-attribution weights model-wide
```

This is *global* top-k masking (`pruning/pruner.py::prune_model_global`) —
contrast with per-layer pruning (each layer truncated independently), which
turns out worse because a layer's head weights get killed without a global
budget perspective. Zeroed weights are simply multiplied by their mask; forward
pass is untouched (dense ops). Baselines: **magnitude** (`|w|`) and **random**.

**Why LRP specifically:** LRP respects the *sign of the contribution* through
the `a_i w_ij` products, so it can keep a small-magnitude but consistently
task-positive weight, whereas magnitude pruning only looks at size. Our
measurement shows LRP attributions are nearly *decorrelated* with magnitudes
(`corr ≈ 0.10`) — i.e., LRP is genuinely different information, not a relabeling.

**Honest caveat:** weight-zeroing does not reduce FLOPs on dense hardware unless
sparse kernels exist; its win is *memory and theoretical compute*, measured here
as sparsity, not wall-clock.

---

## 4. Metrics & calculations (how we quantify "improvement")

Every phase reports a full snapshot (`evaluation/metrics.py::summarize`):

| Metric | Definition | Meaning |
|---|---|---|
| `acc` | correct / total on the test set | quality of the compressed model |
| `acc_drop` | `acc_after − acc_before` | how much quality was traded away |
| `retention` | `acc_after ÷ acc_before` | % of accuracy kept |
| `params` | Σ over all parameters | memory dominated metric |
| `flops` | Σ_Linear `2·in·out + 1` | compute per forward sample |
| `size_mb` | `params × 4 B` (float32) | on-disk/ram footprint |
| `latency_ms` | median forward time over batches | operational speed (noisy, use ratios) |
| `sparsity` | 1 − (nonzero ÷ total params) | pruning depth |
| `params_savings` | 1 − `params_after/params_before` | the *reduction* headline |
| `speedup` | `latency_before ÷ latency_after` | wall-clock factor |

Savings are computed **relative to each track's own baseline**, so the four-row
evidence table (`artifacts/results_table.png`) tells the honest before/after
story. Single-seed numbers are complemented by a multi-seed reproduction
(`scripts/reproduce.py`, `Config.seeds`) that reports mean ± std.

---

## 5. Configuration (every knob, one place)

Everything lives in `Config` (`src/config/default.py`) and is overridable via
`.env` / `LAYER_MERGING_*` environment variables (Python literals).

| Group | Fields |
|---|---|
| Model | `hidden_dims`, `num_classes`, `input_dim`, `dropout`, `model_arch`, `branch_scale` |
| Data | `data_dir`, `mirror`, `batch_size`, `val_size` |
| Training | `seed`, `seeds`, `epochs`, `lr`, `optimizer`, `loss` |
| Similarity | `n_manifold_dims`, `diffusion_time`, `kernel_sigma`, `n_calibration_samples` |
| Merging | `merge_min_alpha`, `merge_max_alpha`, `merge_threshold`, `merge_strategies` |
| Pruning | `lrp_eps`, `prune_keep_ratio`, `prune_sweep`, `prune_methods`, `random_seed` |
| Paths | `data_dir`, `artifacts_dir`, `files`, `gantt_*` |

---

## 6. Why these choices (not the alternatives)

| We could have used… | We use instead | Because |
|---|---|---|
| SVD / low-rank per layer | whole-layer fusion | MKA/CoM are whole-layer techniques; we reproduce *their* geometry |
| Fisher/norm-weighted merging | α = S_lm | α = S is exactly paper-1; Fisher needs gradients over data |
| Magnitude-only pruning | LRP ranking | SparC3's method IS LRP; magnitude is our baseline |
| Per-layer pruning masks | global top-k | SparC3 masks globally; per-layer is measurably worse |
| Pretrained LLMs | MNIST MLP | assignment forbids closed-source/big-LLM reproduction |

See `other_approaches.md` for the full survey.

---

## 7. Documented limitations (honesty for the report)

1. **Merging ≠ pruning**: they live on different models here (merging needs
   residual structure; plain graph makes LRP exact). A single hybrid
   "merge+prune" is not reproducible in one small MLP.
2. **MNIST scale**: we reproduce the *geometry* of the results (curves,
   matrices, trade-offs), not the 8B-LLM numbers.
3. **Latency is noisy**: wall-clock medians jump between runs; FLOPs/params are
   the robust reduction numbers.
4. **Sparsity ≠ speed** on dense hardware without sparse-kernel support.
5. **α = S heuristic**: the paper's own mixing rule; we clamp it to [0.1, 0.9]
   to avoid degenerate single-layer copies.