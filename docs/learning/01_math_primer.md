# Math primer: from first principles to the three papers

> Everything a reader needs to re-derive the mathematical content of the MKA,
> CoM and SparC3 papers, in one place. Each section ends with the *repo
> numbers* it explains. No prior ML background assumed beyond arithmetic and
> basic functions.

---

## 1. Vectors, matrices, and the linear layer

**Vector.** Ordered list of `d` numbers, e.g. one MNIST image flattened:

```text
x = [pixel_0, pixel_1, ..., pixel_783]        # 784 numbers in [0, 255]
```

Arithmetic is elementwise: `(x + y)_i = x_i + y_i`, `(a·x)_i = a·x_i`. The
vector **norm** `||x|| = sqrt(Σ_i x_i²)` measures length.

**Matrix.** A rectangular grid `A` with `out` rows and `in` columns. **Matrix
times vector** — the heart of every neural layer:

```text
(W·x)_j = Σ_i W[j,i]·x[i]                       # j-th output = dot(row j, x)
```

**Linear layer** = matrix multiplication + bias:

```text
y = W·x + b        W: (out, in), x: (in,), y, b: (out,)
```

**Our repo model (residual MLP).** 784 → 256 → 256 → 256 → 256 → 10 plus a
projection layer. Parameter count for one layer `W:(out,in)` is
`out·in + out` (bias). For the first affine map 784→256:
`784·256 + 256 = 200,704 + 256 = 200,960`.

Two concrete counts used everywhere in this repo:

```text
plain MLP:      784·256+256  +  3·(256·256+256)  +  256·10+10  =  400,906
residual MLP:   same        +  256·256+256 (branch projection hidden) + proj  =  466,698
```

(The *redundant* part is the 3 hidden→hidden maps, each 65,792 params — the
part layer merging fuses away.)

## 2. Activations, ReLU, and why "layers are functions"

Given an input batch `X` (n×784), layer `k` produces **activations**
`H_k = act(W_k·H_{k-1} + b_k)` (`act = ReLU`, `max(0, ·)`, applied elementwise).
`H_k` is a `(n, 256)` matrix: one 256-dim *representation* of every input.

A layer is a *function*: it maps an input vector to an output vector. Two
layers with the same input/output shape are **compatible** — they can be
merged because their weights occupy the same "slot" of the computation graph.

## 3. Probability: what a distribution is

A random vector `X` assigns probabilities to outcomes. The two facts we need:

**(a) Expectation** `E[X]` = average over the distribution (sample mean of a
batch approximates it).

**(b) Variance/covariance** measure spread and co-movement:

```text
Var(X_k)    = E[(X_k - E[X_k])²]
Cov(X_i,X_j)= E[(X_i - E[X_i])(X_j - E[X_j])]
```

**Covariance matrix** Σ: entry `(i,j)` = `Cov(X_i, X_j)`. Symmetric,
positive semi-definite; the diagonal is the variances.

**Correlation** normalizes covariance to `[-1,1]`:
`ρ_ij = Cov(X_i,X_j)/(sd(X_i)·sd(X_j))`.

*Repo use:* `_standardize` in `src/similarity/gaussian_mi.py` z-scores each
activation dimension before estimating Σ (keeps determinants well-conditioned).

## 4. Entropy: "how much surprise a distribution holds"

Shannon entropy (bits-optional, we use nats) quantifies uncertainty:

```text
H(P) = -Σ_x p(x)·ln p(x)
```

Uniform over 10 classes: `H = -10·(1/10)ln(1/10) = ln(10) ≈ 2.30`.
A deterministic output has `H = 0` (no surprise).

For continuous random vectors, **differential entropy**. Gaussian with
covariance Σ in `d` dims:

```text
H(N(μ, Σ)) = (d/2)·ln(2πe) + (1/2)·ln|Σ|
```

`|Σ|` = determinant. *Repo use:* the pruning paper's LRP and the merging
papers' similarity both live on entropy/divergence quantities.

## 5. Joint Gaussian, mutual information — the similarity engine

If two random vectors `Ψ^l, Ψ^m` are **jointly Gaussian**, their mutual
information has closed form using covariance determinants:

```text
I(Ψ^l; Ψ^m) = (1/2)·ln( |Σ_l|·|Σ_m| / |Σ_{l,m}| )
```

where `Σ_{l,m}` is the joint covariance of the concatenated `(2d)`-vector.
**Worked scalar case** (used in the unit test on `gaussian_mi`): two
standardized 1-D Gaussians with correlation ρ. Then `|Σ_l|=|Σ_m|=1` and
`|Σ_{l,m}| = 1 - ρ²`, so:

```text
I = - (1/2)·ln(1 - ρ²)
ρ=0.8 -> I = -0.5·ln(0.36) = 0.5000·1.02165 ≈ 0.5108 nats
```

**Normalized mutual information** (paper-1's similarity) scales to `[0,1]`:

```text
S_lm = I(Ψ^l; Ψ^m) / sqrt( H(Ψ^l)·H(Ψ^m) )
```

*Interpretation:* `S_lm = 1` means layer `l` is (information-theoretically)
identical to layer `m`; `0` means unrelated. This is the quantity `phase2`
computes and `phase3` uses to pick fuse-pairs.

*Repo numbers:* residual off-diagonal NMI means ≈ 0.51 (layers overlap a lot —
safe to merge); plain off-diagonal NMI ≈ 0.53 but its *cosine* ~0.17 — the two
views diverge, and merging plain layers still collapses. High NMI similarity
alone is **not sufficient**; residual structure is what makes fusion lossless.

## 6. Eigenvalues, eigenvectors, and diffusion maps

**Eigenpair of a matrix A:** vector `v` and scalar λ such that `A·v = λ·v`
(multiplying by the matrix only stretches `v`). The top eigenvectors form a
low-dimensional summary of the geometry that `A` encodes.

**Gaussian kernel / diffusion.** Given `n` activation samples, build a
similarity graph where nodes = samples and

```text
W_ij = exp( -||H_i - H_j||² / σ² )
```

normalize rows (`P = D^{-1/2} W D^{-1/2}`), take the top `k` eigenvectors as
new coordinates. This is a **diffusion map**: nearby/likely samples (under the
data geometry) land near each other in the new `k`-dim space, while spurious
isolation gaps are smoothed. Paper-1 feeds these *diffusion embeddings*
`Ψ^l, Ψ^m` into the Gaussian-NMI above — measuring layer redundancy in a
geometry that is robust to noise.

*Repo numbers:* `n_manifold_dims = k = 8`, `kernel_sigma σ = 3.0`,
`diffusion_time = 1` (one step of the Markov chain).

## 7. Convex combinations and merge strategies

**Weighted average**: `y = α·a + (1-α)·b` with `α ∈ [0,1]` lies *between* `a`
and `b`. Merging layer weights of pair `(l,m)` with coefficient `α = S_lm`:

```text
θ_c = α·θ_l + (1-α)·θ_m          (MKA weighted merge)
```

If `S_lm ≈ 1` the layers are almost identical → picking any `α` is safe
(`θ_c ≈ θ_l ≈ θ_m`). Hence the insight: *merge only similar layers; the
similarity doubles as the mixing coefficient.*

**α-clamping** (`src/merging/weighted.py`): `α ↦ clip(α, 0.1, 0.9)` so neither
layer's weights vanish.

**Worked example (phase 3/4 numbers):** pair `(1,2)`, `S = 0.988`, so
`α = clip(0.988)=0.9`. The merged layer is `0.9·θ_1 + 0.1·θ_2`. One merge cuts
one 256→256 map (65,792 params): `466,698 → 400,906`.

## 8. Spherical interpolation (SLERP) — paper 2's mixer

SLERP interpolates along the **geodesic on the unit sphere** instead of the
chord (straight line). Angle between weight vectors:

```text
Ω = arccos( 〈w1, w2〉 / (‖w1‖·‖w2‖) )
```

then

```text
w_c = sin((1-α)Ω)/sin(Ω) · w1  +  sin(αΩ)/sin(Ω) · w2
```

**Key property:** `‖w_c‖` stays on the sphere — magnitudes are preserved, which
matters for learned networks (straight-line averaging shrinks norms).
**Degeneracy:** if `w1 ∥ w2` (or antiparallel), `sin Ω ≈ 0` and the formula
blows up; we fall back to the plain average (`sin Ω < 1e-5`).

*Repo numbers (phase 4):* for one identical pair, weighted/uniform/SLERP all
reach ≈ 0.9786±0.0005 — they differ in *how* they mix, not whether they can.

## 9. What "compression" means — the metrics

Given a model `M`, compare against baseline `M0`:

| Metric             | Definition                                     | Repo numbers (phase 6, seed 42)         |
|--------------------|------------------------------------------------|-----------------------------------------|
| parameter count    | `Σ all weights`                                | residual `466,698 → 269,322` (merge)    |
| params savings     | `1 - params_after/params_before`               | `1 - 269322/466698 = 0.4229` (−42.3%)   |
| FLOPs              | `Σ  2·in·out` per linear layer (+1 per unit)   | −42.2% after full merge                 |
| footprint          | `params × 4 bytes` (fp32)                      | `1.79 MB → 1.03 MB`                     |
| latency            | ms per batch on CPU                            | `~2.7×` speedup after merge             |
| accuracy retention | `acc_after / acc_before`                       | merge `0.993 ± 0.000`; LRP-prune `0.74` |
| sparsity           | `1 - nonzero/total` (weights only for `weight_sparsity`) | prune @ 50% keep → 0.50 |

**Reasoning link:** merging *removes* parameters and FLOPs (fewer layers), so
savings AND speedup move together. LRP pruning *zeroes* parameters, so FLOPs
and latency do NOT drop without sparse kernels — an honest asymmetry we report.

## 10. What "importance" means for pruning — gradients to LRP

SparC3 needs a per-weight **relevance score** `R_{i←j}` telling "how much the
*j-th* output unit's verdict depended on weight `(i,j)`". LRP back-propagates
relevance from the output decision through each linear layer:

```text
z_ij = a_i·W_ij          contribution of input i to unit j (pre-activation)
z_j  = Σ_i z_ij + b_j    total pre-activation (the denominator)
R_{i←j} = (z_ij / z_j)·R_j     share of output relevance assigned to weight (i,j)
R_i     = Σ_j R_{i←j}          relevance of input i = sum over outgoing edges
```

The rule redistributes the final relevance (one-hot on the predicted class,
scaled by its softmax probability) proportionally to each edge's contribution.
`eps`-stabilization avoids `/0`: `z_j -> z_j + sign(z_j)·eps·mean(|z_j|)`.

**Why it needs a plain graph (phase 6b's honest finding):** the formula
assumes every unit's pre-activation is a *sum of weighted incoming edges*.
A residual block `h ← h + act(W·h)` has a `+ h` term that is not a weighted
edge — the relevance share `z_ij/z_j` is no longer defined. Hence LRP is exact
only on the plain MLP, and merges are safe only on the residual MLP: the two
techniques must live in two separate pipelines (`phase6`).

**Worked mini-example (phase 5 numbers):** at 50% keep, LRP keeps
`0.7376 ± 0.0677` of accuracy, magnitude keeps `0.9730` on the plain MLP —
attribution-guided pruning *hurts* here relative to magnitude because LRP
targets class-discriminative weights, which are exactly the ones the MNIST-T
copy needs. We report both, and explain *why* magnitude wins on this toy.

---

## Suggested order to teach yourself

1. §1–2 (what a layer is, parameters) → read `src/models/`.
2. §3–5 (probability → MI) → read `src/similarity/`.
3. §6 (diffusion) → re-read `src/similarity/diffusion.py`.
4. §7–8 (mixing) → read `src/merging/`.
5. §9 (metrics) → read `src/evaluation/`.
6. §10 (LRP) → read `src/pruning/`.

Then open `docs/learning/02_mka_calculations.md` and follow the numbers.