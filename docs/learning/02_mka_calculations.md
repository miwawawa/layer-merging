# MKA layer merging — step-by-step calculations

> Full numerical walk-through of Paper 1 (Liu et al., EMNLP 2024,
> arXiv:2406.16330) on **our** MNIST MLP, from activations to fused weights,
> with every intermediate number the repo actually produces.
>
> Prereqs: `docs/learning/01_math_primer.md` §3–7. Code: `src/similarity/*`,
> `src/merging/*`, `src/experiments/phase2_similarity.py`,
> `src/experiments/phase3_merging.py`.

---

## Step 0 — the model we merge

Residual MLP (`src/models/residual_mlp.py`), residual block
`h ← h + act(W·h)`, `d_model = 256`, 4 hidden blocks, 10 classes.

```python
params_first_affine = 784*256 + 256          # = 200,960   (the projection, not merged)
params_hidden_block  = 256*256 + 256          # = 65,792    (x1 per hidden block)
params_head          = 256*10  + 10           # = 2,570
total_residual       = 200,960 + 4*65,792 + 2,570 + 65,792   # +branch proj used by skip
                     = 466,698                # repo: artifacts baseline_residual.pt
```

After removing the three redundant 256→256 maps (`3 × 65,792 = 197,376`):

```python
merged_total = 466,698 - 197,376             # = 269,322  (repo artifact: merged_model.pt)
params_savings = 197,376 / 466,698           # = 0.42294  → 42.3% (reproduce mean±std)
```

## Step 1 — activations (what we measure similarity on)

Feed `n = 512` validation images through every hidden block (the
`n_calibration_samples` in `Config`). For block `k` collect the batch of
activations `H^k ∈ ℝ^{512×256}`, one column per hidden unit.

> scale check: 512 samples × 256 units = 131,072 floats per layer ≈ 1 MB.

## Step 2 — diffusion embeddings (robust activation geometry)

Paper's diffusion-map construction (Sec 3.4), on our sample matrix `H^k`:

1. **Kernel** (all pairs of samples), bandwidth `σ = σ_K * median distance`:

   ```
   W_ij = exp( -‖H_i^k - H_j^k‖² / σ² )        σ = 3.0 in Config
   ```

2. **Anisotropic diffusion operator** `P` (row-normalized): a random walk on
   the sample graph; `P^t` (t=1 step) spreads probability along likely
   transitions and damps noise.

3. **Embedding:** `Ψ^k` = top `k_eig = 8` eigenvectors scaled by their
   eigenvalues (`n_manifold_dims`). Result: `Ψ^k ∈ ℝ^{512×8}` — a
   8-dim smooth representation of the 256-dim activations.

> Why 8 dims? The paper needs *robust* geometry, not full reconstruction;
> 8 eigenvalues capture the diffusion structure (largest eigenvalue of P = 1,
> carrying the connectedness signal).

## Step 3 — Gaussian NMI between two layers (the similarity S)

For layers `l, m` with diffusion embeddings `Ψ^l, Ψ^m ∈ ℝ^{n×8}`:

1. **Standardize** each dimension (z-score) — `gaussian_mi._standardize`.
2. **Covariances:** `Σ_l, Σ_m ∈ ℝ^{8×8}`, joint `Σ_{lm} ∈ ℝ^{16×16}`, each
   regularized `+ εI`, `ε = 1e-6`.
3. **Entropies & MI** (closed form, joint Gaussian):

   ```
   H(Ψ) = (8/2)·ln(2πe) + (1/2)·ln|Σ|
   I = (1/2)·( ln|Σ_l| + ln|Σ_m| − ln|Σ_{lm}| )
   ```

4. **Normalize:**

   ```
   S_lm = I / sqrt( H(Ψ^l)·H(Ψ^m) )      ∈ [0,1]
   ```

Repo numbers (seed 42, `artifacts/phase2_similarity.json` / `similarity_all.png`):

```
residual:  mean off-diag  S_nmi ≈ 0.512     S_cos ≈ 0.645
plain:     mean off-diag  S_nmi ≈ 0.525     S_cos ≈ 0.171
```

**Reading:** residual activations are mutually redundant on *both* views;
plain layers are NMI-similar but geometrically *orthogonal* (cosine 0.171) —
they carry different functions. This divergence is exactly why NMI alone does
not justify merging: the residual *architecture* is what lets fusion succeed.

## Step 4 — choosing the pair (greedy, per CoM too)

`LayerMerger._best_pair`: over all compatible pairs `(i,j)`
(`shape(W_i)=shape(W_j)`), pick the largest `S_ij` above `threshold = 0.05`.

First merge (seed 42): pair `(1,2)`, `S = 0.988`.

## Step 5 — the mixing coefficient α

MKA's per-pair weight = the similarity itself, clamped:

```
α = clip(S_lm, 0.1, 0.9)
```

`S=0.988 → α = 0.9`.
(Phase-4b ablated α ∈ {0.1,0.3,0.5,0.7,0.9} on this exact pair — all reach
0.9786±0.0005: *the pair choice matters more than the exact α*.)

## Step 6 — fusion

```
θ_c = α·θ_l + (1−α)·θ_m        (weights AND biases, elementwise)
```

One fused 256→256 map replaces `l` and `m`: `466,698 → 400,906` params.

## Step 7 — repeat until the threshold rejects everything

Greedy loop over the *reduced* graph: never fuse the projection or the head
(shape-incompatible with hidden blocks), only the 3 remaining hidden maps.
Result (seed 42): 3 merges → `269,322` params, accuracy `0.9775 → 0.9706`
(retention `0.993`). FLOPs: `Σ 2·W_in·W_out` drops two of four hidden maps:
savings `42.3%`; measured single-batch latency `→ ~2.7×` speedup.

## Why it works / when it fails (the paper's information-argument)

MKA frames merging via the Information Bottleneck: compression should keep
`I(T;Y)` while shrinking `I(X;T)`.
`L_IB = I(X;T) − β·I(T;Y)`.
Layers that carry *redundant* information `I(Ψ^l;Ψ^m)` about the labels can be
combined without losing `I(T;Y)`. Plain (non-residual) layers are **not**
redundant — after one merge accuracy collapses `0.9760 → 0.1229` (repo:
`merge_curve.png`, plain curve) — so the residual stream is the *structural*
ingredient that makes the information-argument hold.

## Paper headline vs our implementation

- Paper claims **43.75%** parameter compression at −2.82% MMLU on Llama-3-8B;
  our toy: **42.3%** at −0.7% MNIST accuracy. The *mechanism* transfers; the
  scale differs (paper merges on a transformer with LayerNorm + residual;
  we use the minimal residual MLP that exhibits the same phenomenon).
- Paper's Sec-2 "similarity" is the **NPIB** metric; the formal appendix and
  reference implementation evaluate **Gaussian NMI**, which is what our
  `src/similarity/gaussian_mi.py` implements (equivalent score, cheaper).