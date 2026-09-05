# Related work — calculation notes for the report's related-work section

> Short, formula-level notes on the approaches we compare against conceptually,
> so the paper's related-work claims are each *one equation deep*. Not
> reproductions — context. Prereqs: math primer.

---

## 1. Task arithmetic / model addition

Fuse experts by moving in *task-vector* direction:

```
θ_merged = θ_base + λ · Σ_t (θ_t − θ_base)
```

- λ = 1 → naive averaging; λ = −1 → *forgetting*.
- Why it fails for layer-level merging: one global λ, positional pairing only,
  no compatibility check. (CoM's motivation.)
- Where we'd see it in our code: uniform strategy `α=0.5` fusion is task
  arithmetic with a single expert (`θc = 0.5θl + 0.5θm`).

## 2. TIES-merging (electrically "elect" + "disjoint" merge)

1. Trim each task-vector: keep top `k%` of entries by magnitude
   `τ_t = top_k(θ_t − θ_base, k)`.
2. Resolve sign conflicts by majority vote per parameter (sign mask).
3. Average the masked vectors of the surviving sign.

Equation-level:
```
v_s = sign(Σ_t sign(τ_t))            # voted sign
θ_merged = θ_base + 1/T Σ_t τ_t · 𝟙{sign(τ_t)=v_s}
```

Concept: requires *sparse, disjoint* expert updates. Our merged layers are
*dense, near-identical* — TIES has nothing to trim. It targets task-vector
merging, not redundancy compression.

## 3. RegMean (regression-based linearity)

For linear layers, minimizes an MSE objective with a whitening matrix computed
from feature correlations `X`:

```
W_m = (1/2)(W_1 + W_2) + (1/2)(W_1 − W_2) (X₁^T X₁ − X₂^T X₂)(X₁^T X₁ + X₂^T X₂)^{-1}
```

Data-dependent correction on top of the average; needs per-model calibration
features. Our weighted merge is its λ=constant special case with no decorrelation.

## 4. SLERP vs naive averaging (numbers behind "norms matter")

For unit vectors at angle Ω, α=0.5:

```
‖averaged‖ = ‖ (θ1+θ2)/2 ‖ = cos(Ω/2)      → 0.7071 at Ω=90°
‖slerped‖  = 1                             (stays on the sphere)
```

With a branch-gain 0.1 (our `branch_scale`), hidden weights are small; the
norm-preserving property becomes visible only at larger magnitudes — hence
our honest "phase-4 mixers tie" observation on the toy, and why CoM's SLERP
advantage needs pretrained-scale weights.

## 5. Wanda — SparC3's "non-attribution" strawman

`S_ij = |W_ij| · ‖X_j‖`  (magnitude × input-channel norm; no gradient, no LRP).

- Close to our `MagnitudeImportance` (we drop the `‖X_j‖` factor — acceptable
  for the comparison, documented).
- SparC3's positioning: Wanda can compress but *cannot attribute* (no
  per-prediction relevance), so it cannot discover circuits or fix
  behaviors — the paper's core value-add.

## 6. SparseGPT / structured sparsity (what "kernel support" means)

Magnitude/LRP unstructured pruning zeroes weights but keeps the dense compute
graph: FLOPs and latency unchanged (our phase-6 finding: merging removes
FLOPs, pruning only zeroes). Structured/row-wise pruning (a SparC3 option and
our `prune_mask(row_wise=True))` removes whole output rows so sparse kernels
can skip them — the FLOPs savings materialize only then. We documented both
in `docs/learning/01_math_primer.md` §9.

## 7. Buzzega et al., "Rethinking Layer-wise Model Merging through Chain of
Merges" (arXiv:2508.21421)

A *second* "CoM" — training-free chaining of pairwise merges of LLM experts,
estimating stability of intermediate closures. Distinct from the CVPR 2026
VLM CoM we implement. We cite it as representative of the "merge is a
*process*, not one shot" family that our greedy loop also embodies
(`merger.merge_all`, merging until no pair clears threshold).

## 8. Why our three-paper stack is coherent

- MKA gives the *metric* (Gaussian NMI over diffusion activations → which
  pairs are redundant) + the *residual architecture* requirement.
- CoM gives the *pairing discipline* (search, not position) + *SLERP* as a
  norm-preserving mixer (phase 4 ties together the strategies).
- SparC3 gives the *orthogonal compression* (LRP attribution + rank/mask)
  whose structural constraint (plain graph) is exactly MKA's complement —
  phase 6's two-track pipeline, and phase 6b's honest no-go evidence, are the
  direct consequence.