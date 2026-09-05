# CoM chain-of-merging — step-by-step calculations

> Numerical walk-through of Paper 2 (Zhang et al., CVPR 2026,
> arXiv:2502.16515) as reproduced on our MLP. We take two ideas from CoM:
> **(1) layer matching is a search, not a positional alignment**, and
> **(2) SLERP mixes on the sphere instead of the chord**. Both are computed
> through in `src/`. Prereqs: math primer §7–8.

---

## Idea 1 — matching is a search (our `merging/merger.py`)

CoM replaces "merge model-A layer i with model-B layer i" by an **adaptive
match**: pair layers by structural (weight) and semantic (activation) scores,
drop incompatible pairs, then merge only the good ones.

Our adaptation (single model, many layers): compute `S(i,j)` for *every*
shape-compatible pair, pick the argmax above a threshold:

```python
# phase3_merging.py / merger._best_pair
best_pair = argmax_{i<j, compatible(i,j)} S[i, j]    with  S[i,j] >= 0.05
```

First merge decision (seed 42) → pair `(1,2)`, `S = 0.988`, α=0.9.
Contrast with *positional* merging (windowed pairs `(0,1),(1,2),(2,3)`):
the greedy search converges in 3 merges to `269,322` params at retention
`0.993`; naive positional fusion is only one merge (`(0,1)`) and stops — fewer
params saved (`400,906`) at 0.9706. Search beats alignment here.

## Idea 2 — dynamic per-pair weight (our `weighted.py`)

CoM sets a per-pair weight from the matching score:

```
w = w_min + (w_max − w_min) · S_norm        S_norm ∈ [0,1]
```

Our MKA-style analogue is `α = clip(S_lm, merge_min_alpha, merge_max_alpha)`,
i.e. `w_min = 0.1, w_max = 0.9`. With the chosen pair:
`S = 0.988 → w = 0.1 + 0.8·1.0 = 0.9`.

## Idea 3 — SLERP fusion (our `merging/slerp.py`)

- Flatten weight tensors: `θ1, θ2 ∈ ℝ^{65,792}` (one 256→256 block).
- **Angle between them**:

  ```
  Ω = arccos( 〈θ1, θ2〉 / (‖θ1‖·‖θ2‖) )
  ```

- **Interpolate on the sphere:**

  ```
  θ(α) = sin((1−α)Ω)/sin Ω · θ1  +  sin(αΩ)/sin Ω · θ2
  ```

- **Identity check (unit-test level):** two perpendicular unit vectors at
  α=0.5 give `θ(0.5) = √2/2·θ1 + √2/2·θ2`, `‖θ(0.5)‖ = 1` — the output stays
  *on the sphere*, while the straight-line average has norm `0.707`. This is
  the memory-efficiency property CoM relies on.
- **Degenerate-guard:** `sin Ω < 1e-5` (parallel OR antiparallel) → plain
  average `α·θ1 + (1−α)·θ2`.
  For truly parallel `θ2 = 3θ1`: returns `0.25·θ1 + 0.75·θ2` (linear) — verified
  in `tests/test_merging.py`.

## Head-to-head: the three mixers (phase 4, `strategies.png`)

One identical pair, same baseline `466,698` params, same `S`:

| strategy | α used | merged acc | params | note |
|----------|--------|-----------|--------|------|
| weighted (MKA) | 0.900 | 0.9781 | 400,906 | α from similarity |
| uniform | 0.500 | 0.9786 | 400,906 | plain average |
| slerp (CoM) | 0.900 | 0.9786 | 400,906 | on-sphere interpolation |

All three are near-lossless on residual layers (redundant → any convex combo
fine). They differ in **what they fund**: SLERP preserves weight norms, which
matters for deeper/pretrained nets; the toy's accuracy is insensitive to that
detail — an honest limitation we report.

## Paper headline vs our implementation

- Paper: Qwen2.5-VL-7B + Qwen2.5-Math-7B, one matched chain of merges with
  input-adaptive plans → **+4.4% avg on math benchmarks**. We adopt the two
  mechanisms (search-like matching + SLERP), not the input-conditioned
  planning, which has no analogue in a single small MLP.
- Related, do-not-confuse: **Buzzega et al., arXiv:2508.21421, "Rethinking
  Layer-wise Model Merging through Chain of Merges"** — also "CoM", but a
  *different* paper (training-free chain of merges for LLM experts).
  Our implementation follows the CVPR 2026 Zhang et al. definition.