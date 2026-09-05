# Research Survey — What We Implement vs What Is Ongoing Research

> Compiled Sep 2026. Grounded in the three assignment papers plus current
> surveys (FUSE taxonomy survey, 2026; model-merging survey 2408.07666; 2025).
> Purpose: draw a clear line between the *implementable* scope of this
> assignment and the *active research* directions we deliberately do NOT code.

## 1. The two operations at the core

| Operation | Merging (Paper 1 MKA, Paper 2 CoM) | Pruning (Paper 3 SparC3) |
|---|---|---|
| Unit that changes | a whole `nn.Linear` layer | individual weight / neuron |
| Needs data? | calibration set (activations) | reference samples (LRP) |
| Needs training? | No | No |
| Cost | one similarity pass + rewiring | one backward pass + masking |

Both are *training-free* and *post-hoc*, which is exactly what an undergraduate
ML project can reproduce honestly. Everything below this line is implementable.

---

## 2. IMPLEMENTABLE now (this repo)

### 2.1 Layer similarity / redundancy measures
| Method | Paper | Effort | Status |
|---|---|---|---|
| Gaussian-affinity diffusion map (`W_ij = exp(-||H_i-H_j||²/σ²)`, `P̃ = D^(-1/2)WD^(-1/2)`, eig`Φ_t`) | MKA | src | ✅ `similarity/diffusion.py` |
| Joint-Gaussian NMI `I = ½ln(|Σ_l||Σ_m|/|Σ_{lm}|)` | MKA | src | ✅ `similarity/gaussian_mi.py` |
| Cosine activation baseline | CoM semantic score | trivial | ✅ `similarity/cosine.py` |

### 2.2 Fusion strategies (weight-space geometry)
| Method | Paper | Effort | Status |
|---|---|---|---|
| Weighted convex `θ̃ = αθ₁ + (1-α)θ₂`, `α=S_lm` | MKA | trivial | ✅ `merging/weighted.py` |
| Uniform 50/50 | — | trivial | ✅ `merging/uniform.py` |
| SLERP `sin((1-α)Ω)/sinΩ·θ₁ + sin(αΩ)/sinΩ·θ₂` | CoM | trivial | ✅ `merging/slerp.py` |
| greedy "most-similar compatible pair" loop | MKA Alg.1 | moderate | ✅ `merging/merger.py` |

### 2.3 Attribution-guided pruning
| Method | Paper | Effort | Status |
|---|---|---|---|
| LRP backprop `R_{i←j} = (a_i w_ij / z_j) R_j` | SparC3 | moderate | ✅ `pruning/lrp.py` |
| Per-parameter attribution aggregation over refs | SparC3 Eq.1 | moderate | ✅ `pruning/importance.py` |
| Global top-k masking | SparC3 Eq.3 | trivial | ✅ `pruning/pruner.py` |
| Magnitude / random baselines | — | trivial | ✅ |

### 2.4 What reproducing on a *small* model legitimately means
Sub-8B and small-scale reproduction is allowed by the assignment. We reproduce
the **geometry** of the paper results: layer-similarity matrices, compression–
accuracy trade-off curves, sparsity–accuracy curves — NOT the 8B numbers.

---

## 3. ONGOING research — surveyed, NOT implemented (deliberate)

The field has moved from "static averaging" to **context-sensitive, geometry-aware
and dynamic** merging. We list the active directions and why we stop short.

| Active direction | Representative works | Why we skip it |
|---|---|---|
| Data-free **adaptive weights** (norm/Fisher-based) | FroM, MAGIC, Fisher-Merging | We already vary weights via `S_lm`; extra heuristics add risk not insight |
| **Subspace / sparse** merging (trim + sign-correct) | TIES-Merging, DARE, DELLA | Cross-model; our merging is *within*-model (MKA) — different axis |
| **Task-vector arithmetic** | Task Arithmetic, Edit Model | Merges *across* fine-tunes; we merge layers of one model (assignment scope) |
| **Low-rank** merge projections | KnOTS, HO-GSVD, ISO-Merging | Would need SVD plumbing; beyond our seminar scope |
| **Optimization-based** merging (data-free objectives) | AWD, DOGE, Surgery | Requires backward optimization; not "training-free" anymore |
| **Routing / mixture-of-experts** merging | WEMoE, Twin Merging | Inference-time routing, different experiment budget |
| **Functional-geometry** interpolation | ACE-Merging, Fisher-Rao merging | Advanced manifolds beyond spherical/SLERP |
| LLM-scale training-free merging | Training-free LLM Merging | Compute beyond the assignment constraint |

### Open challenges the field itself names (good for related-work writing)
1. **No unified framework** — the FUSE survey (2026) is the first attempt at one taxonomy.
2. **Mergeability poorly understood** — when are two layers "mergeable"? (loss-landscape
   geometry, mode connectivity — see §4).
3. **Scalability** for very large models and many-tasks → N models at once.
4. **Standardization of evaluation** across merging papers.

---

## 4. Why merging works at all (theory we lean on in the report)

- **Linear mode connectivity**: nets trained from the same init stay linearly
  connected, so simple averages sit in low-loss basins.
- **Redundancy**: over-parameterized nets contain near-duplicate layers; MKA shows
  late layers of an LLM have near-1 NMI — fusing them is *information-preserving*.
- **Residual structure**: transformers' `x + F(x)` makes each layer a small
  perturbation; fusing two similar blocks barely moves the residual stream —
  exactly why Paper 2's SLERP and Paper 1's α = S are safe there, and why our
  `ResidualMLP` reproduces the effect that a plain MLP cannot (our key finding).

---

## 5. Where-to-read map (for the report's Related Work)

| Concept | Read | In our repo |
|---|---|---|
| Diffusion maps | MKA paper §3.2 | `similarity/diffusion.py` |
| NMI / joint Gaussians | MKA paper §3.3 | `similarity/gaussian_mi.py` |
| SLERP & layer-wise granularity | FUSE survey §3.1.3; CoM | `merging/slerp.py` |
| LRP & attribution pruning | SparC3 §3.1–3.2 | `pruning/*` |
| Model soups / weight averaging | Wortsman et al. 2022 | `merging/uniform.py` (baseline) |