# Paper 2 — CoM: Chain-of-Merging (CVPR 2026)

**Title:** Beyond Layer-Wise Merging: Chain-of-Merging for Vision-Language Models
**Authors:** Xinyu Zhang, Yuxuan Dong, Lingling Zhang, et al.
**Link:** https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Beyond_Layer-Wise_Merging_Chain-of-Merging_for_Vision-Language_Models_CVPR_2026_paper.pdf

---

## 1. Problem

Model merging works for LLMs, but naive **layer-wise merging** for Vision-Language
Models (VLMs) has two flaws:

1. **Strict positional correspondence** — always merges `layer_i` with `layer_i`
   of the second model. In a VLM+LLM merge, corresponding layers have *very
   different functions* → misalignment → harmful parameter combinations.
2. **Uniform merging weights** — one weight for all layers, ignoring functional
   disparities between layer pairs.

## 2. Key Idea (Chain-of-Merging, CoM)

**Adapt the merging plan per input** (image + question). Two stages:

### Stage 1 — Adaptive Layer Matching
Don't assume positional pairing. Compute matching scores between layers:
- **Structural score** from weights (e.g., similarity of weight matrices).
- **Semantic score** from activations (e.g., feature similarity).
Filter out **incompatible** pairs → only good pairings survive.

### Stage 2 — Dynamic Weight Merging
- Per-layer-pair merging weight, derived from the matching scores
  (no longer a single global weight).
- Use **Spherical Linear Interpolation (SLERP)** to fuse weights,
  which is more memory-efficient than naive averaging and preserves
  weight norms.

## 3. SLERP recap (we reuse this)

To interpolate weight vectors `θ₁`, `θ₂` on a sphere:

```
Ω   = angle between θ₁, θ₂            Ω = arccos( (θ₁·θ₂) / (||θ₁||·||θ₂||) )
θ_m = [sin((1-λ)Ω)/sin Ω]·θ₁  +  [sin(λΩ)/sin Ω]·θ₂
```

When `sin Ω ≈ 0` (parallel weights), fall back to linear interpolation.
SLERP keeps the **magnitude** of weights, which matters for stable behavior.

## 4. Results

Qwen2.5-VL-7B + Qwen2.5-Math-7B → **+4.4% average** on math benchmarks,
and general visual understanding also improves. Outperforms existing
**training-free** merging methods.

## 5. What we take from this paper

| Concept | Our module |
|---|---|
| Layer matching is a *problem* (not positional) | `merging/merger.py` picks the *most similar* pair, not adjacent |
| Per-pair dynamic weights | `merging/weighted.py` (α from similarity) |
| SLERP fusion | `merging/slerp.py` |
| Structural matching (weights) | `similarity/cosine.py` on weights |
| Semantic matching (activations) | `similarity/` on activations |