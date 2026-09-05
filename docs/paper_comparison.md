# Comparative Review of the Three Papers

| Aspect | MKA (EMNLP 24) | CoM (CVPR 26) | SparC3 (2025) |
|---|---|---|---|
| **Operation** | Merge layers *within* a model | Merge layers *across* two models | Prune weights *within* a model |
| **Unit of change** | Whole layer | Whole layer | Individual weight / neuron |
| **Similarity signal** | Diffusion map + Gaussian NMI (`S_lm`) | Structural (weights) + semantic (activations) scores | LRP relevance scores |
| **Merge/prune rule** | `θ̃ = αθ₁+(1-α)θ₂`, `α=S_lm` | SLERP + per-pair adaptive weight | Mask least-relevant `q` weights |
| **Training needed?** | No (calibration only) | No (training-free) | No (reference samples only) |
| **Compression** | 31-58% layer reduction | — (skill transfer) | up to 50%+ sparsity |
| **Main benchmark** | MMLU, PIQA, HellaSwag… | Math reasoning on VLMs | WikiText2 ppl, IOI, toxicity |
| **Strength** | Preserves knowledge (no discard) | Breaks rigid layer correspondence | Locates *behaviors* precisely |
| **Weakness** | Needs symmetric layer shapes | VLM-specific | Weight-level ops are hardware-unfriendly |

---

## How they complement each other in OUR pipeline

```
MKA (Paper 1)  ──►  similarity + merge strategy ──► the core compression
CoM (Paper 2)  ──►  SLERP merge variant + "don't trust positional pairing" ──► best merge choice
SparC3 (Paper 3) ─►  LRP relevance + pruning ──► complementary compression + explainability
```

Our experiment = **merge** (MKA + SLERP variants) and **prune** (LRP) a small MLP
on MNIST, then compare both against each other on the accuracy-vs-compression axis.

## Reproduce-the-results note (assignment rule)

The assignment says: *"Do not reproduce closed-source / larger LLMs."* We therefore
reproduce the same *geometry* of the results (similarity matrices, compression vs
accuracy curves) on a small open, CPU-friendly model — not the 8B LLMs.