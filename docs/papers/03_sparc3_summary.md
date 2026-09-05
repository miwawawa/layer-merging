# Paper 3 — SparC3: Attribution-Guided Pruning (2025)

**Title:** Attribution-Guided Pruning for Insight and Control: Circuit Discovery and Targeted Correction in Small-scale LLMs
**Authors:** Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, et al.
**Link:** https://arxiv.org/abs/2506.13727 · **Code:** https://github.com/erfanhatefi/SparC3

---

## 1. Problem

- LLMs are hard to diagnose and hard to fix when they misbehave (toxic output,
  repetition loops).
- Fine-tuning to fix behavior is expensive and risks breaking general ability.
- Non-attribution pruning ("Wanda", magnitude) prunes blindly — it removes the
  *least used* weights but can't tell us *why*, and can't target a *behavior*.

## 2. Key Idea

Use **Layer-wise Relevance Propagation (LRP)** — an explainability method — to
score every parameter by how much it contributes to a prediction. Then:
- **Compress:** prune the lowest-relevance parameters.
- **Discover circuits:** prune everything irrelevant to a *specific task*;
  what survives is the task's circuit.
- **Correct behavior:** prune the circuit that is **specific to a harmful
  behavior** but unimportant for general use.

## 3. Core equations

### Attribution-based pruning (Sec 3.1)
Aggregate relevance over reference samples `𝒳_ref`:
`R̄_ψk = (1/n_ref) · Σₓ R_ψk(x)`                                              (Eq. 1)

Rank and mask the `q` least relevant components:
`{c}_q = argsort(ℛ)[1..q]`
`ψ_i → (1 - 𝟙_{i∈{c}_q}) · ψ_i`                                              (Eq. 3)

### LRP back-propagation rule (Sec 3.2)
For a linear layer: `z_ij = a_i · w_ij`, `z_j = Σ_i z_ij`, then
`R_{i←j} = (z_ij / z_j) · R_j` and  `R_i = Σ_j R_{i←j}`                           (Eq. 5-6)
*(Relevance flows backward proportional to contribution; we add an ε-stabilized
denominator in code.)*

### Contrastive relevance for model correction (Sec 3.4)
`R̄_diff_ψk = R̄^General_ψk − R̄^Undesired_ψk`                                     (Eq. 7)
Sort ascending → prune parameters that are **important for the bad behavior**
but **least important for general behavior**.

## 4. Results (paper's table)

| Task | What was pruned | Effect |
|---|---|---|
| Compression (50% sparsity) | weights, row-wise | PPL barely rises (TinyLlama 7.98→11.52) |
| IOI circuit discovery | ~9% of MLP weights (unstructured) | task accuracy preserved |
| Toxicity reduction | ~100 fc1 neurons (~0.3%) | toxicity drops w/o general loss |
| Repetition fix | ~7,000 weight elements (~0.03%) | repetition eliminated, PPL same |

LRP finds **sparser** circuits than Wanda / gradient / activation baselines.

## 5. What we take for our assignment

| Concept | Our module |
|---|---|
| LRP relevance propagation | `pruning/lrp.py` |
| Weight-importance scoring | `pruning/importance.py` |
| Mask + apply (prune) | `pruning/pruner.py` |
| Compare LRP-prune vs random-prune | `experiments/phase5_pruning.py` |

> Full step-by-step numbers on *our* MNIST runs: `docs/learning/04_sparc3_calculations.md`.