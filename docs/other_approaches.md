# Other Approaches — Survey of Model-Compression & Weight-Merging Techniques

This document is the *related-work* companion to `approach.md`: everything `the
same kind of thing` as our layer merging / pruning can do, how each technique
works, when it wins, when it fails, and how it relates to what this repo does.

> Level of detail: survey-depth, not implementation-depth. Each row names the
> mechanism, one or two landmark works, and the decisive trade-off. The four
> techniques at the end (marked 🔗) are implemented here.

---

## 1. Family map

```
            ┌─ Pruning        (remove parts: weights, neurons, layers, ...)
            │    ├─ magnitude / movement / attribution based   (unstructured)
            │    ├─ structured (neuron/channel/filter/layer)   (hardware-friendly)
            │    └─ lottery-ticket (sparse training + prune-retrain)
  model
  compression ──── Low-rank decomposition  (SVD, Tucker, Kronecker, CP)
            │    ├─ post-hoc SVD of weight matrices
            │    └─ rank-adaptive training (LoRA, GaLore)
            ├─ Quantization      (fewer bits per weight: FP16→INT8/INT4)
            ├─ Knowledge distillation  (small student learns teacher outputs)
            ├─ Weight sharing / clustering  (tie parameters, k-means codes)
            └─ WEIGHT MERGING     (combine whole models / layers)   ← we are here
                 ├─ within-model layer fusion   (MKA, CoM)   🔗
                 ├─ cross-model average          (model soups)
                 ├─ task-vector arithmetic       (Task Arithmetic, TIES, DARE)
                 ├─ geometry-aware interpolation (SLERP, Fisher-Rao, ACE)  🔗(SLERP)
                 └─ mixture-of-experts routing   (WEMoE, Twin Merging)
```

---

## 2. Pruning

### 2.1 Unstructured pruning
| Method | Mechanism | Strengths | Weaknesses | Relation to us |
|---|---|---|---|---|
| **Magnitude** (Han et al. 2015) | drop weights near zero | trivial, data-free, strong baseline | conflates "size" with "importance" | our baseline ✅ |
| **LRP / attribution** (SparC3 2025) | rank weights by signed LRP relevance | finds task-critical small weights; decorrelated from magnitude | needs reference samples, backprop-style pass | 🔗 our SparC3 track |
| **Movement pruning** (Movement/Sanh et al. 2020) | prune based on accumulated *gradient of mask* during fine-tune | adapts importance during training | needs fine-tuning (not post-hoc) | out of scope (we are training-free) |
| **Lottery-ticket hypothesis** (Frankle & Carbin 2019) | find sparse subnet that trains as well as dense | *training-time* wins, interpretable | needs iterative prune-retrain | different axis (we compress a *trained* model) |

### 2.2 Structured pruning
| Method | Mechanism | Strengths | Weaknesses | Relation |
|---|---|---|---|---|
| Neuron/channel pruning | drop whole output dims; network narrows | real FLOP/param/memory win, no sparse kernels needed | re-shapes layers → breaks merging's same-shape requirement | complementary; **not** what SparC3 does (weight-level) |
| Filter pruning (CNN) | drop conv filters by norm/saliency | big speedups | needs the layer to be *narrowable* (not symmetrically shared) | n/a on our symmetric MLP |
| Layer-drop / stacking | delete whole layers (e.g., LayerDrop) | extreme speedup | accuracy cliffs; needs training to resist | COUSIN of MKA — but naive positional, not similarity-driven |

### 2.3 Why we chose global weight masking
SparC3 prunes *across the whole model* (global top-k) rather than per-layer —
our Phase 5 shows global, when driven by LRP, beats per-layer and both
baselines; and it behaves identically for every method so comparisons are fair.

---

## 3. Low-rank decomposition

| Method | Mechanism | Strengths | Weaknesses | Relation |
|---|---|---|---|---|
| **SVD** of each weight matrix `W ≈ U Σ Vᵀ_top-r` | replace with two thin factors | large theoretical compression, invertible | rank selection is fiddly; Frobenius–accuracy gap; needs re-tuning after | we keep full-rank layers (MKA is *rank-preserving*) |
| **Tucker / CP / Kronecker** | higher-order / block structure factorization | extreme rates for tensors | MLP `nn.Linear` is a matrix — little to exploit | n/a |
| **LoRA / GaLore** (rank-adaptive *training*) | constrain ΔW to low rank during training | trains small, deploys full | requires retraining from scratch | out of scope (post-hoc) |

**Relation note:** SVD gives the best *per-layer* FLOP reduction, but it is
orthogonal to merging — MKA deletes whole layers instead of shrinking them. A
real system would merge redundant layers *and* low-rank the fat ones; that is a
natural "future work" for our report.

---

## 4. Quantization

| Variant | Mechanism | Strength | Weakness | Relation |
|---|---|---|---|---|
| FP16/INT8 PTQ | map weights to fewer bits with calibration | near-free, big memory win | accuracy loss at 4-bit without fine-tune | orthogonal (we keep float32) |
| QAT | learn quantization scales during training | best accuracy per bit | needs training | out of scope |
| Vector / GPTQ / AWQ | group-wise + Hessian-aware rounding | LLM-scale 4-bit | complex | n/a for the seminar |

Quantization and our merging reduce *different* resources (bits vs structure)
— the evidence table reports both params and `size_mb`, and a hybrid
"merge+quantize" would multiply the savings.

---

## 5. Knowledge distillation

| Mechanism | Strength | Weakness | Relation |
|---|---|---|---|
| teacher `T` teaches student `S ≪ T` via logits/features; train `S` | can go far smaller than pruning; no per-weight decisions | needs the whole training pipeline again | **training-based**, we are training-free; complements (distill the merged model) |

---

## 6. Weight sharing / clustering

| Method | Mechanism | Strength | Weakness | Relation |
|---|---|---|---|---|
| **Huffman / k-means codes** (Han et al.) | many weights share one codebook entry | shrinks disk footprint | does not reduce FLOPs | orthogonal (memory vs compute) |
| **Weight tying** (e.g., tied embeddings) | force equal tensors to be the same object | 2× on tied parts | only layout-specific | conceptually related to merging (we *make* layers equal) |

---

## 7. Weight merging — the closest family to our task

### 7.1 Within-model layer fusion 🔗 (this repo)
MKA (EMNLP 2024) and CoM (CVPR 2026) fuse redundant layers of a *single*
network using similarity (both implemented here):

| | MKA | CoM | Ours |
|---|---|---|---|
| similarity | diffusion-map + Gaussian NMI | structural (weights) + semantic (activations) scores | 🔗 diffusion+NMI, cosine baseline |
| fusion | α=S_lm convex blend | SLERP + adaptive per-pair weight | 🔗 weighted/uniform/SLERP |
| pair choice | greedy max-similarity | non-positional matching | 🔗 greedy, non-positional |
| scope | layer layers of one model | layers across TWO models (skill transfer) | within-model (MKA scope) |

### 7.2 Cross-model averaging
| Method | Mechanism | Strength | Weakness | Relation |
|---|---|---|---|---|
| **Model soups** (Wortsman 2022) | average weights of fine-tunes of the same init | free robustness | needs same init/architecture | our `uniform` strategy is the same algebra *within* one model |
| **Task Arithmetic, TIES, DARE** | add/subtract task vectors; trim + sign-merge | compose capabilities | cross-model identity mismatch; sign/outlier magic | surveyed (`research_survey.md`), not needed for within-model fusion |
| **Fisher-Rao / ACE (geometry-aware)** | interpolate on functional-geometry manifolds | theoretically principled | advanced math, needs careful estimators | we stay at spherical (SLERP) granularity |
| **WEMoE / Twin Merging** | route/expert-weight the two models at inference | keeps both specialists | inference-time overhead, complex | beyond seminar scope |

### 7.3 Why within-model fusion (not cross-model)
The assignment's three papers center on **compressing a trained model**; the
cross-model family optimizes *capability composition*. Our experiments stay in
the MKA/CoM lane and therefore reproduce those papers' geometry directly.

---

## 8. Head-to-head summary (for the Related Work table)

| Approach | Params↓ | FLOPs↓ | Memory↓ | Needs retrain? | Needs data? | Hardware-friendly? | In this repo |
|---|---|---|---|---|---|---|---|
| **Layer merging (ours)** | **yes** | **yes** | yes | no | calibration | yes (real, dense) | 🔗 |
| LRP pruning (SparC3) | sparsity | only w/ sparse kernels | yes | no | reference | needs sparse kernels | 🔗 |
| Magnitude pruning | sparsity | only w/ sparse kernels | yes | no | no | needs sparse kernels | 🔗 |
| Structured pruning | yes | yes | yes | often | calibration | yes | — |
| SVD / low-rank | yes | yes | yes | light retune | sometimes | yes | — |
| Quantization | — | no | yes (bits) | PTQ: no | calibration | yes | — |
| Distillation | yes | yes | yes | **yes** | yes | yes | — |
| Weight clustering | — | no | yes | no | calibration | yes | — |
| Cross-model merging | composes | — | — | no | yes | — | *(surveyed, out of scope)* |

**Net takeaway.** Layer merging (MKA/CoM) is the only *training-free* method
that removes genuine compute (FLOPs) at the structural level; pruning is best
for memory-limited sparsity; low-rank gives the deepest per-layer cuts; the
other families optimize different axes (bits, codebooks, or capability
composition). Our pipeline therefore reports params, FLOPs, footprint, latency,
**and** sparsity — a reduction story no single axis captures.

---

## 9. Sources referenced

- FUSE — unified taxonomy of model merging (survey, 2026).
- Model-merging survey arXiv 2408.07666 (2025).
- Han, Mao, Dally 2015 (deep compression); Frankle & Carbin 2019 (lottery).
- Sanh et al. 2020 (movement pruning); Wortsman et al. 2022 (model soups).
- Ilharco et al. (task arithmetic); Yadav et al. (TIES); Yu et al. (DARE).
- MKA 2024, CoM 2026, SparC3 2025 — the three assignment papers
  (`docs/papers/*_summary.md`).