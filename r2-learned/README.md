# R2 — Learned GPT-2 Layer Merging

## Research goal

Building on the strong direct-removal baseline identified in [R1](../r1-baselines/), we test whether **learned merging of adjacent Transformer blocks** can preserve GPT-2's predictive behavior better than deletion as compression increases.

Two **different methods** were developed and evaluated. The experiments used GPT-2 Small (12 blocks, 124,439,808 parameters), with up to five merge operations (seven blocks remaining; **28.48% fewer parameters**). Evaluation focuses on PPL measured on **DBpedia-14, AG News, and IMDb** with model selection performed separately.

**Important:** Although the two approaches use the same model family and output size, they have different sampling and evaluation settings. Compare each method with its **own** uncompressed GPT-2 and removal control; the absolute PPL numbers below are **not a standardized head-to-head comparison**.

## Approach A — Teacher-guided nonlinear replacement (Hiroyoshi)

### Idea

Instead of choosing one existing layer or averaging their pretrained parameters, **train a single replacement block** to approximate two adjacent blocks' transformation:

```text
Teacher:   x → block_i → block_(i+1) → y
Student:   x → learned replacement M      → ŷ
```

The remaining pretrained blocks are frozen. The replacement is optimized for hidden-state reconstruction and similarity to the teacher's next-token distribution (distillation with a KL term), not merely parameter distance.

### Procedure

1. Test direct removal, averaging, affine multiplication, and nonlinear learned merging.
2. For a candidate adjacent pair or group boundary, train the proposed replacement block while keeping the remaining GPT-2 blocks fixed.
3. Select the next compression operation greedily using validation PPL.
4. Repeat through five compression operations, allowing both independent pairs and repeated merging into already compressed groups.
5. Lock selected merge plans before evaluating on unseen external text.

The reported training configuration used **AdamW**, `3e-5` learning rate, **15 epochs** in full mode, weight decay `0.01`, and distillation temperature `2.0`.

### Results: PPL increase (%) relative to Approach A's own original GPT-2

| Merges | Parameter reduction | DBpedia: Remove → Learned | AG News: Remove → Learned | IMDb: Remove → Learned |
|---:|---:|---:|---:|---:|
| 1 | 5.70% | 4.0 → 4.1 | 2.9 → **2.4** | 3.4 → **3.0** |
| 2 | 11.39% | 19.4 → **14.6** | **21.2** → 22.0 | 14.9 → **12.8** |
| 3 | 17.09% | 41.4 → **31.6** | 50.9 → **40.8** | 40.3 → **25.9** |
| 4 | 22.78% | 86.0 → **61.5** | 101.5 → **85.3** | 76.8 → **49.5** |
| **5** | **28.48%** | **174.8 → 93.0** | **191.4 → 123.8** | **128.8 → 77.5** |

At five merges, absolute PPL was **59.81 (DBpedia), 115.97 (AG News), and 66.03 (IMDb)**. The uncompressed reference values within this experiment were **30.99, 51.82, and 37.20** respectively.

**Finding:** Learned nonlinear replacement outperformed its removal baseline on **all three datasets for three through five compression operations**. Results at one or two merges were mixed.

**Code**
- [`Hiroyoshi/single_merge.ipynb`](Hiroyoshi/single_merge.ipynb): initial single-pair experiments.
- [`Hiroyoshi/multi_merge.ipynb`](Hiroyoshi/multi_merge.ipynb): repeated greedy pair selection, training, evaluation, and checkpoint export.

## Approach B — Learned weight-space merger (Amrita)

### Idea

Train a **shared merger network that produces updated block weights**, using the second block's pretrained parameters as the initial anchor. Instead of optimizing a fresh replacement block independently for each pair, a learned correction to the anchor is generated for candidate pairs. The merger network is used offline; it does **not** add its parameters to the deployed compressed GPT-2.

### Procedure

1. Initialize from a **drop-first / retain-second** control; correction heads start at zero.
2. Train teacher-guided weight corrections, using hidden-state and next-token distribution objectives plus regularization.
3. Measure candidate-pair quality and apply a guard against degraded performance relative to the anchor.
4. Select non-overlapping adjacent pairs and measure multi-pair model PPL.
5. Report performance against first-drop, ordinary averaging, and learned alternatives.

The provided notebook describes an updated **dual NVIDIA T4** configuration with **10 training epochs**, `0.01` weight decay, and distillation temperature `2.0`.

### Results: PPL increase (%) relative to Approach B's own original GPT-2

| Merges | Parameter reduction | DBpedia: Drop → Learned | AG News: Drop → Learned | IMDb: Drop → Learned |
|---:|---:|---:|---:|---:|
| 1 | 5.70% | 2.9 → **0.1** | **6.3** → 6.6 | 3.8 → **3.2** |
| 2 | 11.39% | 17.8 → **9.3** | **23.1** → 24.9 | 19.8 → **14.7** |
| 3 | 17.09% | 42.8 → **25.0** | 46.3 → **44.5** | 38.0 → **30.2** |
| 4 | 22.78% | 91.0 → **48.1** | 100.4 → **79.6** | 76.0 → **54.8** |
| **5** | **28.48%** | **163.9 → 74.1** | **178.8 → 123.0** | **145.3 → 84.9** |

At five merges, absolute PPL was **70.41 (DBpedia), 139.15 (AG News), and 68.23 (IMDb)**, versus this experiment's uncompressed PPL of **40.44, 62.40, and 36.90**. The selected 1-indexed pairs were `2–3`, `4–5`, `7–8`, `9–10`, and `11–12`.

**Finding:** Learned weight-space merging improves on its own drop-first control at **three through five merges on all three datasets**; one- and two-merge comparisons include small exceptions.

**Code and results**
- [`Amrita/weight_merge_v1.ipynb`](Amrita/weight_merge_v1.ipynb): earlier weight-merger implementation.
- [`Amrita/weight_merge_v2.ipynb`](Amrita/weight_merge_v2.ipynb): subsequent iteration.
- [`Amrita/pair_merge_early.ipynb`](Amrita/pair_merge_early.ipynb): earlier exploration.
- [`Amrita/weight_merge_results.pdf`](Amrita/weight_merge_results.pdf): reported comparison table and selected pairs.
- [`Amrita/weight_merge_results.csv`](Amrita/weight_merge_results.csv): experiment records.

## Reproducibility and limitations

- **Run notebooks in a compatible GPU notebook environment** (e.g., Kaggle), following their individual setup and configuration cells. They are exploratory notebooks, not a combined command-line package.
- Check model/tokenizer versions and dataset access when re-running; some cells were written for specific GPU memory budgets.
- The two approaches use **different evaluation protocols**, so direct cross-approach rankings require re-evaluation on identical token samples, lengths, and PPL implementations.
- Layer selection is greedy or pairwise; neither strategy proves a globally optimal compression pattern.
- The remaining PPL degradation at high compression and hardware-dependent latency are still important open challenges.

## Conclusions

The first study showed that deleting a redundant layer can be surprisingly effective and that naive averaging is usually a poor replacement. These learned approaches **surpass the relevant deletion control at higher compression** without restoring the uncompressed model's full performance. This motivates further work on merge selection, multi-merge interactions, and end-to-end functional objectives.

The numerical results above are taken from the supplied experiment summaries and accompanying notebooks. The folders retain the individual researcher names for code attribution.
