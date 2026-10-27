# GPT-2 Layer Merging for Language Model Compression

**Research question:** Can adjacent Transformer blocks be consolidated into fewer blocks while retaining useful next-token prediction performance, and can learned merging outperform simple layer removal?

This repository documents a chronological experimental study of **GPT-2 Small (124M)**. We study redundant or overlapping transformations in adjacent Transformer blocks as a possible opportunity for **parameter reduction**, evaluating the resulting models primarily by **perplexity (PPL)**. Lower PPL is better.

> **Key result:** At five adjacent-pair merge operations, two separately developed learned-merging methods each reduced the GPT-2 parameter count by **28.48%** (from **124,439,808 to 89,000,448** parameters), while showing **less PPL degradation than their respective removal baselines** on DBpedia-14, AG News, and IMDb. The two experimental protocols differ, so the methods should not be directly ranked against each other using their reported raw PPL.

This is an ongoing **research investigation**, not a claim of state-of-the-art compression.

## 1. Motivation

Transformer layers can learn partially overlapping representations, raising the possibility of replacing several transformations with fewer computational blocks. Our objective is to reduce deployed model size while preserving model behavior.

For a baseline and compressed model evaluated under the same protocol, we measure

```text
PPL degradation (%) = 100 × (PPL_compressed / PPL_original − 1)
```

A successful method should be evaluated against **direct layer removal at the same compression level**, not only against naive parameter averaging.

| Experimental setting | Value |
|---|---|
| Model | GPT-2 Small, 12 Transformer blocks |
| Original parameter count | 124,439,808 |
| Compression | One to five merge operations (down to 7 blocks) |
| Five-merge parameter count | 89,000,448 (−28.48%) |
| Main model-development corpus | WikiText-2 |
| External evaluation | DBpedia-14, AG News, IMDb |
| Main metric | Perplexity (PPL) and percentage increase over the same experiment's original-model baseline |

## 2. Research chronology

### Phase I — Baselines, affine approximations, and the strength of layer removal

We first tested all 11 neighboring GPT-2 layer pairs using:

- **Remove:** directly delete one block.
- **Average:** average weights of adjacent blocks.
- **Random-weight average:** randomized averaging as a control.
- **Affine approximations / composition:** fit linear-affine maps to local hidden-state changes and compare averaging with composing fitted maps.

**Results (validation PPL; lower is better):**

| Configuration | Validation PPL | Parameters reduced |
|---|---:|---:|
| Original GPT-2 | **29.65** | 0% |
| Remove block 1 (0-indexed) | **30.95** | **5.70%** |
| Best affine-average replacement (pair 6–7, 0-indexed) | 41.98 | 10.92% |

The affine replacement changes the architecture more than simple block deletion; these rows have **different compression ratios** and are not a matched-size comparison.

**Findings and challenges**

1. **Direct removal was unexpectedly strong.** Removing block 1 increased PPL only **4.4%**, making deletion a challenging baseline to surpass.
2. **Naive merging was unstable.** Affine averaging outperformed affine multiplication for **9 of 11 pairs**, while random-weight averaging often produced extremely high PPL.
3. **Local reconstruction error did not imply language-model fidelity.** On pair 1–2, affine multiplication achieved hidden-state MSE **0.841** but PPL **109.12**; averaging yielded MSE **24.599**, yet better PPL **54.46**.
4. Compression lowered measured inference latency in this exploratory setup; performance on other hardware remains to be established.

**Research direction established:** A replacement block must preserve the original model's behavior, rather than only minimizing weight distance or local hidden-state MSE. This motivated **learned, teacher-guided merging** in Phase II.

**Code:** [Phase I notebook](r1-baselines/gpt2_affine_sweep.ipynb) · [Modular GPT-2 implementation](r1-baselines/gpt2_layer_merging/) · [Run instructions](r1-baselines/README.md)

### Phase II — Two learned approaches to multi-block compression

The second phase tested two distinct approaches against direct removal and other training-free baselines. Both evaluated models after **1–5** merge operations.

#### Approach A — Sequential nonlinear merging

**Concept:** For adjacent transformations `Φ₁` and `Φ₂`, train a replacement Transformer block `M` so that `M(x) ≈ Φ₂(Φ₁(x))` and the final next-token distribution remains close to the original model.

**Method**

1. Evaluate **Remove, Average, LinearMultiply (affine), and NonlinearMultiply (trained replacement)**.
2. Train only the newly merged block; freeze the other pretrained GPT-2 blocks.
3. Optimize normalized hidden-state reconstruction error together with teacher–student **KL divergence**.
4. Greedily select the next pair by WikiText-2 validation PPL; allow merging previously consolidated spans.
5. Evaluate selected models on external datasets, not used in candidate selection.

The reported configuration uses **AdamW**, learning rate **3e−5**, **15 epochs**, context length **512**, distillation temperature **2**, and seed **42**.

**PPL degradation versus original GPT-2** — removal vs. nonlinear merging:

| Merged pairs | Parameters reduced | DBpedia-14 (Remove → Learned) | AG News (Remove → Learned) | IMDb (Remove → Learned) |
|---:|---:|---:|---:|---:|
| 1 | 5.70% | 4.0% → 4.1% | 2.9% → **2.4%** | 3.4% → **3.0%** |
| 2 | 11.39% | 19.4% → **14.6%** | 21.2% → 22.0% | 14.9% → **12.8%** |
| 3 | 17.09% | 41.4% → **31.6%** | 50.9% → **40.8%** | 40.3% → **25.9%** |
| 4 | 22.78% | 86.0% → **61.5%** | 101.5% → **85.3%** | 76.8% → **49.5%** |
| **5** | **28.48%** | **174.8% → 93.0%** | **191.4% → 123.8%** | **128.8% → 77.5%** |

At five merges, the trained model recorded PPL of **59.81 / 115.97 / 66.03** (DBpedia / AG News / IMDb), versus original-model PPL of **30.99 / 51.82 / 37.20** in this experiment.

**Interpretation:** The learned replacement was not always best at one or two merges, but improved over direct removal on all three datasets at **three through five merges**. Greedy stage-wise search does not guarantee a globally optimal merge plan.

**Implementation and final results:** [Final results (PDF)](r2-learned/nonlinear-merge/Final_Results.pdf) · [One-layer experiments](r2-learned/nonlinear-merge/single_merge.ipynb) · [Two-to-five-layer experiments](r2-learned/nonlinear-merge/multi_merge.ipynb)

#### Approach B — Learned weight-space merging

**Concept:** Train a **shared neural network in weight space** to generate a replacement from two adjacent pretrained blocks. Unlike Approach A, this method does not retrain a new replacement independently for each compression step.

**Method**

1. Start from the **drop-first** anchor (the second block's weights), with **zero-initialized** learned correction heads.
2. Train a shared weight generator on teacher-guided losses, including normalized hidden-state error, KL divergence, and a correction regularizer.
3. Score original non-overlapping neighboring pairs on held-out data, applying a guard that rejects changes that do not improve over the deletion anchor.
4. Select compatible non-overlapping pairs and evaluate complete compressed models.

The weight-merger network is an **offline generation tool**; its parameters are not deployed in the resulting compressed GPT-2.

**PPL degradation versus original GPT-2** — drop-first vs. learned merging:

| Merged pairs | Parameters reduced | DBpedia-14 (Drop → Learned) | AG News (Drop → Learned) | IMDb (Drop → Learned) |
|---:|---:|---:|---:|---:|
| 1 | 5.70% | 2.9% → **0.1%** | **6.3%** → 6.6% | 3.8% → **3.2%** |
| 2 | 11.39% | 17.8% → **9.3%** | **23.1%** → 24.9% | 19.8% → **14.7%** |
| 3 | 17.09% | 42.8% → **25.0%** | 46.3% → **44.5%** | 38.0% → **30.2%** |
| 4 | 22.78% | 91.0% → **48.1%** | 100.4% → **79.6%** | 76.0% → **54.8%** |
| **5** | **28.48%** | **163.9% → 74.1%** | **178.8% → 123.0%** | **145.3% → 84.9%** |

At five merges, learned PPL was **70.41 / 139.15 / 68.23**, versus uncompressed baseline PPL of **40.44 / 62.40 / 36.90** in this separate experiment. Its five selected original-layer pairs (1-indexed) were **2–3, 4–5, 7–8, 9–10, and 11–12**.

**Interpretation:** The learned weight generator improved over its removal baseline on all three datasets at **three through five merges**. Pair selection uses a single-pair cost proxy and does not guarantee the best combined multi-pair PPL.

**Implementation and results:** [Weight-merging notebooks](r2-learned/weight-merge/) · [Detailed results PDF](r2-learned/weight-merge/weight_merge_results.pdf) · [CSV results](r2-learned/weight-merge/weight_merge_results.csv)

### A note about comparing these approaches

**Approach A and Approach B are not evaluated under identical protocols.** They differ in data sampling, test token budgets, pair selection, and model construction. Compare each approach with **its own original GPT-2 baseline and removal control**; avoid directly ranking the two by their absolute PPL values.

## 3. Main research insights

- **Strong deletion baselines:** At low compression, removing a well-chosen Transformer block can outperform naive parameter merging.
- **Model-level objectives matter:** Small local MSE is not sufficient to predict full-model PPL.
- **Learned merging helps at higher compression:** Both approaches improve over their own removal baselines across all three external datasets at 3–5 merges.
- **Compression still has a cost:** Neither learned method fully preserves original GPT-2 performance at five merges, even though both outperform their removal controls.
- **Reproducibility and generalization are open:** Matched test protocols, additional model families, and hardware benchmarks are required.

## 4. Next research directions

1. Measure representation similarity and causal importance to identify mergeable adjacent layers.
2. Improve optimization objectives, initialization, and teacher-guided post-merge training.
3. Replace greedy selection and pairwise proxies with methods accounting for interactions between merges.
4. Test across models and standardized evaluation contexts; analyze PPL versus compression, speed, and memory.
5. Aim to consistently **surpass direct layer removal at equal parameter reduction**.

## 5. Repository layout and reproducibility

The two research stages are organized by method:

| Stage | Code and results |
|---|---|
| **R1 — Baselines and affine replacement** | [`r1-baselines/`](r1-baselines/) — direct deletion, averaging, fitted affine maps, adjacent-pair sweeps, [detailed README](r1-baselines/README.md) |
| **R2A — Nonlinear replacement blocks** | [`r2-learned/nonlinear-merge/`](r2-learned/nonlinear-merge/) — teacher-guided notebooks and the final results PDF |
| **R2B — Learned weight merger** | [`r2-learned/weight-merge/`](r2-learned/weight-merge/) — weight-merger notebooks, [results PDF](r2-learned/weight-merge/weight_merge_results.pdf), and CSV |
| **Detailed R2 findings** | [`r2-learned/README.md`](r2-learned/README.md) |
| **Original code attribution** | [`PROVENANCE.md`](PROVENANCE.md) |

The research was conducted collaboratively by the project team.

The notebooks are research artifacts. The R1 package includes a run script; the R2 notebooks require individual GPU setups and do not yet form a single tested production pipeline.

**Research status:** Ongoing investigation into lightweight Transformer language models.
