# R1 — GPT-2 Layer Removal and Affine Merging Baselines

## Objective

We investigate whether potentially redundant adjacent Transformer blocks in **GPT-2 Small** can be consolidated to reduce model parameters without a large increase in next-token **perplexity (PPL)**. The first phase establishes strong non-trained controls and tests whether fitted affine transformations can approximate the output of two blocks.

- **Model:** GPT-2 Small, 12 Transformer blocks, 124,439,808 parameters.
- **Data:** WikiText-2 training data for affine fitting; validation for selecting and comparing configurations; held-out WikiText-2 test data for the validation-selected model.
- **Metrics:** PPL (lower is better), hidden-state mean squared error (MSE), parameter count, and inference latency.
- **Adjacent pairs:** 11 candidate pairs, indexed `0–1` through `10–11` (zero-based layer indices).

## Methods

### A. Direct Transformer-block baselines

- **Remove:** delete one complete GPT-2 block from an adjacent pair.
- **Average:** replace the two original blocks with a single block with averaged corresponding weights.
- **Random-weight average:** a randomized comparison for testing sensitivity to parameter compatibility.

The direct block methods are implemented in [`gpt2_layer_merging/direct_baselines.py`](gpt2_layer_merging/direct_baselines.py) and [`model.py`](gpt2_layer_merging/model.py).

### B. Fitted affine replacement

Fit an affine map for each neighboring block's input/output hidden states using training text:

```text
h_out ≈ h_in @ A + b
```

Use the two fitted maps `(A1,b1)` and `(A2,b2)` to construct alternative **single affine replacement blocks**:

| Condition | Replacement matrix | Replacement bias |
|---|---|---|
| Remove (affine control) | A2 | b2 |
| Average | (A1 + A2)/2 | (b1 + b2)/2 |
| Multiply / composition | A1 @ A2 | b1 @ A2 + b2 |
| Random-weight average | fitted map and distribution-preserving shuffled weights | same principle |

The affine **Remove** control is **not** identical to deleting a real Transformer block: the experiments belong to different implementation families and should be compared separately.

Code: [`affine.py`](gpt2_layer_merging/affine.py) for fitting and replacement; [`experiments.py`](gpt2_layer_merging/experiments.py) for the full adjacent-pair sweep.

## Results and interpretation

### Direct removal was surprisingly strong

| Model | WikiText-2 validation PPL | Change from original |
|---|---:|---:|
| Original GPT-2 | **29.65** | — |
| Directly remove block 1 (0-indexed) | **30.95** | **+4.4%** |

A single direct block deletion reduces the GPT-2 parameter count by **approximately 5.7%**, yet this particular deletion barely changed validation perplexity.

### Naive parameter merging was less effective

In the **direct GPT-2** comparison for pair `1–2`, removal achieved **30.95 PPL** while ordinary weight averaging reached **166.25 PPL**. Across most directly tested middle-layer pairs, removal was consistently stronger than naive averaging.

In the **fitted affine** comparison, averaging had lower PPL than multiplying fitted maps for **9 out of 11** adjacent pairs. For example, pair `6–7` yielded **41.98 PPL** with affine averaging, versus **45.43** with affine multiplication. Neither result matched the original model's 29.65 PPL.

### Hidden-state MSE is an imperfect proxy for language modeling quality

For the affine pair `1–2`:

| Method | Hidden-state MSE | Validation PPL |
|---|---:|---:|
| Affine average | 24.599 | **54.46** |
| Affine multiplication | **0.841** | 109.12 |

Although composition reconstructs the locally measured representation more accurately, its downstream language modeling PPL is substantially worse. **Low hidden-state MSE does not by itself guarantee low end-to-end PPL.**

Measured latency was below the original GPT-2 latency for the tested compressed configurations, but results are hardware- and evaluation-protocol-dependent.

## Reproduce and inspect

```bash
cd r1-baselines
pip install -r requirements.txt
python run.py
```

The package is organized as follows:

| File | Role |
|---|---|
| [`gpt2_affine_sweep.ipynb`](gpt2_affine_sweep.ipynb) | Original exploratory notebook and outputs |
| [`gpt2_layer_merging/config.py`](gpt2_layer_merging/config.py) | Model, token budgets, seeds, and quick/full mode |
| [`gpt2_layer_merging/direct_baselines.py`](gpt2_layer_merging/direct_baselines.py) | Deletion and weight-averaging baselines |
| [`gpt2_layer_merging/affine.py`](gpt2_layer_merging/affine.py) | Ridge-regularized affine fits and replacement |
| [`gpt2_layer_merging/experiments.py`](gpt2_layer_merging/experiments.py) | Adjacent-pair experiments |
| [`gpt2_layer_merging/analysis.py`](gpt2_layer_merging/analysis.py) | Validation selection and held-out test evaluation |
| [`gpt2_layer_merging/pipeline.py`](gpt2_layer_merging/pipeline.py) | Complete experiment orchestration |

**Environment note:** This experimental package was arranged for Colab, with `RESULTS_DIR = "/content"` and notebook-style `display(...)` calls in the pipeline. Change the output directory and run it in a compatible notebook environment if needed.

Generated files include `affine_all_pairs_raw.csv`, `affine_all_pairs_summary.csv`, `best_method_per_pair.csv`, `multiply_vs_average.csv`, and (if enabled) `direct_all_pairs_summary.csv`. These are **generated on execution**, not committed results in this folder.

## Conclusion and next step

Our initial hypothesis—that merging redundant layers would be straightforward—did not hold for naive averaging or simple affine approximations. A well-chosen **removal baseline** was especially strong, and **local reconstruction error** proved insufficient as a selection criterion. These findings motivate learned replacement blocks that explicitly preserve the model's functional behavior. See [R2 — Learned Merging](../r2-learned/).

Results are from the experiment notebook and the accompanying Phase I result summary; different replacement families should not be treated as matched-parameter experiments.
