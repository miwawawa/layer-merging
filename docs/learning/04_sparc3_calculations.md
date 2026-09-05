# SparC3 attribution-guided pruning — step-by-step calculations

> Numerical walk-through of Paper 3 (Hatefi, Dreyer et al., 2025,
> arXiv:2506.13727) on our plain MLP, from forward pass to mask. Prereqs:
> math primer §10. Code: `src/pruning/lrp.py`, `importance.py`, `pruner.py`,
> `src/experiments/phase5_pruning.py`.

---

## Step 0 — the model we prune

Plain MLP (`SimpleMLP`): `784 → 256 → 256 → 256 → 256 → 10`, ReLU. No residual
stream: LRP's backwards rule is *exact* here (the honest reason the pruning
study runs on `arch="plain"`).

## Step 1 — forward pass, per sample

For one input `x` (784 dims) through layer `k` (`W_k: (256,256)`, bias `b_k`):

```
a_k   = relu( W_k·a_{k-1} + b_k )
logits = W_head·a_4 + b_head
p      = softmax(logits)                # class probabilities
c*     = argmax p                      # predicted label
```

### Worked LRP rule (bias-free for clarity)

Unit `j`, inputs `a`, weights `W_{·j}`:

```
z_ij = a_i · W_ij               # contribution of edge (i → j)
z_j  = Σ_i z_ij                 # total pre-activation
R_{i←j} = (z_ij / z_j) · R_j    # relevance redistributed proportionally
R_i     = Σ_j R_{i←j}           # this input's total relevance
```

Numbers: `a = [1.0, 3.0]`, `W_{·j} = [2.0, -1.0]`, `R_j = 0.7`
→ `z_j = 2 − 3 = −1`,
→ `R_{0←j} = (2/−1)·0.7 = −1.4`, `R_{1←j} = (−3/−1)·0.7 = 2.1`.
Check: `−1.4 + 2.1 = 0.7 = R_j` ✓ (conserved).

Bias is included in the real denominator; the code (`pruning/lrp.py`) uses the
**epsilon-stabilized** denominator so no edge `/0`:

```
z_j  ->  z_j + sign(z_j)·eps·mean(|z_j|)         eps = 1e-2 (Config.lrp_eps)
```

Relevance starts at the output as a one-hot on the predicted class, scaled by
its own softmax probability (`_output_relevance`):
`R_c* = p_c*`.

## Step 2 — per-weight relevance (SparC3 Eq. 5–6)

At hidden layer `k`, the pre-activation `z_j = Σ_i a_i·W_ij` makes weight
`(i,j)`'s contribution exactly `z_ij = a_i·W_ij`. The code keeps the **per-edge
matrix** `R_{i←j}^{(k)}` for every hidden layer (not just `R_i`):

```
contrib^{(k)}  =  (C / zs) * R_out         # C_{b,i,j} = a_b,i · W_{k,j,i}
per_layer[k]   =  mean_b contrib^{(k)}     # shape (in, out) = (256, 256)
```

## Step 3 — aggregate over reference samples (Eq. 1)

```
R̄_ψk = (1/n_ref) · Σ_{x ∈ 𝒳_ref} R_ψk(x)     n_ref = n_calibration_samples = 512
```

⇒ one (aggregated) importance score per weight per layer. The output layer has
no `R_{i←j}` attribution (it defines relevance), so it gets a **magnitude**
fallback (`abs(w)`) — a code-level decision we document in `importance.py`.

## Step 4 — rank & mask (Eq. 3)

```
{c}_q = argsort(ℛ)[1..q]        # indices of the q weakest-scoring weights
ψ_i  →  (1 − 𝟙_{i∈{c}_q}) · ψ_i # zero them
```

Keep fraction `keep_ratio = cfg.prune_sweep` ∈ `(1.0, 0.9, 0.75, 0.5, 0.3, 0.1)`.
Globally across all layers (`prune_model_global`) at `0.5`:

```
pruned_params  = 0.5 · 400,906             ≈ 200,453  (weights only; biases = 2,507 stay)
sparsity       = 1 − nonzero/total         = 0.4987   (repo: phase6 final_results.json)
```

## Step 5 — the three scorers vs each other (phase 5, `pruning_curves.png`)

At the operating point `keep_ratio = 0.5` (seed 42):

| scorer | retention | test acc | note |
|--------|-----------|----------|------|
| **lrp** (SparC3) | 0.7508 | 0.7330 | attribution-guided |
| magnitude | 0.9969 | 0.9730 | naive |w| |
| random | 0.6594 | 0.6436 | chance baseline |

Multi-seed (42, 7, 2026): LRP retention `0.7376 ± 0.0677`, `prune_acc_delta`
`−0.2565 ± 0.0661`.

**Why LRP < magnitude here:** LRP highlights *class-discriminative* weights;
pruning them removes the exact features this shallow MNIST network needs. The
paper's advantage (circuit discovery) appears on deep LLMs where sparsity /
PPL are the metric (TinyLlama 50% sparsity: PPL 7.98 → 11.52; IOI circuit at
~9% of MLP weights, toxicity/repetition fixes at ~0.3% of weights). On a
scaled-down model with no OOD-behavior target, magnitude's "remove the unused"
is simply the right inductive bias. We report both honestly — the *reproduced
pipeline* (LRP) and the *toy-appropriate baseline* (magnitude).

## Step 6 — contrastive relevance (Eq. 7) — the intended correction use

```
R̄_diff_ψk = R̄^General_ψk − R̄^Undesired_ψk
c_q       = argsort(R̄_diff)[1..q]     # least desired for general, most for harmful
```

Pruning those *targets* the harmful circuit. No harmful behavior exists in
MNIST, so we implement the machinery and do not claim a correction result —
the compression evidence is Eq. 1 + Eq. 3, which is the paper's primary
compression recipe.