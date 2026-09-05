# Paper 1 — MKA: Pruning via Merging (EMNLP 2024)

**Title:** Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging
**Authors:** Deyuan Liu, Zhanyue Qin, Hairu Wang, et al.
**Link:** https://arxiv.org/abs/2406.16330 · **Code:** https://github.com/SempraETY/Pruning-via-Merging

---

## 1. Problem

LLMs are too large for edge devices. Existing compression:
- **Quantization** (8-bit weights) → needs hardware support, often fine-tuning.
- **Pruning** (remove weights/blocks) → throws away knowledge; hard to prune without damage.

**Core idea:** Merging is usually done *across models* (Model Soup, Task Arithmetic).
This paper asks: **why not merge redundant layers *inside* one model?**

## 2. Key Insight

Trained LLMs are **overparameterized** — many layers learn nearly the same mapping.
Late layers especially have high output redundancy (`Fig. 3` of the paper shows
late-layer similarity). If two layers produce almost identical activations `H` for
the same inputs, they can be fused into one layer using a weighted average of their
parameters.

## 3. Method (exact equations)

### Step A — Extract activations
For calibration data `ω`, run the model and record the activation `H^l` of every layer.

### Step B — Diffusion map embedding
1. Affinity matrix (Gaussian kernel on pairwise distances):
   `W_ij = exp(-||H_i^l - H_j^l||² / σ²)`
2. Degree matrix: `D_ii = Σ_j W_ij`
3. Symmetric diffusion operator: `P̃ = D^(-1/2) W D^(-1/2)`
4. Eigen-decomposition of `P̃` → eigenpairs `{(λ_k, φ_k)}`.
   Eigenvalues order: `1 = λ₁ ≥ λ₂ ≥ …`.
5. Diffusion map at time `t` (omitting the trivial eigenvector `φ₁`):
   `Φ_t(i) = (λ₂ᵗ φ₂(i), …, λ_{k+1}ᵗ φ_{k+1}(i))`
   → `Ψ^l` is the embedding of layer `l`. Euclidean distance in this space ≈ diffusion distance (manifold geometry preserved).

### Step C — Layer similarity (NPIB → Gaussian NMI)

The paper's headline metric is the **Normalized Pairwise Information
Bottleneck (NPIB)** — a normalized mutual-information score over the diffusion
embeddings `E^i, E^j` (paper Eq. 3):

`S_ij = NPIB(E^i, E^j) = Σ_ij p(x,y)·log(p(x,y)/(p(x)p(y))) / sqrt( Σ_x p(x)log p(x) · Σ_y p(y)log p(y) )`

The formal appendix (v2/v3) makes the same score concrete: assume the diffusion
embeddings `Ψ^l`, `Ψ^m` are **jointly Gaussian**, then MI has a closed form in
terms of covariance determinants, and the NMI is

`I(Ψ^l; Ψ^m) = ½·ln(|Σ_l|·|Σ_m| / |Σ_{l,m}|)`, with `H(Ψ) = ½·ln((2πe)^d |Σ|)`, and

`S_lm = I(Ψ^l; Ψ^m) / sqrt(H(Ψ^l)·H(Ψ^m))`

`S_lm ∈ [0,1]`: 1 = identical layers, 0 = unrelated layers. **Our repo
implements the Gaussian-NMI form** (`src/similarity/gaussian_mi.py`) — identical
in spirit to the NPIB and reference-implementation's default.

### Step D — Merge loop (Algorithm 1)
```
while a pair (l,m) with S_lm ≥ threshold τ exists:
    α = S_lm                          # (or softmax version)
    θ̃_c = α·θ_l + (1-α)·θ_m          # fused parameters
    replace layers l,m with layer c
```

### Step E — Information Bottleneck justification
The merged representation `Ψ_c = α·Ψ_l + (1-α)·Ψ_m` minimizes
`L_IB = I((Ψ_l,Ψ_m); Ψ_c) - β·I(Ψ_c; Y)`.
Setting `α = S_lm` is a tractable heuristic that weights the layer sharing more
information more heavily.

## 4. Results (for our related-work table)

| Model | Compression | Acc drop (MMLU) |
|---|---|---|
| Llama3-8B | 43.75% | −2.82% |
| Llama3-70B | ~45% | ~ −5% |
| Llama2-7B | 31.25% | < 5% |
| Llama2-13B | 57.5% | < 5% |

MKA **beats ShortGPT** (pruning) at the same compression ratio and combines with
quantization (SmoothQuant/GPTQ/AWQ).

## 5. What we must implement (mapping)

| Paper step | Our module |
|---|---|
| Extract activations | `utils/activations.py` |
| Diffusion operator + diffusion map | `similarity/diffusion.py` |
| Gaussian NMI similarity | `similarity/gaussian_mi.py` |
| Similarity matrix / strategy pattern | `similarity/` package |
| Merge weight α = S_lm (or softmax) | `merging/weighted.py` |
| Fuse θ̃ = αθ₁+(1-α)θ₂ | `merging/weighted.py` |
| Model rewiring loop | `merging/merger.py` |