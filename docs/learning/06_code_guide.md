# Coding concepts — implementing the three papers in PyTorch

> The *engineering* ideas behind every line of `src/`, in the order you'd need
> them to build this yourself. Each concept names the exact file(s) that
> demonstrate it, and is sized so you can look the code up in seconds.
> Prereqs: `docs/learning/01_math_primer.md`.

---

## 1. Tensors, nn.Module, Parameter (no hidden magic)

`torch.nn.Linear(out, in)` holds `weight: (out, in)` and `bias: (out,)` as
`nn.Parameter` (tensors that register on the module and get gradients in
training). `model.parameters()` iterates ALL of them — that's what
`count_parameters` sums for the report's evidence numbers.

```python
# src/models/mlp.py, SimpleMLP
self.layer1 = nn.Linear(input_dim, hidden_dims[0])   # 784 -> 256
self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
```

**Why it matters for merging:** to fuse a pair we read `layer.weight.data`,
build a new `nn.Linear`, and use `copy_` to write the merged tensor in place.
Modules are Python objects — we can remove/replace them mid-run
(`merging/merger.py:_fuse` via `remove_linears` / `insert_linear`).

## 2. Forward pass vs forward trace (why "plain" matters)

`nn.Module.__call__` runs a model's `forward`. To *analyze* a model we
sometimes rebuild the sequence of `(activation_in, layer)` pairs ourselves:

```python
# src/pruning/lrp.py:_forward_trace
for layer in self.model.linears:          # NOTE: assumes a plain chain
    trace.append((h, layer))
    h = torch.relu(layer(h))
```

This is why LRP works on `SimpleMLP` and is *structurally undefined* on
`ResidualMLP` (whose block output is `h + act(W·h)`, not `act(W·h)`):
a trace that ignores the skip connection mis-sizes the tensors. Phase 6b
documents this as evidence, not an accident.

## 3. Activations = hooks or explicit forward

Two ways to get intermediate activations. We use the explicit one
(`extract_activations`) so there are zero hooks to manage and results are
deterministic — important because similarity runs on the SAME 512 samples
every phase.

```python
# src/utils/activations.py
_, acts = model.forward_with_activations(x)   # residual path returns acts
```

## 4. numpy <-> torch round-trips

Everything *structural* (masks, similarity, diffusion) is numpy; everything
*trainable* (weights) is torch. The bridge is explicit:

```python
mask_t = torch.from_numpy(mask.astype(np.float32)).reshape_as(layer.weight)
with torch.no_grad():
    layer.weight.data.copy_(layer.weight.detach() * mask_t)
```

`torch.no_grad()` = no autograd graph (we're rearranging parameters, not
training). `detach()` breaks the graph twice defensively. Phase 1 made the
mask write go through an explicit copy instead of an aliased `reshape(-1)`
view — look at `pruning/pruner.py:apply_prune`'s comment.

## 5. Masking: argsort + boolean array (SparC3 Eq. 3)

"Keep the top-q weakest-weighted to prune" is 3 lines:

```python
top = np.argsort(importance)[-keep:]      # indices of the largest keep scores
mask = np.zeros(importance.size, dtype=bool)
mask[top] = True
```

Structured (row-wise) masking ranks **rows** of the weight matrix, then
repeats the per-row decision — see the new `shape=` parameter in
`prune_mask`.

## 6. Diffusion maps: kernels, normalization, eigenvectors

```python
# src/similarity/diffusion.py
W   = np.exp(-dists**2 / sigma**2)               # Gaussian affinity (n,n)
np.fill_diagonal(W, 0.0)                          # no self-loops
Dinv = np.diag(1 / np.sqrt(D + 1e-12))
P   = Dinv @ W @ Dinv                             # symmetric D^-1/2 W D^-1/2
lam, phi = eigh(P)                                # symmetric eigen-solve, ascending
return (lam[1:n_dims+1] ** time) * phi[:, 1:n_dims+1]   # skip trivial λ=1
```

The symmetric normalization `D^{-1/2} W D^{-1/2}` lets us use the stable real
eigen-solver `scipy.linalg.eigh`; the leading eigenpair (λ=1, constant vector)
is degenerate and dropped, and `time` just exponentiates the eigenvalues
(`Φ_t = Σ λ^t·φ`).

## 7. The Config pattern: one source of truth + env override

Every tunable lives in one dataclass; runs never hardcode numbers.

```python
# src/config/default.py
@dataclass
class Config:
    hidden_dims: tuple = (256, 256, 256, 256)
    ...
    def __post_init__(self):
        self._apply_env()      # read LAYER_MERGING_<FIELD> from env/.env
        self._resolve_paths()  # absolute paths under the project root
```

Because `_apply_env` only overrides a field when the caller didn't pass a
value, `cfg = Config(seed=7)` still respects your explicit seed. Every
experiment calls `setup()` (the only place that reads config and touches the
filesystem) — `src/experiments/common.py`.

## 8. The factory pattern (strategy = pluggable object)

Merge strategies implement one protocol; the caller asks for a named one.

```python
# src/merging/protocol.py
class MergeStrategy(ABC):
    def weight(self, similarity): ...
    def fuse(self, w1, w2, alpha): ...

# src/merging/factory.py  get_merger("weighted" | "uniform" | "slerp", ...)
```

This is what makes phase 4's apples-to-apples comparison trivial: swap the
strategy object, keep the same baseline model and pair.

## 9. Caching baselines (reproducibility across runs)

`setup()` trains a model ONCE, saves `state_dict`, reloads on later runs:

```python
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
else:
    train_model(model, loaders, cfg)          # deterministic: set_seed first
    torch.save(model.state_dict(), model_path)
```

`weights_only=True` = no arbitrary code execution from a checkpoint file.
Different seeds cache under different filenames (`baseline_*_seed{seed}.pt`);
`scripts/reproduce.py` just re-seeds and re-runs the same pipeline.

## 10. The seed discipline

`utils/seed.py:set_seed` fixes python/torch/numpy + `torch.backends.cudnn`:
without it, "same seed" gives different weights → different evidence. Every
script calls it once at the top. The multi-seed mean±std in the report
(`seeds = (42, 7, 2026)`) is only meaningful because the per-seed runs are
otherwise identical.

## 11. Snapshot metrics with loaders, not magic

`summarize(model, loader, ...)` runs the model over batches and buckets every
report number (params, flops, size MB, latency ms, sparsity, layers, acc) into
one dict. Phases 6/6b slice these into before/after rows and serialize CSV +
JSON with a single `snap()` helper — the artifacts table and the typst
report consume the *same* numbers, so the PDF can't drift from the code.

## 12. Test anatomy in this repo

`tests/` mirror `src/` one-to-one. Fast suite (~5 s) — models are tiny
(`hidden_dims=(12, 12)`), loaders are 64-sample toys. Three patterns to
copy:

```python
# 1. fixture-scoped, real modules:  tests/conftest.py
@pytest.fixture
def trained_toy_model():  ...          # tiny train, ~1 s

# 2. property tests with assertions that document WHY
assert slerp_mid_norm == approx(1.0)   # "on the sphere"

# 3. known-value tests (regression pins)
assert gaussian_mi(x, y) == approx(-0.5*log(1 - 0.8**2), abs=0.06)
```

Run everything with `make verify` before any push (pytest + smoke).

---

## The mental map (what to read when)

| Concept (this doc)         | file to open                              |
|----------------------------|-------------------------------------------|
| 1 modules / parameters     | `src/models/mlp.py`, `residual_mlp.py`    |
| 2 forward trace / plain    | `src/pruning/lrp.py`                      |
| 3 activations              | `src/utils/activations.py`                |
| 4 numpy↔torch              | `src/pruning/pruner.py`                   |
| 5 masking                  | `src/pruning/pruner.py`                   |
| 6 diffusion                | `src/similarity/diffusion.py`             |
| 7 config                   | `src/config/default.py`, `common.py`      |
| 8 factory                  | `src/merging/*`, `similarity/factory.py`  |
| 9 caching baselines        | `src/experiments/common.py`               |
| 10 seeds                   | `src/utils/seed.py`, `scripts/reproduce.py` |
| 11 snapshot metrics        | `src/evaluation/metrics.py`               |
| 12 tests                   | `tests/`, `conftest.py`                   |