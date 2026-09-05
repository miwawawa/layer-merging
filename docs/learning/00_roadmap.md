# Learning roadmap

> How to go from "basic linear algebra" to "I can read the three papers and
> re-derive their numbers" — in one well-ordered path. Every doc in
> `docs/learning/` is self-contained and (where possible) walks a *worked
> example with this repo's actual numbers.

## Learning path (read in order)

| Step | Document                                        | You learn                                                    | Feeds                                   |
|------|-------------------------------------------------|--------------------------------------------------------------|-----------------------------------------|
| 0    | `docs/learning/01_math_primer.md`               | Vectors, matrices, linear layers, probability, entropy, covariance, Gaussian/MI, eigenvalues/diffusion maps, convex combinations, SLERP, model-compression metrics | everything        |
| 1    | `docs/papers/01_mka_summary.md`                 | Paper 1 (MKA layer merging) — what & why                     | `src/merging/`, `src/similarity/`       |
| 2    | `docs/learning/02_mka_calculations.md`          | Paper 1, step-by-step numbers (diffusion → NMI → α → fuse)   | `phase2`, `phase3`, `phase6`            |
| 3    | `docs/papers/02_com_summary.md`                 | Paper 2 (CoM chain-of-merging) — what & why                  | `src/merging/slerp.py`                  |
| 4    | `docs/learning/03_com_calculations.md`          | Paper 2, step-by-step numbers (pairing, dynamic weight, SLERP)| `phase4`                                |
| 5    | `docs/papers/03_sparc3_summary.md`              | Paper 3 (SparC3 LRP pruning) — what & why                    | `src/pruning/`                          |
| 6    | `docs/learning/04_sparc3_calculations.md`       | Paper 3, step-by-step numbers (LRP rule → aggregation → mask)| `phase5`, `phase6`                      |
| 7    | `docs/learning/05_related_work_calculations.md` | Other approaches + why we picked these three                 | report related-work section             |
| 8    | `docs/learning/06_code_guide.md`                | The *coding concepts* needed to implement all of the above   | `src/**`                                |
| 9    | `docs/learning/07_code_tour.py`                 | A guided walk through the repository file-by-file            | any reader                             |

## How the pieces connect (one mental model)

```
        one MLP per architecture
   residual  -----------------------  plain
      |  h = h + act(W h)               |  h = relu(W h + b)
      |  (redundant hidden layers)      |  (linear chain)
      |                                 |
      |  phase 2: measure redundancy    |  phase 5: measure weight importance
      |  Gaussian NMI on activations    |  LRP relevance R_{i<-j}
      |  S_lm in [0,1]                  |  rank, mask the weakest
      |                                 |
      |  phase 3/4: fuse pairs          |  phase 6 track B: prune @ 50%
      |  theta_c = a theta_l+(1-a)S     |  exact (plain graph)
      |  weighted / uniform / SLERP     |
      |                                 |
      +---------- phase 6 ------------+
      track A: residual merge (MKA)   |  track B: plain prune (SparC3)
      two complementary, honest results; phase 6b shows why ONE model
      can't do both (merge collapses plain; LRP undefined on residual)
```