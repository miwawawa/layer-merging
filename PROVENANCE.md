# Research code provenance

The current repository presents a GPT-2 layer-merging investigation in two stages:

- [`r1-baselines/`](r1-baselines/) contains the initial GPT-2 direct-removal, weight-average, and fitted affine experiments, including a modularized package from the collaborative project.
- [`r2-learned/Hiroyoshi/`](r2-learned/Hiroyoshi/) contains Hiroyoshi's nonlinear replacement-block experiments.
- [`r2-learned/Amrita/`](r2-learned/Amrita/) contains Amrita's learned weight-merger notebooks and reported results.

**Source repository:** [sumitjhaa/gpt2-layermerging](https://github.com/sumitjhaa/gpt2-layermerging). The source repository is left unchanged. The copied files were selected and renamed for navigation; their original development is credited to the researchers who produced them. GitHub commit attribution for the restructuring is distinct from original authorship of each experiment.

The initial baseline research is retained for its scientific context. Other researchers' separate experimental branches and unrelated reports have been omitted from the current working tree. Those historical sources remain available in the original collaborative repository and in the commit history of this repository.

### File naming

- The initial baseline notebook is published as `r1-baselines/gpt2_affine_sweep.ipynb`.
- The two main nonlinear notebooks are `r2-learned/Hiroyoshi/single_merge.ipynb` and `multi_merge.ipynb`.
- The weight-merger notebooks and result documents are in `r2-learned/Amrita/`.

Reorganization does not imply a new independent study or that every file was written by the same researcher.
