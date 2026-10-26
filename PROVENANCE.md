# Team research and source provenance

This repository documents collaborative research on GPT-2 Transformer layer compression. **All experiments presented here are part of the team's joint research effort.** Folder names distinguish **methods**, not individual researchers, owners, or exclusive contributions.

## Research stages

- [R1 — Baselines and affine merging](r1-baselines/): direct layer removal, parameter averaging, and fitted affine approximations.
- [R2 — Nonlinear merging](r2-learned/nonlinear-merge/): teacher-guided nonlinear replacement-block experiments, multi-step compression, and the final results.
- [R2 — Learned weight-space merging](r2-learned/weight-merge/): learned weight corrections for adjacent block merging, notebooks, and experimental results.

The code and results were collected from the team's collaborative source repository, [sumitjhaa/gpt2-layermerging](https://github.com/sumitjhaa/gpt2-layermerging), and the team's research summaries. The original source repository has not been changed. Some files have been renamed and reorganized for clarity, but their technical content remains otherwise unchanged.

The GitHub commit author identifies the maintainer who reorganized this repository; it does not indicate sole authorship of the research. Individual contribution records, when relevant, can be found in the original collaborative source history.

## Final results artifact

The [final results PDF](r2-learned/nonlinear-merge/Final_Results.pdf) reproduces all numerical tables from the team's supplied one-page result sheet. The file is a **reformatted PDF recreation**, not a byte-for-byte copy of the uploaded document; the values, methods, compression levels, and reported PPLs are preserved.
