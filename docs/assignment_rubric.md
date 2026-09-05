# Assignment Rubric → Where Things Live

## I. Report (50%)

| # | Section | Weight | Where we get content |
|---|---------|--------|----------------------|
| 1 | Introduction (Motivation / NLP-relatn / Problem type) | 10% | `docs/papers/*_summary.md` |
| 2 | Related Work (SOTA, baselines, prior results table) | 10% | `docs/paper_comparison.md`, table in `01_mka_summary.md` |
| 3 | Datasets (availability, size, labels, bias) | 10% | `data/mnist.py` + MNIST stats |
| 4 | Experimental Plan (split, tuning, metric, demo) | 10% | `docs/phase_plan.md`, `evaluation/metrics.py` |
| 5 | Project Management (Gantt, compute, risk, roles) | 10% | Gantt in Phase 6, risk table in `docs/phase_plan.md` |

## II. Experiment (50%)

| # | Section | Weight | Where we get content |
|---|---------|--------|----------------------|
| 1 | Reproduce results (no closed-source big LLMs) | 20% | `experiments/phase2–6` |
| 2 | Report findings/pattern | 10% | outputs of `phase6_pipeline.py` |
| 3 | Viva (report + code execution) | 20% | understand `docs/architecture.md` fully |

## Team contribution
Dedicated section at report end — assign by real work per phase.

---
## Deliverables checklist (for submission)
- [ ] GitHub repo public with all `src/`, `notebooks/`, `docs/`
- [ ] PDF with screenshots of every notebook step
- [ ] Link to repo inside the PDF