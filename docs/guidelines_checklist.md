# Guidelines Compliance Checklist

Cross-check between the *assignment rubric*, the *three papers*, and what the
repo actually does. Run this list before every demo/submission.

---

## A. Assignment rubric (`docs/assignment_rubric.md`)

### Report (50%)
| Rubric item | Evidence in repo | Status |
|---|---|---|
| §1 Introduction — motivation, NLP-relevance, problem type | `docs/papers/01_mka_summary.md`, `02_com_summary.md`, `03_sparc3_summary.md` (§1 of each) | ✅ |
| §2 Related Work — SOTA, baselines, prior-results table | `docs/paper_comparison.md`, MKA results table; `docs/research_survey.md` §3 | ✅ |
| §3 Datasets — availability, size, labeling, bias | MNIST in `src/data/mnist.py` (60k+10k, 10 classes, class balance ≈ uniform after split → minimal bias); cite source URL | ✅ |
| §4 Experimental Plan — split, tuning, metric, demo | `docs/phase_plan.md` + `src/config/default.py` (val split, seed, hyperparams); metrics in `src/evaluation/metrics.py` | ✅ |
| §5 Project Management — Gantt, compute, risk, roles | Gantt PNG from `src/utils/viz.py` (Phase 6); risk table in `docs/phase_plan.md` | ✅ |
| §6 Team contribution (report end) | Section reserved in report; fill per phase | ⏳ report |

### Experiment (50%)
| Rubric item | Evidence | Status |
|---|---|---|
| Reproduce results (no closed-source big LLMs) | `experiments/phase{1..6}*.py` on from-scratch MNIST MLP + `scripts/smoke_test.py` | ✅ |
| Report a finding / pattern | `phase3` merge-vs-drop curve, `phase5` LRP-vs-baseline, `phase6` two-track pipeline results | ✅ |
| Viva — report + code execution | `pytest` suite green (78 tests) + notebooks run end-to-end (all OK) | ✅ |

### Deliverables (submission)
| Item | Status |
|---|---|
| GitHub repo public: `src/`, `notebooks/`, `docs/`, tests, README | ⏳ (user pushes) |
| PDF with screenshots of each notebook step | 🔨 `paper/CS613_Layer_Merging_Report.pdf` (8 pp, real figures, `make report`); insert literal notebook screenshots |
| PDF links the repo | 🔨 add repo URL in `paper/cs613_report.typ` header |
| Repo installs + runs on a fresh machine (`make setup test run-all`) | 🔨 verify on clean clone |

---

## B. Paper-fidelity checks

### Paper 1 (MKA) — `docs/papers/01_mka_summary.md`
| Check | Repo | Status |
|---|---|---|
| Affinity `W_ij=exp(-‖H_i-H_j‖²/σ²)` | `similarity/diffusion.py` | ✅ |
| `P̃=D^(-½)WD^(-½)` + eigen | `similarity/diffusion.py` | ✅ |
| Diffusion map omits λ₁ (trivial) | `similarity/diffusion.py` | ✅ |
| Gaussian NMI closed-form | `similarity/gaussian_mi.py` (standardize features) | ✅ |
| α = S_lm ∈[0,1] | `merging/weighted.py` + clamp in `merger.py` | ✅ |
| Greedy pair loop ≥ τ | `merging/merger.py` | ✅ |
| Fused `θ̃=αθ_l+(1-α)θ_m` | `merging/weighted.py` | ✅ |
| Extend to residual blocks (Paper-1 §4 style) | `ResidualMLP` in `src/models/residual_mlp.py` | ✅ |
| *Info-bottleneck justification* | treated as theory (§1 of mka summary) — not implemented | ✅ (by design) |

### Paper 2 (CoM) — `docs/papers/02_com_summary.md`
| Check | Repo | Status |
|---|---|---|
| SLERP geodesic formula + linear fallback | `merging/slerp.py` (sinΩ≈0 fallback) | ✅ |
| Layer-wise (per-tensor) SLERP, not global flatten | `merging/slerp.py` fuses each `linear.weight/bias` tensor | ✅ |
| Non-positional pair choice | `merging/merger.py` picks max-similarity pair | ✅ |
| Per-pair dynamic weight | `merging/weighted.py` α from similarity | ✅ |
| *Cross-model* VLM merge | out of scope (we merge layers of one model — same algebra) | ✅ documented |

### Paper 3 (SparC3) — `docs/papers/03_sparc3_summary.md`
| Check | Repo | Status |
|---|---|---|
| LRP `R_{i←j}=(a_i w_ij / z_j) R_j`, ε-stabilized | `pruning/lrp.py` | ✅ |
| Aggregation `R̄_ψk=(1/n)Σₓ R_ψk(x)` | `pruning/importance.py` | ✅ |
| Global ranking, keep top-q | `pruning/pruner.py::prune_model_global` | ✅ |
| Mask `ψ_i → (1-𝟙)ψ_i` | `pruning/pruner.py` | ✅ |
| Magnitude & random baselines | `experiments/phase5_pruning.py` | ✅ |
| *Contrastive Eq.7 (behavior correction)* | documented, not implemented (needs a "bad behavior" dataset) | ✅ by design (documented in `docs/research_survey.md`) |

---

## C. Engineering-policy checks (from latest requirements)
| Requirement | Repo | Status |
|---|---|---|
| No hardcoded values — everything configurable | `src/config/default.py` + `.env` | ✅ |
| One place to control all params | `Config` dataclass loaded from env (constructor wins over env) | ✅ |
| Env used for downloads (MNIST mirror URL, data dir) | `src/data/mnist.py` reads mirror from Config; paths anchored to `PROJECT_ROOT` | ✅ |
| Comments so others understand | docstrings pass (reviewed during refactor) | ✅ |
| Tests for everything | pytest suite — 78 tests green (`make verify`) | ✅ |
| Workflow anyone can run | `Makefile` targets in `README.md` | ✅ |
| `requirements.txt` | pinned (torch, numpy, scipy, matplotlib, nbformat, python-dotenv, pytest) | ✅ |
| `.env.example` documents every override | keys mirror `Config` fields 1:1 | ✅ |

---

## D. Results calculations & representations (report-evidence checks)
| Requirement | Repo | Status |
|---|---|---|
| Reduction quantified beyond params (FLOPs, size, latency) | `evaluation/metrics.py` (`count_flops`, `model_size_bytes`, `inference_latency_ms`) | ✅ |
| Before/after evidence with savings per track | `phase6` → `artifacts/results_table.png` + `final_results.csv/json` | ✅ |
| Per-merge reason recorded (which pair, what α/S) | `merger.merge_log` (full history) + `phase3_merge_curve.json` | ✅ |
| α-ablation shows pair > weight (Phase 4b) | `alpha_sweep.png` + `phase4b_alpha_sweep.json` | ✅ |
| Honest cross-track no-go evidence (Phase 6b) | `hybrid_tracks.png` + `phase6b_hybrid.json` | ✅ |
| Figures for report: merge curve, strategies, pruning curves, similarity grid | `merge_curve.png`, `strategies.png`, `pruning_curves.png`, `similarity_all.png` | ✅ |
| Learning path: math primer → per-paper numbers → code guide | `docs/learning/00_roadmap.md` … `07_code_tour.py` | ✅ |
| Multi-seed mean±std | `scripts/reproduce.py` → `reproduce_summary.csv/json` + `reproduce_table.png` | ✅ |
| Fixed the phase-6 "MKA/slerp" mislabel | pipeline uses weighted (MKA); labels say so | ✅ |
| Paper + methodology + survey docs | `paper.md`, `approach.md`, `other_approaches.md` | ✅ |