# Layer-merging workflow. From a fresh clone run:
#     make setup env test smoke phases run-notebooks
# (or simply: make run-all)

PYTHON ?= python3
SCRIPTS := scripts

.PHONY: help setup env test verify smoke phases reproduce run-notebooks report run-all clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Install runtime + dev dependencies from requirements.txt
	$(PYTHON) -m pip install -r requirements.txt

env: ## Create .env from the template (does not overwrite an existing one)
	cp -n .env.example .env || true
	@test -f .env && echo ".env ready" || echo "WARNING: .env already exists — edit it manually"

test: ## Run the pytest suite (fast, toy-sized models — no MNIST download)
	$(PYTHON) -m pytest

smoke: ## Quick module-level smoke test (no MNIST needed)
	$(PYTHON) $(SCRIPTS)/smoke_test.py

verify: test smoke ## Fast gate: unit tests + smoke, run before every commit/push

phases: ## Full reproduction: run all eight experiment phases on MNIST
	$(PYTHON) $(SCRIPTS)/run_phases.py

reproduce: ## Multi-seed reproduction (mean±std evidence for the report)
	$(PYTHON) $(SCRIPTS)/reproduce.py

run-notebooks: ## Execute every notebook end-to-end (verifies the Colab path)
	$(PYTHON) $(SCRIPTS)/run_notebooks.py notebooks

report: ## Compile the printable report (typst) from artifacts + paper/cs613_report.typ
	typst compile --root . paper/cs613_report.typ paper/CS613_Layer_Merging_Report.pdf
	@echo "report -> paper/CS613_Layer_Merging_Report.pdf"

run-all: env test smoke phases run-notebooks ## Everything: env, tests, smoke, phases, notebooks
	@echo "ALL DONE ✔"

clean: ## Remove generated data/artifacts and caches
	rm -rf data artifacts notebooks/data notebooks/artifacts .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "cleaned"