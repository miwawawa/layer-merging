# GPT-2 Layer Merging Experiment

This project reorganizes the original Colab-generated Python script into a
small Python package while preserving the experiment structure.

## Project structure

```text
gpt2_layer_merging_project/
├── run.py
├── README.md
└── gpt2_layer_merging/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── data.py
    ├── model.py
    ├── metrics.py
    ├── direct_baselines.py
    ├── state_collection.py
    ├── affine.py
    ├── experiments.py
    ├── analysis.py
    ├── plots.py
    └── pipeline.py
```

## What belongs where

- `config.py` — experiment constants and switches.
- `data.py` — WikiText loading and text preparation.
- `model.py` — GPT-2/tokenizer loading and direct Transformer-block operations.
- `metrics.py` — PPL, parameter count, latency, and model evaluation.
- `utils.py` — reproducibility, memory cleanup, and distribution-preserving shuffling.
- `state_collection.py` — forward hooks and hidden-state collection.
- `affine.py` — affine fitting, four affine conditions, and affine replacement block.
- `direct_baselines.py` — direct GPT-2 Remove/Average/Random-Weight Average experiments.
- `experiments.py` — affine experiments and result summarization.
- `analysis.py` — model selection, comparisons, pivots, and final test evaluation.
- `plots.py` — all plotting code.
- `pipeline.py` — orchestration: runs the whole experiment and saves results.
- `run.py` — minimal executable entry point.

## Run in Colab

Upload/extract the whole folder, then run:

```bash
python run.py
```

Or from the project directory:

```bash
python -m gpt2_layer_merging.pipeline
```

The code expects the same main dependencies as the original:

```bash
pip install torch transformers datasets pandas numpy matplotlib
```

## Changing experiment settings

Edit only `gpt2_layer_merging/config.py`.

For example:

```python
MODE = "quick"
RUN_DIRECT_SWEEP = True
TEST_ALL_PAIRS = False
```

Start with `quick` on a T4. Once the pipeline works end-to-end, change to
`full`.

## Output files

Results are written to `/content` in Colab:

- `affine_all_pairs_raw.csv`
- `affine_all_pairs_summary.csv`
- `best_method_per_pair.csv`
- `multiply_vs_average.csv`
- `direct_all_pairs_raw.csv`
- `direct_all_pairs_summary.csv`
- optionally `test_all_pairs.csv`

## Important design choice

The code keeps the original experimental split:

- fit affine maps on training text,
- compare methods on validation text,
- select the best pair/method using validation,
- evaluate the selected configuration on test.

This avoids moving model-selection decisions onto the test set.
