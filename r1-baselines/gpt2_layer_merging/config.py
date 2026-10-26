"""Experiment configuration."""

import torch

MODEL_ID = "openai-community/gpt2"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment size: "quick" or "full"
MODE = "quick"

if MODE == "quick":
    FIT_TOKENS = 4096
    EVAL_TOKENS = 10000
    RANDOM_RUNS = 3
    LATENCY_REPEATS = 10
else:
    FIT_TOKENS = 20000
    EVAL_TOKENS = 50000
    RANDOM_RUNS = 5
    LATENCY_REPEATS = 30

EVAL_BLOCK_SIZE = 512
STATE_CHUNK_SIZE = 256
RIDGE = 1e-4

# GPT-2 has 12 Transformer blocks, so these are (0,1), ..., (10,11).
PAIR_STARTS = list(range(11))

RUN_DIRECT_SWEEP = True
TEST_ALL_PAIRS = False

RESULTS_DIR = "/content"
