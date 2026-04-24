from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR_ENV = os.getenv("DATA_DIR")
if DATA_DIR_ENV:
    DATA_DIR = Path(DATA_DIR_ENV).expanduser().resolve()
else:
    DATA_DIR = REPO_ROOT / "Data"

TRIPLE_SORT_DIR = REPO_ROOT / "TripleSort"

CHUNK_DIR = DATA_DIR / "data_chunk_files_quantile"
FACTOR_DIR = DATA_DIR / "factor"

TS32_DIR = TRIPLE_SORT_DIR / "ts_portfolio_py"
TS64_DIR = TRIPLE_SORT_DIR / "ts64_portfolio_py"

OUTPUT_DIR = TRIPLE_SORT_DIR / "outputs"

DEFAULT_CHARS = ["LME", "OP", "Investment"]
DEFAULT_Y_MIN = 1964
DEFAULT_Y_MAX = 2016

# Triple-sort bucket specs: TS32 = (2,4,4), TS64 = (4,4,4).
DEFAULT_N_BINS = (2, 4, 4)

# Optimizer / backtest parameters (Model_Versions.pdf)
N_TRAIN_VALID = 360
CV_N = 3

STATIC_LAMBDA_L1 = 0.0
STATIC_LAMBDA_L2 = 1e-3
STATIC_MU0 = None
LONG_ONLY = True

ROLLING_WINDOW = 120
TC_COST = 0.0025
TC_LAMBDA_L1 = 0.0
TC_LAMBDA_L2 = 1e-3
TC_LAMBDA_TC = 0.0025
TC_MU0 = None

