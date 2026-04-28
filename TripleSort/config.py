from __future__ import annotations

import os
import numpy as np
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
# Default to TS32 (paper baseline triple-sort basis).
DEFAULT_N_BINS = (2, 4, 4)

# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
# Training:   1964-1983  → 240 months
# Validation: 1984-1993  → 120 months
# Testing:    1994-2016  → 276 months
# N_TRAIN_VALID = 240 + 120 = 360
N_TRAIN_VALID = 360
CV_N = 3  # last fold only (matching R code: only i = cvN = 3 is used)

# ---------------------------------------------------------------------------
# LARS / AP-Pruning hyperparameter grids  (paper Section 3 / Appendix A.4)
# ---------------------------------------------------------------------------
# mu0 grid: 19 values {0, 0.05, ..., 0.90}
STATIC_MU0_GRID = list(np.round(np.arange(0.0, 0.95, 0.05), 10))

# lambda_l2 grid: 13 log-spaced values from 10^{-5.0} to 10^{-8.0}
# FIX: was incorrectly set to 1e-3, which is orders of magnitude too large.
STATIC_LAMBDA_L2_GRID = list(np.logspace(-5.0, -8.0, 13))

# K: number of nonzero portfolios to select via LARS path
STATIC_K = 40          # target K reported in paper
STATIC_K_MIN = 10      # minimum K to consider during path walk
STATIC_K_MAX = 40      # maximum K to consider during path walk

# Kept for rolling/TC variants (not LARS-based, so a single lambda_l2 is fine;
# use the geometric midpoint of the paper grid as a reasonable default).
LONG_ONLY = True

ROLLING_WINDOW = 120
TC_COST = 0.0025
TC_LAMBDA_L2 = 1e-6   # midpoint of paper's log grid (was 1e-3, now corrected)
TC_LAMBDA_TC = 0.0025
TC_ETA = 1.0
TC_LONG_ONLY = False
TC_MU0 = None
