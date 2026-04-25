from pathlib import Path
import numpy as np
import os

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

DATA_DIR_ENV = os.getenv("DATA_DIR")
if DATA_DIR_ENV:
    DATA_DIR = Path(DATA_DIR_ENV).expanduser().resolve()
else:
    DATA_DIR = REPO_ROOT / "Data"

APTREES_DIR = REPO_ROOT / "APTrees"
CHUNK_DIR = DATA_DIR / "data_chunk_files_quantile"
FACTOR_DIR = DATA_DIR / "factor"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

for _p in [OUTPUT_DIR, PLOT_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

DEFAULT_CHARS = ["LME", "OP", "Investment"]
DEFAULT_Y_MIN = 1964
DEFAULT_Y_MAX = 2016
DEFAULT_TREE_DEPTH = 4
DEFAULT_Q_NUM = 2

# tau=0 => pure value-weighted AP-Trees, no residual-momentum tilt
DEFAULT_TAU = 0.0

# Residual momentum (unused at tau=0; kept so tree_portfolios.py signature is satisfied)
MOM_LOOKBACK = 12
MOM_SKIP_RECENT = 1
BETA_WINDOW = 36

# ── Static AP-pruning optimizer (Variants A1, A2) ──────────────────────────
N_TRAIN_VALID = 360    # first 30 years: 1964-1993 (240 train + 120 valid)
CV_N = 3               # validation is last N_TRAIN_VALID/CV_N months

# lambda0 grid: mean shrinkage toward cross-sectional average (Table D.2 of paper)
AP_LAMBDA0_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25,
                   0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                   0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# lambda2 grid: ridge / variance shrinkage (Table D.2 of paper)
AP_LAMBDA2_GRID = list(10.0 ** np.arange(-5.0, -8.25, -0.25))

AP_K_MIN = 10     # minimum number of selected portfolios
AP_K_MAX = 40     # maximum number of selected portfolios
AP_PORT_N = 40    # target portfolio count for reporting

# ── Rolling TC-aware optimizer (Variants B, C) ────────────────────────────
ROLLING_WINDOW = 120   # 10-year rolling estimation window
TC_COST = 0.0025       # 25 bps per unit of turnover
TC_LAMBDA_L2 = 1e-3
TC_LAMBDA_TC = 0.0025
TC_ETA = 0.15          # mean shrinkage in rolling window (analog of lambda0)
TC_LONG_ONLY = True

# ── Tree-building flags ───────────────────────────────────────────────────
DEDUPLICATE_CANDIDATES = True
RUN_FULL_TREE_SET = True
USE_STOCK_LEVEL_TURNOVER = True