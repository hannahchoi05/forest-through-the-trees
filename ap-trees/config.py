from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR_ENV = os.getenv("DATA_DIR")

if DATA_DIR_ENV:
    DATA_DIR = Path(DATA_DIR_ENV).expanduser().resolve()
else:
    DATA_DIR = REPO_ROOT / "Data"

APTREES_DIR = REPO_ROOT / "APTrees"
CHUNK_DIR = DATA_DIR / "data_chunk_files_quantile"
FACTOR_DIR = DATA_DIR / "factor"
OUTPUT_DIR = APTREES_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

for _p in [OUTPUT_DIR, PLOT_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

FEATS_LIST = [
    "LME", "BEME", "r12_2", "OP", "Investment",
    "ST_Rev", "LT_Rev", "AC", "IdioVol", "LTurnover",
]

DEFAULT_CHARS = ["LME", "OP", "Investment"]
DEFAULT_Y_MIN = 1964
DEFAULT_Y_MAX = 2016
DEFAULT_TREE_DEPTH = 4
DEFAULT_Q_NUM = 2

# Pure AP-Trees: no residual-momentum tilt. tau=0 => tilt_stock_w == base_stock_w,
# so the tilt_ and baseline_ columns in tree_portfolios collapse to the same thing.
# We then read the baseline_ columns to keep the naming explicit.
DEFAULT_TAU = 0.0

# Kept for compatibility with tree_portfolios signature; unused at tau=0
MOM_LOOKBACK = 12
MOM_SKIP_RECENT = 1
BETA_WINDOW = 36

# ---- Static optimizer (A1, A2) ----
N_TRAIN_VALID = 360
CV_N = 3
STATIC_LAMBDA_L1 = 0.0
STATIC_LAMBDA_L2 = 1e-3
STATIC_MU0 = None
LONG_ONLY = True

# ---- Rolling TC-aware optimizer (B, C) ----
ROLLING_WINDOW = 120
TC_COST = 0.0025
TC_LAMBDA_L1 = 0.0
TC_LAMBDA_L2 = 1e-3
TC_LAMBDA_TC = 0.0025
TC_MU0 = None

DEDUPLICATE_CANDIDATES = True

# Set True for the paper-faithful full set of 3^4 = 81 trees.
# Set False for a single-tree smoke test (fast; NOT for final results).
RUN_FULL_TREE_SET = True

# Not used in pure AP-Trees (kept so imports from optimizer still work cleanly)
USE_STOCK_LEVEL_TURNOVER = True