from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR_ENV = os.getenv("DATA_DIR")

if DATA_DIR_ENV:
    DATA_DIR = Path(DATA_DIR_ENV).expanduser().resolve()
else:
    DATA_DIR = REPO_ROOT / "Data"

RESIDUAL_MOMENTUM_DIR = REPO_ROOT / "ResidualMomentum"
CHUNK_DIR = DATA_DIR / "data_chunk_files_quantile"
FACTOR_DIR = DATA_DIR / "factor"
OUTPUT_DIR = RESIDUAL_MOMENTUM_DIR / "outputs"
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

DEFAULT_TAU = 0.50

MOM_LOOKBACK = 12
MOM_SKIP_RECENT = 1
BETA_WINDOW = 36

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

DEDUPLICATE_CANDIDATES = True
RUN_FULL_TREE_SET = False

# New: use actual underlying stock-weight turnover instead of portfolio-weight proxy.
USE_STOCK_LEVEL_TURNOVER = True