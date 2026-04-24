from pathlib import Path

# Change this if your repo path differs.
REPO_ROOT = Path(r"C:\Users\hongv\OneDrive\Tài liệu\forest-through-the-trees")

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

# Main AP-tree experiment.
DEFAULT_CHARS = ["LME", "OP", "Investment"]
DEFAULT_Y_MIN = 1964
DEFAULT_Y_MAX = 2016
DEFAULT_TREE_DEPTH = 4
DEFAULT_Q_NUM = 2

# Residual momentum tilt strength. tau=0 exactly recovers value-weighted node returns.
DEFAULT_TAU = 0.50

# Residual momentum settings.
MOM_LOOKBACK = 12
MOM_SKIP_RECENT = 1
BETA_WINDOW = 36

# Paper-style static optimizer settings.
# Original R default is passed into AP_Pruning as n_train_valid; set here explicitly.
N_TRAIN_VALID = 360
CV_N = 3
STATIC_LAMBDA_L1 = 0.0
STATIC_LAMBDA_L2 = 1e-3
STATIC_MU0 = None
LONG_ONLY = True

# Transaction-cost rolling optimizer settings from the PDF framework.
ROLLING_WINDOW = 120
TC_COST = 0.0025
TC_LAMBDA_L1 = 0.0
TC_LAMBDA_L2 = 1e-3
TC_LAMBDA_TC = 0.0025
TC_MU0 = None

# If True, deduplicate candidate portfolios with identical return histories, matching R combine step.
DEDUPLICATE_CANDIDATES = True
