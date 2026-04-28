from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "Data"
CHUNK_DIR = DATA_DIR / "data_chunk_files_quantile"
FACTOR_DIR = DATA_DIR / "factor"

OUTPUT_DIR = Path("outputs")
PLOT_DIR = Path("plots")

DEFAULT_CHARS = ["LME", "OP", "Investment"]
DEFAULT_TAU = 0.5
DEFAULT_Y_MIN = 1964
DEFAULT_Y_MAX = 2016
DEFAULT_TREE_DEPTH = 4
DEFAULT_Q_NUM = 2

MOM_LOOKBACK = 12
MOM_SKIP_RECENT = 2
BETA_WINDOW = 36

N_TRAIN_VALID = 360
CV_N = 3
ROLLING_WINDOW = 120

import numpy as np
AP_LAMBDA0_GRID = list(np.round(np.arange(0.0, 0.95, 0.05), 10))  # 19 values {0, 0.05, ..., 0.90}
AP_LAMBDA2_GRID = list(np.logspace(-5.0, -8.0, 13))                 # 13 log-spaced values 10^{-5} to 10^{-8}
AP_K_MIN = 5
AP_K_MAX = 50
AP_PORT_N = 40

TC_COST = 0.0025
TC_LAMBDA_L2 = 1e-6
TC_LAMBDA_TC = 0.0025
TC_ETA = 1.0
TC_LONG_ONLY = False

USE_STOCK_LEVEL_TURNOVER = True
DEDUPLICATE_CANDIDATES = True
RUN_FULL_TREE_SET = True