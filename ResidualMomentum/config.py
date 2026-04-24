from pathlib import Path

# Paths
OUTPUT_DIR = Path("outputs")
PLOT_DIR = Path("plots")

# Dataset
DEFAULT_CHARS = ["LME", "OP", "Investment"]
DEFAULT_TAU = 0.5

# Training
N_TRAIN_VALID = 360
CV_N = 3
ROLLING_WINDOW = 120

# ===== AP-PRUNING (R STYLE) =====
AP_LAMBDA0_GRID = [0.0, 0.5, 1.0, 2.0, 5.0]
AP_LAMBDA2_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
AP_K_MIN = 5
AP_K_MAX = 50
AP_PORT_N = 40   # EXACT K (no fallback)

# ===== TRANSACTION COST =====
TC_COST = 0.0025
TC_LAMBDA_L2 = 1e-3
TC_LAMBDA_TC = 0.0025
TC_ETA = 1.0
TC_LONG_ONLY = False

USE_STOCK_LEVEL_TURNOVER = True