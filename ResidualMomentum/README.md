# Residual Momentum AP-Tree Ablation

The code produces exactly the two methods:

1. **Static paper-style optimizer + residual momentum tilt**
   - Reconstructs the full AP-tree candidate set.
   - Applies residual momentum tilt inside each AP-tree node.
   - Uses a fixed train/validation/test split, with no rolling window.
   - Fits one set of portfolio weights on the first `N_TRAIN_VALID` months.
   - Applies those fixed weights to the test period.

2. **Rolling TC-aware optimizer + residual momentum tilt**
   - Uses the same residual-momentum-tilted AP-tree candidate returns.
   - Estimates mean/covariance on a rolling `ROLLING_WINDOW`.
   - Adds transaction-cost turnover penalty `lambda_tc * ||w_t - w_{t-1}||_1`.
   - Computes gross returns, turnover, costs, and net returns.

## Run

```bash
cd "ResidualMomentum"
pip install -r requirements.txt
python run_all.py
```

## Main settings

Edit `config.py`:

```python
DEFAULT_CHARS = ["LME", "OP", "Investment"]
DEFAULT_TREE_DEPTH = 4
DEFAULT_TAU = 0.50
N_TRAIN_VALID = 360
ROLLING_WINDOW = 120
TC_COST = 0.0025
TC_LAMBDA_TC = 0.0025
```

## Outputs

All outputs are saved under:

```text
ResidualMomentum/outputs/<chars>/
ResidualMomentum/outputs/plots/<chars>/
```

Key files:

- `candidate_returns_full_ap_tree_tau_*.csv`
- `tilted_candidate_matrix_tau_*.csv`
- `backtest_static_paper_style_plus_residual_momentum_tilt.csv`
- `backtest_rolling_tc_aware_plus_residual_momentum_tilt.csv`
- `backtest_comparison.csv`
- `summary_metrics_comparison.csv`
- Plots in PNG files.
