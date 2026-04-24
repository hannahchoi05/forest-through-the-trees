# Residual Momentum + AP-Trees + Transaction-Cost-Aware Portfolio Construction

## Overview

This project replicates and extends an AP-tree portfolio construction method by:

1. Replicating AP-pruning from the paper  
2. Adding residual momentum tilt within nodes  
3. Converting SDF weights into tradable portfolios  
4. Incorporating transaction-cost-aware optimization  

---

## Methods

We implement four portfolio construction variants:

### A1 — AP-pruning (static, no transaction cost)

- Faithful implementation of AP-pruning
- Uses LASSO path to select exactly K portfolios
- Hyperparameters selected via validation Sharpe (SDF-based)

The paper produces **SDF weights**, which are not directly tradable.  
We convert them into investable weights:

```
w_trade = w_sdf / sum(|w_sdf|)
```

No transaction costs applied.

---

### A2 — AP-pruning (static + stock-level transaction cost)

- Same portfolio as A1
- Applies transaction costs ex-post

Turnover:

```
turnover = sum_i |w_i,t - w_i,t-1|
```

Transaction cost:

```
cost = 25 bps × turnover
```

---

### B — Rolling TC-aware optimization (portfolio-level)

- Rolling mean-variance optimization
- Objective:

```
min_w 0.5 w'Σw - η w'μ + 0.5 λ2 ||w||² + λ_tc ||w - w_prev||₁
```

- Turnover computed at portfolio weight level

---

### C — Rolling TC-aware optimization (stock-level)

- Same as B, but turnover computed at stock level:

```
turnover = sum_i |W_stock_i,t - W_stock_i,t-1|
```

- Uses AP-tree mapping from portfolio weights → stock weights

---

## Key Concept: SDF vs Tradable Portfolio

The AP-pruning method produces **SDF (stochastic discount factor) weights**, not directly investable portfolios.

Paper normalization:

```
w_sdf = b / |sum(b)|
```

This does **not control leverage**.

### Our extension

We construct tradable weights:

```
w_trade = w_sdf / sum(|w_sdf|)
```

This ensures:

- controlled leverage  
- stable returns  
- meaningful cumulative wealth  

---

## Pipeline

Run:

```
python run_all.py
```

### Steps

1. Load stock-level data  
2. Compute residual momentum  
3. Build AP-tree candidate portfolios  
4. Generate candidate return matrix (~3000 portfolios)  
5. Run optimization (A1/A2/B/C)  
6. Backtest and generate metrics + plots  

---

## Outputs

All outputs are saved under:

```
outputs/<chars>/
```

### Backtests

```
backtest_A1_*.csv
backtest_A2_*.csv
backtest_B_*.csv
backtest_C_*.csv
backtest_comparison.csv
backtest_comparison_with_wealth_drawdown.csv
```

### Metrics

```
summary_metrics_comparison.csv
```

Includes:

- Sharpe (gross/net)  
- turnover  
- drawdown  
- terminal wealth  

---

### Plots

```
outputs/<chars>/plots/
```

- Plot A — cumulative gross returns  
- Plot B — cumulative net returns  
- Plot C — gross vs net  
- Plot D — drawdown  
- Plot E — turnover  
- Plot F — summary metrics  

---

## Configuration

Key parameters in `config.py`:

```python
AP_PORT_N = 40
ROLLING_WINDOW = 120

TC_COST = 0.0025
TC_LAMBDA_TC = 0.0025
TC_LAMBDA_L2 = 1e-3

TC_ETA = 1.0
TC_LONG_ONLY = False
```

---

## Notes

- AP-pruning is replicated faithfully for model selection  
- Trading performance uses normalized weights (`w_trade`)  
- Transaction costs are applied only in A2, B, and C  
- Rolling methods (B, C) provide more realistic execution dynamics  

---

## Summary

- AP-pruning identifies sparse, high-signal portfolios  
- Residual momentum enhances signal strength  
- Raw SDF weights are not tradable without normalization  
- Transaction costs significantly impact performance  
- Stock-level turnover is more realistic than portfolio-level turnover  
```
