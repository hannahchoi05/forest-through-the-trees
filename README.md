# Forest Through the Trees: Building Cross-Sections of Stock Returns

**FIN 580 / ECO 480 · Princeton University**  
Abby Pham · Hannah Choi · Ram Narayanan

## Abstract
This project replicates and extends the frameworks of [Bryzgalova, Pelger & Zhu (2023)](https://doi.org/10.1111/jofi.13250) — *Journal of Finance*, which introduces Asset Pricing Trees (AP-Trees) as a flexible alternative to traditional sorting-based methods for constructing cross-sections of stock returns. While conventional double and triple sorting impose coarse, unconditional partitions that fail to capture nonlinear interactions, AP-Trees use sequential conditional splits and global pruning to construct more informative basis assets for spanning the stochastic discount factor (SDF). We first confirm that AP-Trees outperform traditional sorting methods in terms of Sharpe ratios and diversification, and then extend it in two directions not covered by the authors: 
1. We modify the framework's static SDF estimation into an implementable trading setting by incorporating monthly-rebalancing portfolio with explicit stock-level turnover penalties (25 bps).
2. We further examine a residual momentum augmentation, tilting each AP-tree node's value-weighted returns with an exponential tilt toward recent idiosyncratic winners, to enhance cross-sectional return predictability.

Our results show that although AP-Trees provide a richer representation of the cross-section, their strong statistical performance does not directly translate into implementable strategies without accounting for trading frictions. Incorporating transaction costs reduces returns but improves realism, while signal-based enhancements introduce a trade-off between performance and turnover, highlighting the importance of jointly evaluating asset pricing models and portfolio construction constraints.

## Data
Our analysis follows the data construction in (Bryzgalova et al.). We use monthly U.S. equity return data from CRSP over the sample period 1964–2016, along with firm characteristics constructed from CRSP/Compustat following the Kenneth French Data Library conventions. Excess returns are computed using the one-month Treasury bill rate as the risk-free rate.

For benchmarking, we include the S&P 500 index, proxied using adjusted close prices from Yahoo Finance, and a hedge fund long/short equity benchmark (CS L/S Equity Hedge Fund Main Index) obtained from [HFR](https://hedgeindex.com/indexes/en/HEDG_LOSHO/overview). These benchmarks provide comparisons to both passive market exposure and actively managed market-neutral strategies.

## Code 
The folders `triple_sort`, `ap_trees`, and `residual_momentum` are each self-contained and follow the same internal structure. 

We load the CRSP data chunks and benchmarks via Yahoo Finance via `data_io.py` and then construct the AP-tree and node return computation using `tree_portfolios.py`. We then implement static AP-pruning using LARS (Least Angle Regression) to select a sparse set of AP-tree candidate portfolios that maximize the out-of-sample SDF Sharpe ratio, and we implement the rolling TC-aware optimizer, which re-estimates portfolio weights each month using a mean-variance objective that explicitly penalizes stock-level turnover (`optimizer.py`). For each stock, we estimate a rolling 36-month market model, compute idiosyncratic residuals, and compound them from t-12 to t-2 to produce a cross-sectionally standardized signal that drives the exponential reweighting within each node (`residual_momentum.py`). We then take the backtest output dataframe and compute portfolio performance statistics (annualized gross and net Sharpe, average turnover, max drawdown, and terminal wealth) to get what we need for plotting (`metrics.py`). All of this is contained inside `run_all.py` which serves as an entry point to run the full pipeline and write outputs. 

Each module is run independently from its own folder.
```bash
# Triple sort replication (TS32 and TS64)
cd triple_sort
pip install -r requirements.txt
python run_all.py
 
# AP-Trees replication (all 36 cross-sections)
cd ap_trees
pip install -r requirements.txt
python run_all.py
 
# AP-Trees + Residual Momentum extension
cd residual_momentum
pip install -r requirements.txt
python run_all.py
```

Each `run_all.py` runs four strategy variants labeled A1-C:

| Label | Description |
|---|---|
| **A1** | Static AP-pruning, no transaction costs — replication baseline |
| **A2** | Same static weights as A1, with stock-level TC applied ex-post |
| **B** | Rolling TC-aware optimizer, portfolio-level turnover penalty |
| **C** | Rolling TC-aware optimizer, stock-level turnover penalty (main result) |

Each `run_all.py` produces outputs in `<module>/outputs/LME_OP_Investment/` and `backtest/`
(some of these are not included in this repository due to file size). 

In `Plots/`, we recreate Figures 7 and 8 of the original paper, which describes the t-stat of the robust SDF alpha and the R^2 within cross-sections, with respect to the Fama-French 5 factor model. The pipeline first slices raw characteristic data into per-triple panels, orchestrates the full 36-cross-section backtest runs, then aggregates results and renders the figures. To run:

```bash
cd plots
python Code/slice_quantile_panels.py \
    --raw-dir /path/to/characteristics \
    --out-dir ../data/data_chunk_files_quantile
python Code/orchestrate_36.py --repo-root ..
python Code/run_figure_7.py \
    --backtest-root Code/backtest_results \
    --factors ../data/factor/tradable_factors.csv \
    --out-dir Code/figures
python Code/run_figure_8.py \
    --backtest-root Code/backtest_results \
    --factors ../data/factor/tradable_factors.csv \
    --out-dir Code/figures
```

## Paper
The full paper with results can be found [here](paper.pdf).


