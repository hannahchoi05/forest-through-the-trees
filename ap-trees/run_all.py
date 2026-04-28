"""
Pure AP-Trees backtest for the Forest Through the Trees project.

Runs the 4 variants specified in the methodology PDF:
  A1: Static optimizer, no transaction costs
  A2: Static optimizer, stock-level transaction costs
  B : Rolling TC-aware optimizer, portfolio-level turnover (for cost accounting)
  C : Rolling TC-aware optimizer, stock-level turnover (for cost accounting)

Output schema matches ResidualMomentum/outputs/*/backtest_comparison.csv:
  date, date_dt, yy, mm, method, gross_ret, turnover_raw, turnover, cost, net_ret
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CHUNK_DIR,
    FACTOR_DIR,
    OUTPUT_DIR,
    DEFAULT_CHARS,
    DEFAULT_Y_MIN,
    DEFAULT_Y_MAX,
    DEFAULT_TREE_DEPTH,
    DEFAULT_Q_NUM,
    DEFAULT_TAU,
    N_TRAIN_VALID,
    CV_N,
    AP_LAMBDA0_GRID,
    AP_LAMBDA2_GRID,
    AP_K_MIN,
    AP_K_MAX,
    AP_PORT_N,
    ROLLING_WINDOW,
    TC_COST,
    TC_LAMBDA_L2,
    TC_LAMBDA_TC,
    TC_ETA,
    TC_LONG_ONLY,
    USE_STOCK_LEVEL_TURNOVER,
    DEDUPLICATE_CANDIDATES,
    RUN_FULL_TREE_SET,
)
from data_io import load_yearly_chunks, load_yahoo_monthly_benchmark
from tree_portfolios import (
    build_all_ap_tree_candidate_returns,
    select_candidate_matrix,
)
from optimizer import ap_pruning_static_optimize, rolling_tc_optimize
from metrics import performance_metrics, add_wealth_drawdown
from plots import make_all_plots


METHOD_A1 = "AP-Trees baseline (static, no TC)"
METHOD_A2 = "AP-Trees static + stock-level TC"
METHOD_B  = "AP-Trees rolling TC-aware + portfolio-level TC"
METHOD_C  = "AP-Trees rolling TC-aware + stock-level TC"


def _load_rf() -> np.ndarray | None:
    """
    Load monthly risk-free rates from rf_factor.csv and convert to decimal.

    R Step3_RmRf_Combine_Trees.R does:
        port_ret[,i] = port_ret[,i] - (rf)/100
    meaning rf_factor.csv stores values as percentage points (e.g. 0.45
    means 0.45% per month). We divide by 100 here to match that convention.
    """
    rf_path = FACTOR_DIR / "rf_factor.csv"

    if not rf_path.exists():
        print(
            f"WARNING: rf_factor.csv not found at {rf_path}. "
            "Using raw returns instead of excess returns.",
            flush=True,
        )
        return None

    rf_raw = pd.read_csv(rf_path, header=None).squeeze().astype(float).to_numpy()

    if float(np.median(np.abs(rf_raw))) < 0.01:
        print(
            f"Loaded rf series ({len(rf_raw)} months) — values appear already in "
            "decimal form, using as-is.",
            flush=True,
        )
        return rf_raw
    else:
        print(
            f"Loaded rf series ({len(rf_raw)} months) — dividing by 100 to convert "
            "from percentage points to decimal.",
            flush=True,
        )
        return rf_raw / 100.0


def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots"
    stock_weights_dir = out_dir / "stock_weights_by_month"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    stock_weights_dir.mkdir(parents=True, exist_ok=True)

    # Load rf once — passed to all optimizer calls.
    rf_series = _load_rf()

    # ------------------------------------------------------------------
    # 1. Load CRSP/Compustat panel
    # ------------------------------------------------------------------
    print(f"Loading stock-month yearly chunks for {subdir}...", flush=True)
    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    # Pure AP-Trees has no tilt signal. Add a dummy column so the shared
    # tree builder is happy; with tau=0 this has no effect on output.
    panel["residual_mom"] = 0.0

    # ------------------------------------------------------------------
    # 2. Build AP-Tree candidate portfolios (or reuse checkpoint)
    # ------------------------------------------------------------------
    candidate_path  = out_dir / "candidate_returns.csv"
    node_stats_path = out_dir / "node_stats.csv"

    reuse_tree_build = (
        candidate_path.exists()
        and node_stats_path.exists()
        and any(stock_weights_dir.glob("*.pkl"))
    )

    if reuse_tree_build:
        print("Reusing existing tree-build checkpoint from disk...", flush=True)
        candidate_returns = pd.read_csv(candidate_path)
        if "date_dt" in candidate_returns.columns:
            candidate_returns["date_dt"] = pd.to_datetime(candidate_returns["date_dt"])
        node_stats = pd.read_csv(node_stats_path)
        if "date_dt" in node_stats.columns:
            node_stats["date_dt"] = pd.to_datetime(node_stats["date_dt"])
    else:
        tree_mode = "full AP-tree set (paper-faithful)" if RUN_FULL_TREE_SET else "single-tree smoke test"
        print(f"Building {tree_mode}, tau={DEFAULT_TAU}...", flush=True)

        candidate_returns, _, node_stats = build_all_ap_tree_candidate_returns(
            panel=panel,
            chars=chars,
            tree_depth=DEFAULT_TREE_DEPTH,
            tau=DEFAULT_TAU,
            q_num=DEFAULT_Q_NUM,
            signal_col="residual_mom",
            deduplicate=DEDUPLICATE_CANDIDATES,
            run_full_tree_set=RUN_FULL_TREE_SET,
            stock_weights_dir=stock_weights_dir,
        )

        candidate_returns.to_csv(candidate_path, index=False)
        node_stats.to_csv(node_stats_path, index=False)

    print(f"Candidate returns shape: {candidate_returns.shape}", flush=True)

    # Select the baseline_ columns and rename to port_ prefix for the optimizer.
    ap_tree_candidates = select_candidate_matrix(candidate_returns, prefix="baseline_")
    ap_tree_candidates.to_csv(out_dir / "ap_tree_candidate_matrix.csv", index=False)

    # ------------------------------------------------------------------
    # Variant A1: Static AP-pruning, no transaction cost
    # ------------------------------------------------------------------
    print("\n[A1] Static AP-pruning, no TC...", flush=True)
    bt_a1, w_a1, diag_a1, sel_a1 = ap_pruning_static_optimize(
        ap_tree_candidates,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda0_grid=AP_LAMBDA0_GRID,
        lambda2_grid=AP_LAMBDA2_GRID,
        port_n=AP_PORT_N,
        kmin=AP_K_MIN,
        kmax=AP_K_MAX,
        method_name=METHOD_A1,
        cost_per_turnover=0.0,
        stock_weights=stock_weights_dir,
        use_stock_level_turnover=USE_STOCK_LEVEL_TURNOVER,
        rf=rf_series,
    )
    print(f"  Selected params: {sel_a1}", flush=True)
    bt_a1.to_csv(out_dir / "backtest_A1_static_no_tc.csv", index=False)
    w_a1.to_csv(out_dir / "weights_A1_static_no_tc.csv", index=False)
    diag_a1.to_csv(out_dir / "diagnostics_A1_static_no_tc.csv", index=False)
    selected_candidates = w_a1["candidate"].tolist()

    # ------------------------------------------------------------------
    # Variant A2: Same AP-pruning selection, stock-level TC ex-post
    # Reuses A1's best (lambda0, lambda2, K) — no re-grid-search needed.
    # ------------------------------------------------------------------
    print("\n[A2] Static AP-pruning, stock-level TC ex-post...", flush=True)
    bt_a2, w_a2, diag_a2, _ = ap_pruning_static_optimize(
        ap_tree_candidates,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda0_grid=[sel_a1.lambda0],
        lambda2_grid=[sel_a1.lambda2],
        port_n=AP_PORT_N,
        kmin=sel_a1.k,
        kmax=sel_a1.k,
        method_name=METHOD_A2,
        cost_per_turnover=TC_COST,
        stock_weights=stock_weights_dir,
        use_stock_level_turnover=USE_STOCK_LEVEL_TURNOVER,
        rf=rf_series,
    )
    bt_a2.to_csv(out_dir / "backtest_A2_static_stock_level_tc.csv", index=False)
    w_a2.to_csv(out_dir / "weights_A2_static_stock_level_tc.csv", index=False)
    diag_a2.to_csv(out_dir / "diagnostics_A2_static_stock_level_tc.csv", index=False)

    # ------------------------------------------------------------------
    # Variant B: Rolling TC-aware, portfolio-level turnover
    # ------------------------------------------------------------------
    print("\n[B] Rolling TC-aware, portfolio-level turnover...", flush=True)
    bt_b, w_b = rolling_tc_optimize(
        ap_tree_candidates,
        window=ROLLING_WINDOW,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        method_name=METHOD_B,
        turnover_mode="portfolio",
        stock_weights=None,
        selected_candidates=selected_candidates,
        long_only=TC_LONG_ONLY,
        rf=rf_series,
    )
    bt_b.to_csv(out_dir / "backtest_B_rolling_tc_portfolio_level_tc.csv", index=False)
    w_b.to_csv(out_dir / "weights_B_rolling_tc_portfolio_level_tc.csv", index=False)

    # ------------------------------------------------------------------
    # Variant C: Rolling TC-aware, stock-level turnover
    # ------------------------------------------------------------------
    print("\n[C] Rolling TC-aware, stock-level turnover...", flush=True)
    bt_c, w_c = rolling_tc_optimize(
        ap_tree_candidates,
        window=ROLLING_WINDOW,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        method_name=METHOD_C,
        turnover_mode="stock",
        stock_weights=stock_weights_dir,
        selected_candidates=selected_candidates,
        long_only=TC_LONG_ONLY,
        rf=rf_series,
    )
    bt_c.to_csv(out_dir / "backtest_C_rolling_tc_stock_level_tc.csv", index=False)
    w_c.to_csv(out_dir / "weights_C_rolling_tc_stock_level_tc.csv", index=False)

    # ------------------------------------------------------------------
    # Align all variants to a common test window
    # ------------------------------------------------------------------
    common_start = max(
        bt_a1["date_dt"].min(), bt_a2["date_dt"].min(),
        bt_b["date_dt"].min(),  bt_c["date_dt"].min(),
    )
    common_end = min(
        bt_a1["date_dt"].max(), bt_a2["date_dt"].max(),
        bt_b["date_dt"].max(),  bt_c["date_dt"].max(),
    )
    print(f"\nCommon test window: {common_start.date()} to {common_end.date()}", flush=True)

    def _clip(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["date_dt"] >= common_start) & (df["date_dt"] <= common_end)].copy()

    bt_a1 = _clip(bt_a1)
    bt_a2 = _clip(bt_a2)
    bt_b  = _clip(bt_b)
    bt_c  = _clip(bt_c)

    pieces = [bt_a1, bt_a2, bt_b, bt_c]

    # S&P 500 benchmark from Yahoo Finance
    try:
        print("\nLoading S&P 500 benchmark (SPY)...", flush=True)
        bt_spy = load_yahoo_monthly_benchmark(
            ticker="SPY",
            start_date=common_start,
            end_date=common_end,
            method_name="S&P 500 (adjusted close)",
        )
        bt_spy = bt_spy[bt_spy["date_dt"].isin(bt_a1["date_dt"])].copy()
        pieces.append(bt_spy)
    except Exception as e:
        print(f"WARNING: Could not load SPY benchmark: {e}", flush=True)

    # ------------------------------------------------------------------
    # Combine and write outputs
    # ------------------------------------------------------------------
    backtest = pd.concat(pieces, ignore_index=True)
    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)

    schema = ["date", "date_dt", "yy", "mm", "method",
              "gross_ret", "turnover_raw", "turnover", "cost", "net_ret"]
    backtest = backtest[[c for c in schema if c in backtest.columns]]

    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)
    print(f"\nbacktest_comparison.csv -> {out_dir / 'backtest_comparison.csv'}", flush=True)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown.csv", index=False)

    metrics = performance_metrics(backtest)
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)

    print("\nMaking plots...", flush=True)
    make_all_plots(backtest, plot_dir)

    print(f"\nDone. Outputs saved to: {out_dir}", flush=True)
    print(f"Plots saved to:          {plot_dir}", flush=True)


if __name__ == "__main__":
    main()