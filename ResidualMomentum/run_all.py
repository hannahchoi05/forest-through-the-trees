from __future__ import annotations

import shutil
import pandas as pd

from config import (
    CHUNK_DIR,
    FACTOR_DIR,
    DEFAULT_Y_MIN,
    DEFAULT_Y_MAX,
    DEFAULT_TREE_DEPTH,
    DEFAULT_Q_NUM,
    MOM_LOOKBACK,
    MOM_SKIP_RECENT,
    BETA_WINDOW,
    DEDUPLICATE_CANDIDATES,
    RUN_FULL_TREE_SET,
    OUTPUT_DIR,
    PLOT_DIR,
    DEFAULT_CHARS,
    DEFAULT_TAU,
    N_TRAIN_VALID,
    CV_N,
    ROLLING_WINDOW,
    TC_COST,
    TC_LAMBDA_L2,
    TC_LAMBDA_TC,
    USE_STOCK_LEVEL_TURNOVER,
    AP_LAMBDA0_GRID,
    AP_LAMBDA2_GRID,
    AP_K_MIN,
    AP_K_MAX,
    AP_PORT_N,
    TC_ETA,
    TC_LONG_ONLY,
)
from data_io import (
    load_yearly_chunks,
    load_market_proxy,
    load_yahoo_monthly_benchmark,
)
from residual_momentum import (
    add_raw_momentum_signal,
    add_market_residual_momentum_signal,
)
from tree_portfolios import (
    build_all_ap_tree_candidate_returns,
    select_candidate_matrix,
)
from optimizer import ap_pruning_static_optimize, rolling_tc_optimize
from metrics import performance_metrics, add_wealth_drawdown
from plots import make_all_plots


def _align_to_common_window(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    common_start = max(df["date_dt"].min() for df in dfs)
    common_end = min(df["date_dt"].max() for df in dfs)

    out = []
    for df in dfs:
        out.append(
            df[
                (df["date_dt"] >= common_start)
                & (df["date_dt"] <= common_end)
            ].copy()
        )

    print(f"Common test window: {common_start.date()} to {common_end.date()}", flush=True)
    return out


def _run_optimization_and_backtests(
    tilted_returns: pd.DataFrame,
    stock_weights_path,
    out_dir,
    plot_dir,
) -> None:
    if "date_dt" in tilted_returns.columns:
        tilted_returns["date_dt"] = pd.to_datetime(tilted_returns["date_dt"])

    print("\n[A1] AP-pruning static, no TC...", flush=True)
    bt_a1, w_a1, diag_a1, sel = ap_pruning_static_optimize(
        tilted_returns,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda0_grid=AP_LAMBDA0_GRID,
        lambda2_grid=AP_LAMBDA2_GRID,
        port_n=AP_PORT_N,
        kmin=AP_K_MIN,
        kmax=AP_K_MAX,
        method_name="AP-tree + RM AP-pruning (static, no TC)",
        cost_per_turnover=0.0,
        stock_weights=None,
        use_stock_level_turnover=False,
    )
    print(f"Selected AP-pruning params: {sel}", flush=True)

    bt_a1.to_csv(out_dir / "backtest_A1_ap_pruning_static_no_tc.csv", index=False)
    w_a1.to_csv(out_dir / "weights_A1_ap_pruning_static_no_tc.csv", index=False)
    diag_a1.to_csv(out_dir / "diagnostics_A1_ap_pruning_static_no_tc.csv", index=False)

    selected_candidates = w_a1["candidate"].tolist()

    print("\n[A2] Same AP-pruning selection, stock-level TC ex-post...", flush=True)
    bt_a2, w_a2, diag_a2, _ = ap_pruning_static_optimize(
        tilted_returns,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda0_grid=[sel.lambda0],
        lambda2_grid=[sel.lambda2],
        port_n=AP_PORT_N,
        kmin=AP_K_MIN,
        kmax=AP_K_MAX,
        method_name="AP-tree + RM AP-pruning (static + stock-level TC)",
        cost_per_turnover=TC_COST,
        stock_weights=stock_weights_path,
        use_stock_level_turnover=USE_STOCK_LEVEL_TURNOVER,
    )

    bt_a2.to_csv(out_dir / "backtest_A2_ap_pruning_static_stock_level_tc.csv", index=False)
    w_a2.to_csv(out_dir / "weights_A2_ap_pruning_static_stock_level_tc.csv", index=False)
    diag_a2.to_csv(out_dir / "diagnostics_A2_ap_pruning_static_stock_level_tc.csv", index=False)

    print("\n[B] TC-aware rolling ablation, portfolio-level turnover...", flush=True)
    bt_b, w_b = rolling_tc_optimize(
        tilted_returns,
        window=ROLLING_WINDOW,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        method_name="AP-tree + RM rolling TC-aware (portfolio-level TC)",
        turnover_mode="portfolio",
        stock_weights=None,
        selected_candidates=selected_candidates,
        long_only=TC_LONG_ONLY,
    )

    bt_b.to_csv(out_dir / "backtest_B_rolling_tc_portfolio_level_tc.csv", index=False)
    w_b.to_csv(out_dir / "weights_B_rolling_tc_portfolio_level_tc.csv", index=False)

    print("\n[C] TC-aware rolling ablation, stock-level turnover...", flush=True)
    bt_c, w_c = rolling_tc_optimize(
        tilted_returns,
        window=ROLLING_WINDOW,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        method_name="AP-tree + RM rolling TC-aware (stock-level TC)",
        turnover_mode="stock",
        stock_weights=stock_weights_path,
        selected_candidates=selected_candidates,
        long_only=TC_LONG_ONLY,
    )

    bt_c.to_csv(out_dir / "backtest_C_rolling_tc_stock_level_tc.csv", index=False)
    w_c.to_csv(out_dir / "weights_C_rolling_tc_stock_level_tc.csv", index=False)

    bt_a1, bt_a2, bt_b, bt_c = _align_to_common_window([bt_a1, bt_a2, bt_b, bt_c])

    pieces = [bt_a1, bt_a2, bt_b, bt_c]

    try:
        print("\nLoading S&P 500 benchmark from Yahoo Finance using SPY...", flush=True)
        common_start = bt_a1["date_dt"].min()
        common_end = bt_a1["date_dt"].max()

        bt_spy = load_yahoo_monthly_benchmark(
            ticker="SPY",
            start_date=common_start,
            end_date=common_end,
            method_name="S&P 500 (SPY adjusted close)",
        )
        bt_spy = bt_spy[bt_spy["date_dt"].isin(bt_a1["date_dt"])].copy()
        pieces.append(bt_spy)
    except Exception as e:
        print(f"WARNING: Could not load SPY benchmark: {e}", flush=True)

    backtest = pd.concat(pieces, ignore_index=True)
    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)
    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown.csv", index=False)

    metrics = performance_metrics(backtest)
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)

    print("\nMaking plots...", flush=True)
    make_all_plots(backtest, plot_dir)

    print(f"\nDone. Outputs saved to: {out_dir}", flush=True)
    print(f"Plots saved to: {plot_dir}", flush=True)


def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    stock_weights_path = out_dir / f"stock_weights_by_month_tau_{DEFAULT_TAU}"

    if stock_weights_path.exists():
        shutil.rmtree(stock_weights_path)
    stock_weights_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading stock-month yearly chunks for {subdir}...", flush=True)
    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    print("Computing residual momentum signal...", flush=True)
    market = load_market_proxy(FACTOR_DIR)

    if market is not None:
        panel = add_market_residual_momentum_signal(
            panel,
            market_returns=market,
            lookback=MOM_LOOKBACK,
            skip_recent=MOM_SKIP_RECENT,
            beta_window=BETA_WINDOW,
        )
        signal_version = "market_residual_momentum"
    else:
        panel = add_raw_momentum_signal(
            panel,
            lookback=MOM_LOOKBACK,
            skip_recent=MOM_SKIP_RECENT,
        )
        signal_version = "raw_momentum_fallback"

    panel.to_csv(out_dir / f"stock_panel_with_{signal_version}.csv", index=False)

    tree_mode = "full AP-tree set" if RUN_FULL_TREE_SET else "single-tree debug set"
    print(f"Building {tree_mode} with residual momentum tilt...", flush=True)
    print(f"Streaming stock weights to: {stock_weights_path}", flush=True)

    candidate_returns, stock_weights_path, node_stats = build_all_ap_tree_candidate_returns(
        panel=panel,
        chars=chars,
        tree_depth=DEFAULT_TREE_DEPTH,
        tau=DEFAULT_TAU,
        q_num=DEFAULT_Q_NUM,
        signal_col="residual_mom",
        deduplicate=DEDUPLICATE_CANDIDATES,
        run_full_tree_set=RUN_FULL_TREE_SET,
        stock_weights_dir=stock_weights_path,
    )

    print(f"Candidate returns shape: {candidate_returns.shape}", flush=True)
    print(f"Stock weights streamed to: {stock_weights_path}", flush=True)

    candidate_returns.to_csv(out_dir / f"candidate_returns_tau_{DEFAULT_TAU}.csv", index=False)
    node_stats.to_csv(out_dir / f"node_stats_tau_{DEFAULT_TAU}.csv", index=False)

    tilted_returns = select_candidate_matrix(candidate_returns, prefix="tilt_")
    tilted_returns.to_csv(out_dir / f"tilted_candidate_matrix_tau_{DEFAULT_TAU}.csv", index=False)

    _run_optimization_and_backtests(
        tilted_returns=tilted_returns,
        stock_weights_path=stock_weights_path,
        out_dir=out_dir,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
