from __future__ import annotations

import shutil
import pandas as pd

from config import (
    CHUNK_DIR,
    FACTOR_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
    DEFAULT_CHARS,
    DEFAULT_Y_MIN,
    DEFAULT_Y_MAX,
    DEFAULT_TREE_DEPTH,
    DEFAULT_Q_NUM,
    DEFAULT_TAU,
    MOM_LOOKBACK,
    MOM_SKIP_RECENT,
    BETA_WINDOW,
    N_TRAIN_VALID,
    CV_N,
    STATIC_LAMBDA_L1,
    STATIC_LAMBDA_L2,
    STATIC_MU0,
    LONG_ONLY,
    ROLLING_WINDOW,
    TC_COST,
    TC_LAMBDA_L1,
    TC_LAMBDA_L2,
    TC_LAMBDA_TC,
    TC_MU0,
    DEDUPLICATE_CANDIDATES,
    RUN_FULL_TREE_SET,
    USE_STOCK_LEVEL_TURNOVER,
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
from optimizer import static_paper_style_optimize, rolling_tc_optimize
from metrics import performance_metrics, add_wealth_drawdown
from plots import make_all_plots


def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    plot_dir = PLOT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    stock_weights_dir = out_dir / f"stock_weights_by_month_tau_{DEFAULT_TAU}"

    if stock_weights_dir.exists():
        shutil.rmtree(stock_weights_dir)

    stock_weights_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Streaming stock weights to: {stock_weights_dir}", flush=True)

    candidate_returns, stock_weights_path, node_stats = build_all_ap_tree_candidate_returns(
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

    print(f"Candidate returns shape: {candidate_returns.shape}", flush=True)
    print(f"Stock weights streamed to: {stock_weights_path}", flush=True)

    candidate_returns.to_csv(out_dir / f"candidate_returns_tau_{DEFAULT_TAU}.csv", index=False)
    node_stats.to_csv(out_dir / f"node_stats_tau_{DEFAULT_TAU}.csv", index=False)

    tilted_returns = select_candidate_matrix(candidate_returns, prefix="tilt_")
    tilted_returns.to_csv(out_dir / f"tilted_candidate_matrix_tau_{DEFAULT_TAU}.csv", index=False)

    print(
        "Running AP-tree + RM baseline: "
        "static optimizer, no transaction costs...",
        flush=True,
    )

    bt_rm_static_no_tc, rm_static_no_tc_weights, rm_static_no_tc_diag = static_paper_style_optimize(
        tilted_returns,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda_l1=STATIC_LAMBDA_L1,
        lambda_l2=STATIC_LAMBDA_L2,
        mu0=STATIC_MU0,
        long_only=LONG_ONLY,
        method_name="AP-tree + RM baseline (static, no TC)",
        cost_per_turnover=0.0,
        stock_weights=None,
        use_stock_level_turnover=False,
    )

    bt_rm_static_no_tc.to_csv(out_dir / "backtest_ap_tree_rm_static_no_tc.csv", index=False)
    rm_static_no_tc_weights.to_csv(out_dir / "weights_ap_tree_rm_static_no_tc.csv", index=False)
    rm_static_no_tc_diag.to_csv(out_dir / "diagnostics_ap_tree_rm_static_no_tc.csv", index=False)

    print(
        "Running AP-tree + RM static optimizer with stock-level transaction costs...",
        flush=True,
    )

    bt_static, static_weights, static_diag = static_paper_style_optimize(
        tilted_returns,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda_l1=STATIC_LAMBDA_L1,
        lambda_l2=STATIC_LAMBDA_L2,
        mu0=STATIC_MU0,
        long_only=LONG_ONLY,
        method_name="AP-tree + RM static + stock-level TC",
        cost_per_turnover=TC_COST,
        stock_weights=stock_weights_path,
        use_stock_level_turnover=USE_STOCK_LEVEL_TURNOVER,
    )

    bt_static.to_csv(out_dir / "backtest_ap_tree_rm_static_stock_level_tc.csv", index=False)
    static_weights.to_csv(out_dir / "weights_ap_tree_rm_static_stock_level_tc.csv", index=False)
    static_diag.to_csv(out_dir / "diagnostics_ap_tree_rm_static_stock_level_tc.csv", index=False)

    print(
        "Running AP-tree + RM rolling TC-aware optimizer with stock-level transaction costs...",
        flush=True,
    )

    bt_tc, tc_weights = rolling_tc_optimize(
        tilted_returns,
        window=ROLLING_WINDOW,
        lambda_l1=TC_LAMBDA_L1,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        cost_per_turnover=TC_COST,
        mu0=TC_MU0,
        long_only=LONG_ONLY,
        method_name="AP-tree + RM rolling TC-aware + stock-level TC",
        stock_weights=stock_weights_path,
        use_stock_level_turnover=USE_STOCK_LEVEL_TURNOVER,
    )

    bt_tc.to_csv(out_dir / "backtest_ap_tree_rm_rolling_tc_stock_level_tc.csv", index=False)
    tc_weights.to_csv(out_dir / "weights_ap_tree_rm_rolling_tc_stock_level_tc.csv", index=False)

    common_start = max(
        bt_rm_static_no_tc["date_dt"].min(),
        bt_static["date_dt"].min(),
        bt_tc["date_dt"].min(),
    )

    common_end = min(
        bt_rm_static_no_tc["date_dt"].max(),
        bt_static["date_dt"].max(),
        bt_tc["date_dt"].max(),
    )

    bt_rm_static_no_tc = bt_rm_static_no_tc[
        (bt_rm_static_no_tc["date_dt"] >= common_start)
        & (bt_rm_static_no_tc["date_dt"] <= common_end)
    ].copy()

    bt_static = bt_static[
        (bt_static["date_dt"] >= common_start)
        & (bt_static["date_dt"] <= common_end)
    ].copy()

    bt_tc = bt_tc[
        (bt_tc["date_dt"] >= common_start)
        & (bt_tc["date_dt"] <= common_end)
    ].copy()

    print("Loading S&P 500 benchmark from Yahoo Finance using SPY...", flush=True)

    try:
        bt_spy = load_yahoo_monthly_benchmark(
            ticker="SPY",
            start_date=common_start,
            end_date=common_end,
            method_name="S&P 500 (SPY adjusted close)",
        )

        strategy_dates = pd.concat(
            [
                bt_rm_static_no_tc[["date_dt"]],
                bt_static[["date_dt"]],
                bt_tc[["date_dt"]],
            ],
            ignore_index=True,
        )["date_dt"].drop_duplicates()

        bt_spy = bt_spy[bt_spy["date_dt"].isin(strategy_dates)].copy()

        print(f"SPY benchmark rows: {len(bt_spy)}", flush=True)

    except Exception as e:
        print(f"WARNING: Could not load SPY benchmark from Yahoo Finance: {e}", flush=True)
        bt_spy = pd.DataFrame()

    pieces = [bt_rm_static_no_tc, bt_static, bt_tc]

    if not bt_spy.empty:
        pieces.append(bt_spy)

    backtest = pd.concat(pieces, ignore_index=True)

    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)
    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown.csv", index=False)

    metrics = performance_metrics(backtest)
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)

    print("Making plots...", flush=True)
    make_all_plots(backtest, plot_dir)

    print(f"Done. Outputs saved to: {out_dir}", flush=True)
    print(f"Plots saved to: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()