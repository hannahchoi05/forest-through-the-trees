from __future__ import annotations
import pandas as pd

from config import (
    CHUNK_DIR, FACTOR_DIR, OUTPUT_DIR, PLOT_DIR,
    DEFAULT_CHARS, DEFAULT_Y_MIN, DEFAULT_Y_MAX, DEFAULT_TREE_DEPTH, DEFAULT_Q_NUM,
    DEFAULT_TAU, MOM_LOOKBACK, MOM_SKIP_RECENT, BETA_WINDOW,
    N_TRAIN_VALID, CV_N, STATIC_LAMBDA_L1, STATIC_LAMBDA_L2, STATIC_MU0, LONG_ONLY,
    ROLLING_WINDOW, TC_COST, TC_LAMBDA_L1, TC_LAMBDA_L2, TC_LAMBDA_TC, TC_MU0,
    DEDUPLICATE_CANDIDATES,
)
from data_io import load_yearly_chunks, load_market_proxy
from residual_momentum import add_raw_momentum_signal, add_market_residual_momentum_signal
from tree_portfolios import build_all_ap_tree_candidate_returns, select_candidate_matrix
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

    print(f"Loading stock-month yearly chunks for {subdir}...")
    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    print("Computing residual momentum signal...")
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

    print("Building full AP-tree candidate set with residual momentum tilt...")
    candidate_returns, stock_weights, node_stats = build_all_ap_tree_candidate_returns(
        panel=panel,
        chars=chars,
        tree_depth=DEFAULT_TREE_DEPTH,
        tau=DEFAULT_TAU,
        q_num=DEFAULT_Q_NUM,
        signal_col="residual_mom",
        deduplicate=DEDUPLICATE_CANDIDATES,
    )
    candidate_returns.to_csv(out_dir / f"candidate_returns_full_ap_tree_tau_{DEFAULT_TAU}.csv", index=False)
    stock_weights.to_csv(out_dir / f"stock_weights_full_ap_tree_tau_{DEFAULT_TAU}.csv", index=False)
    node_stats.to_csv(out_dir / f"node_stats_full_ap_tree_tau_{DEFAULT_TAU}.csv", index=False)

    # We compare the two requested methods using the residual-momentum-tilted AP-tree candidates.
    tilted_returns = select_candidate_matrix(candidate_returns, prefix="tilt_")
    tilted_returns.to_csv(out_dir / f"tilted_candidate_matrix_tau_{DEFAULT_TAU}.csv", index=False)

    print("Running paper-faithful static optimization: no rolling window, residual momentum tilt...")
    bt_static, static_weights, static_diag = static_paper_style_optimize(
        tilted_returns,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda_l1=STATIC_LAMBDA_L1,
        lambda_l2=STATIC_LAMBDA_L2,
        mu0=STATIC_MU0,
        long_only=LONG_ONLY,
        method_name="Static paper-style optimizer + residual momentum tilt",
    )
    bt_static.to_csv(out_dir / "backtest_static_paper_style_plus_residual_momentum_tilt.csv", index=False)
    static_weights.to_csv(out_dir / "weights_static_paper_style_plus_residual_momentum_tilt.csv", index=False)
    static_diag.to_csv(out_dir / "diagnostics_static_train_valid_test.csv", index=False)

    print("Running rolling TC-aware optimization: residual momentum tilt + transaction cost penalty...")
    bt_tc, tc_weights = rolling_tc_optimize(
        tilted_returns,
        window=ROLLING_WINDOW,
        lambda_l1=TC_LAMBDA_L1,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        cost_per_turnover=TC_COST,
        mu0=TC_MU0,
        long_only=LONG_ONLY,
        method_name="Rolling TC-aware optimizer + residual momentum tilt",
    )
    bt_tc.to_csv(out_dir / "backtest_rolling_tc_aware_plus_residual_momentum_tilt.csv", index=False)
    tc_weights.to_csv(out_dir / "weights_rolling_tc_aware_plus_residual_momentum_tilt.csv", index=False)

    backtest = pd.concat([bt_static, bt_tc], ignore_index=True)
    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)
    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown.csv", index=False)

    metrics = performance_metrics(backtest)
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)
    print("\nSummary metrics:")
    print(metrics.to_string(index=False))

    print("Making plots...")
    make_all_plots(backtest, plot_dir)
    print(f"Done. Outputs saved to: {out_dir}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
