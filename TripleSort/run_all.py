from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


def _bootstrap_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_bootstrap_imports()

from config import (  # noqa: E402
    CHUNK_DIR,
    DEFAULT_CHARS,
    DEFAULT_N_BINS,
    DEFAULT_Y_MIN,
    DEFAULT_Y_MAX,
    N_TRAIN_VALID,
    CV_N,
    STATIC_LAMBDA_L2,
    STATIC_MU0,
    LONG_ONLY,
    ROLLING_WINDOW,
    TC_COST,
    TC_LAMBDA_L2,
    TC_LAMBDA_TC,
    TC_ETA,
    TC_LONG_ONLY,
    OUTPUT_DIR,
    TS32_DIR,
)
from check_portfolios import check_all  # noqa: E402
from data_io import (  # noqa: E402
    load_triplesort_excess_returns,
    load_yearly_chunks,
    load_yahoo_monthly_benchmark,
)
from optimizer import static_paper_style_optimize, rolling_tc_optimize  # noqa: E402
from metrics import performance_metrics, add_wealth_drawdown  # noqa: E402


def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)
    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Checking triple-sort portfolio parity vs Data/ ...", flush=True)
    check_all()

    print(f"Loading triple-sort candidate returns for {subdir}...", flush=True)
    portfolio_csv = TS32_DIR / subdir / "excess_ports.csv"
    returns = load_triplesort_excess_returns(portfolio_csv)

    print("Loading stock-month panel for stock-level turnover...", flush=True)
    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    # ---------------------------------------------------------------------
    # A1: Static optimizer, no transaction costs.
    # ---------------------------------------------------------------------
    print("Running Triple Sort static optimizer (no TC)...", flush=True)
    bt_a1, w_a1, diag_a1 = static_paper_style_optimize(
        returns,
        n_train_valid=N_TRAIN_VALID,
        panel=None,
        n_bins=DEFAULT_N_BINS,
        cv_n=CV_N,
        lambda_l2=STATIC_LAMBDA_L2,
        mu0=STATIC_MU0,
        long_only=LONG_ONLY,
        method_name="Triple Sort static (no TC)",
        cost_per_turnover=0.0,
        use_stock_level_turnover=False,
    )

    # Legacy outputs (backtest_triplesort_* / diagnostics_triplesort_* / weights_triplesort_*)
    # are no longer emitted; keep only canonical A1/A2/B/C + comparison outputs.

    # ---------------------------------------------------------------------
    # A2: Static optimizer with stock-level transaction costs.
    # ---------------------------------------------------------------------
    print("Running Triple Sort static optimizer (stock-level TC)...", flush=True)
    bt_a2, w_a2, diag_a2 = static_paper_style_optimize(
        returns,
        n_train_valid=N_TRAIN_VALID,
        panel=panel,
        n_bins=DEFAULT_N_BINS,
        cv_n=CV_N,
        lambda_l2=STATIC_LAMBDA_L2,
        mu0=STATIC_MU0,
        long_only=LONG_ONLY,
        method_name="Triple Sort static + stock-level TC",
        cost_per_turnover=TC_COST,
        use_stock_level_turnover=True,
    )

    # (weights are not emitted by default to avoid clutter)

    # ---------------------------------------------------------------------
    # B: Rolling TC-aware optimizer with portfolio-level turnover costs.
    # ---------------------------------------------------------------------
    print("Running Triple Sort rolling TC-aware optimizer (portfolio-level TC)...", flush=True)
    bt_b, w_b = rolling_tc_optimize(
        returns,
        window=ROLLING_WINDOW,
        panel=None,
        n_bins=DEFAULT_N_BINS,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        long_only=TC_LONG_ONLY,
        method_name="Triple Sort rolling TC-aware + portfolio-level TC",
        turnover_mode="portfolio",
    )

    # (rolling weights not emitted by default)

    # ---------------------------------------------------------------------
    # C: Rolling TC-aware optimizer with stock-level turnover costs.
    # ---------------------------------------------------------------------
    print("Running Triple Sort rolling TC-aware optimizer (stock-level TC)...", flush=True)
    bt_c, w_c = rolling_tc_optimize(
        returns,
        window=ROLLING_WINDOW,
        panel=panel,
        n_bins=DEFAULT_N_BINS,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        long_only=TC_LONG_ONLY,
        method_name="Triple Sort rolling TC-aware + stock-level TC",
        turnover_mode="stock",
    )

    # (rolling weights not emitted by default)

    # ---------------------------------------------------------------------
    # Write residual_momentum-style filenames (A1/A2/B/C).
    # ---------------------------------------------------------------------
    bt_a1.to_csv(out_dir / "backtest_A1_triple_sort_static_no_tc.csv", index=False)
    diag_a1.to_csv(out_dir / "diagnostics_A1_triple_sort_static_no_tc.csv", index=False)

    bt_a2.to_csv(out_dir / "backtest_A2_triple_sort_static_stock_level_tc.csv", index=False)
    diag_a2.to_csv(out_dir / "diagnostics_A2_triple_sort_static_stock_level_tc.csv", index=False)

    bt_b.to_csv(out_dir / "backtest_B_rolling_tc_portfolio_level_tc.csv", index=False)
    bt_c.to_csv(out_dir / "backtest_C_rolling_tc_stock_level_tc.csv", index=False)

    # ---------------------------------------------------------------------
    # Align all variants to a common calendar sample and write comparison.
    # ---------------------------------------------------------------------
    common_start = max(
        bt_a1["date_dt"].min(),
        bt_a2["date_dt"].min(),
        bt_b["date_dt"].min(),
        bt_c["date_dt"].min(),
    )
    common_end = min(
        bt_a1["date_dt"].max(),
        bt_a2["date_dt"].max(),
        bt_b["date_dt"].max(),
        bt_c["date_dt"].max(),
    )

    bt_a1 = bt_a1[(bt_a1["date_dt"] >= common_start) & (bt_a1["date_dt"] <= common_end)].copy()
    bt_a2 = bt_a2[(bt_a2["date_dt"] >= common_start) & (bt_a2["date_dt"] <= common_end)].copy()
    bt_b = bt_b[(bt_b["date_dt"] >= common_start) & (bt_b["date_dt"] <= common_end)].copy()
    bt_c = bt_c[(bt_c["date_dt"] >= common_start) & (bt_c["date_dt"] <= common_end)].copy()

    pieces = [bt_a1, bt_a2, bt_b, bt_c]

    # Add S&P 500 benchmark from Yahoo Finance (adjusted close).
    # We intentionally do not fall back to the factor proxy here.
    bt_sp = load_yahoo_monthly_benchmark(
        ticker="SPY",
        start_date=common_start,
        end_date=common_end,
        method_name="S&P 500 (adjusted close)",
    )
    bt_sp = bt_sp[bt_sp["date_dt"].isin(bt_a1["date_dt"])].copy()
    pieces.append(bt_sp)

    backtest = pd.concat(pieces, ignore_index=True)
    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)
    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown.csv", index=False)

    metrics = performance_metrics(backtest)
    ordered_cols = [
        "method",
        "n_months",
        "start_date",
        "end_date",
        "mean_gross_monthly",
        "mean_net_monthly",
        "mean_gross_ann",
        "mean_net_ann",
        "vol_gross_monthly",
        "vol_net_monthly",
        "vol_gross_ann",
        "vol_net_ann",
        "sharpe_gross_ann",
        "sharpe_net_ann",
        "sharpe_decay_due_to_costs",
        "hit_rate_gross",
        "hit_rate_net",
        "avg_turnover",
        "avg_cost",
        "max_drawdown_net",
        "terminal_wealth_gross",
        "terminal_wealth_net",
    ]
    metrics = metrics[[c for c in ordered_cols if c in metrics.columns]]
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(f"\nDone. Outputs saved to: {out_dir}", flush=True)
    print("Generate plots from existing CSVs with: python3 TripleSort/run_plots.py", flush=True)


if __name__ == "__main__":
    main()
