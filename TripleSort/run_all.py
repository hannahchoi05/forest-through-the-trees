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
    STATIC_MU0_GRID,
    STATIC_LAMBDA_L2_GRID,
    STATIC_K,
    STATIC_K_MIN,
    STATIC_K_MAX,
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

    print(f"Loading triple-sort candidate returns (TS32) for {subdir}...", flush=True)
    portfolio_csv = TS32_DIR / subdir / "excess_ports.csv"
    returns = load_triplesort_excess_returns(portfolio_csv)

    print("Loading stock-month panel for stock-level turnover...", flush=True)
    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    # -------------------------------------------------------------------------
    # A1: LARS-based AP-Pruning, no transaction costs.
    #     Grid-searches (mu0, lambda_l2) on validation window.
    # -------------------------------------------------------------------------
    print("Running Triple Sort LARS optimizer (no TC) — grid search...", flush=True)
    bt_a1, w_a1, diag_a1 = static_paper_style_optimize(
        returns,
        n_train_valid=N_TRAIN_VALID,
        panel=None,
        n_bins=DEFAULT_N_BINS,
        cv_n=CV_N,
        long_only=LONG_ONLY,
        method_name="Triple Sort static (no TC)",
        cost_per_turnover=0.0,
        use_stock_level_turnover=False,
        mu0_grid=STATIC_MU0_GRID,
        lambda_l2_grid=STATIC_LAMBDA_L2_GRID,
        k_target=STATIC_K,
        k_min=STATIC_K_MIN,
        k_max=STATIC_K_MAX,
    )

    # -------------------------------------------------------------------------
    # A2: Same LARS weights, stock-level TC applied ex-post.
    # -------------------------------------------------------------------------
    print("Running Triple Sort LARS optimizer (stock-level TC)...", flush=True)
    bt_a2, w_a2, diag_a2 = static_paper_style_optimize(
        returns,
        n_train_valid=N_TRAIN_VALID,
        panel=panel,
        n_bins=DEFAULT_N_BINS,
        cv_n=CV_N,
        long_only=LONG_ONLY,
        method_name="Triple Sort static + stock-level TC",
        cost_per_turnover=TC_COST,
        use_stock_level_turnover=True,
        mu0_grid=STATIC_MU0_GRID,
        lambda_l2_grid=STATIC_LAMBDA_L2_GRID,
        k_target=STATIC_K,
        k_min=STATIC_K_MIN,
        k_max=STATIC_K_MAX,
    )

    # -------------------------------------------------------------------------
    # B: Rolling TC-aware, portfolio-level turnover in objective.
    # -------------------------------------------------------------------------
    print("Running Triple Sort rolling TC-aware (portfolio-level TC)...", flush=True)
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

    # -------------------------------------------------------------------------
    # C: Rolling TC-aware, stock-level turnover in objective.
    # -------------------------------------------------------------------------
    print("Running Triple Sort rolling TC-aware (stock-level TC)...", flush=True)
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

    # -------------------------------------------------------------------------
    # Write individual A1/A2/B/C files.
    # -------------------------------------------------------------------------
    bt_a1.to_csv(out_dir / "backtest_A1_triple_sort_static_no_tc.csv",               index=False)
    diag_a1.to_csv(out_dir / "diagnostics_A1_triple_sort_static_no_tc.csv",          index=False)
    bt_a2.to_csv(out_dir / "backtest_A2_triple_sort_static_stock_level_tc.csv",       index=False)
    diag_a2.to_csv(out_dir / "diagnostics_A2_triple_sort_static_stock_level_tc.csv", index=False)
    bt_b.to_csv(out_dir / "backtest_B_rolling_tc_portfolio_level_tc.csv",             index=False)
    bt_c.to_csv(out_dir / "backtest_C_rolling_tc_stock_level_tc.csv",                 index=False)

    # -------------------------------------------------------------------------
    # Align all variants to common calendar and write comparison.
    # -------------------------------------------------------------------------
    common_start = max(
        bt_a1["date_dt"].min(), bt_a2["date_dt"].min(),
        bt_b["date_dt"].min(),  bt_c["date_dt"].min(),
    )
    common_end = min(
        bt_a1["date_dt"].max(), bt_a2["date_dt"].max(),
        bt_b["date_dt"].max(),  bt_c["date_dt"].max(),
    )

    def _trim(bt: pd.DataFrame) -> pd.DataFrame:
        return bt[
            (bt["date_dt"] >= common_start) & (bt["date_dt"] <= common_end)
        ].copy()

    bt_a1 = _trim(bt_a1)
    bt_a2 = _trim(bt_a2)
    bt_b  = _trim(bt_b)
    bt_c  = _trim(bt_c)
    pieces = [bt_a1, bt_a2, bt_b, bt_c]

    # S&P 500 benchmark from Yahoo Finance.
    try:
        bt_sp = load_yahoo_monthly_benchmark(
            ticker="SPY",
            start_date=common_start,
            end_date=common_end,
            method_name="S&P 500 (adjusted close)",
        )
        bt_sp = bt_sp[bt_sp["date_dt"].isin(bt_a1["date_dt"])].copy()
        pieces.append(bt_sp)
    except Exception as e:
        print(f"WARNING: Could not load SPY benchmark: {e}", flush=True)

    backtest = pd.concat(pieces, ignore_index=True)
    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)
    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown.csv", index=False)

    metrics = performance_metrics(backtest)
    ordered_cols = [
        "method", "n_months", "start_date", "end_date",
        "mean_gross_monthly", "mean_net_monthly",
        "mean_gross_ann", "mean_net_ann",
        "vol_gross_monthly", "vol_net_monthly",
        "vol_gross_ann", "vol_net_ann",
        "sharpe_gross_ann", "sharpe_net_ann",
        "sharpe_decay_due_to_costs",
        "hit_rate_gross", "hit_rate_net",
        "avg_turnover", "avg_cost",
        "max_drawdown_net",
        "terminal_wealth_gross", "terminal_wealth_net",
    ]
    metrics = metrics[[c for c in ordered_cols if c in metrics.columns]]
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(f"\nDone. Outputs saved to: {out_dir}", flush=True)
    print("Generate plots from existing CSVs with: python3 TripleSort/run_plots.py", flush=True)


if __name__ == "__main__":
    main()
