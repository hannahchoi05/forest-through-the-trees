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
from data_io import load_triplesort_excess_returns, load_yearly_chunks  # noqa: E402
from optimizer import static_paper_style_optimize, rolling_tc_optimize  # noqa: E402
from metrics import performance_metrics  # noqa: E402


def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    bt_a1.to_csv(out_dir / "backtest_triplesort_static_no_tc.csv", index=False)
    w_a1.to_csv(out_dir / "weights_triplesort_static_no_tc.csv", index=False)
    diag_a1.to_csv(out_dir / "diagnostics_triplesort_static_no_tc.csv", index=False)

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

    bt_a2.to_csv(out_dir / "backtest_triplesort_static_stock_level_tc.csv", index=False)
    w_a2.to_csv(out_dir / "weights_triplesort_static_stock_level_tc.csv", index=False)
    diag_a2.to_csv(out_dir / "diagnostics_triplesort_static_stock_level_tc.csv", index=False)

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

    bt_b.to_csv(out_dir / "backtest_triplesort_rolling_tc_port_level.csv", index=False)
    w_b.to_csv(out_dir / "weights_triplesort_rolling_tc_port_level.csv", index=False)

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

    bt_c.to_csv(out_dir / "backtest_triplesort_rolling_tc_stock_level.csv", index=False)
    w_c.to_csv(out_dir / "weights_triplesort_rolling_tc_stock_level.csv", index=False)

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

    backtest = pd.concat([bt_a1, bt_a2, bt_b, bt_c], ignore_index=True)
    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)
    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)

    metrics = performance_metrics(backtest)
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(f"\nDone. Outputs saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
