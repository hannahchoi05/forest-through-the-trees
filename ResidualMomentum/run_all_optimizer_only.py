from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    FACTOR_DIR,
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
from data_io import load_yahoo_monthly_benchmark
from optimizer import ap_pruning_static_optimize, rolling_tc_optimize
from metrics import performance_metrics, add_wealth_drawdown
from plots import make_all_plots


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

    # Detect if already in decimal form (median absolute value << 0.01 means decimal)
    if float(np.median(np.abs(rf_raw))) < 0.01::
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


def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = out_dir / f"tilted_candidate_matrix_tau_{DEFAULT_TAU}.csv"
    stock_weights_path = out_dir / f"stock_weights_by_month_tau_{DEFAULT_TAU}"

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Missing candidate matrix: {candidate_path}. "
            "Run the full tree-building pipeline first."
        )

    if not stock_weights_path.exists():
        raise FileNotFoundError(
            f"Missing stock weights directory: {stock_weights_path}. "
            "Run the full tree-building pipeline first."
        )

    # Load rf once here and pass it to both A1 and A2.
    # Matches R Step3_RmRf_Combine_Trees.R which subtracts rf before AP-pruning.
    rf_series = _load_rf()

    print(f"Loading existing candidate matrix: {candidate_path}", flush=True)
    tilted_returns = pd.read_csv(candidate_path)

    if "date_dt" in tilted_returns.columns:
        tilted_returns["date_dt"] = pd.to_datetime(tilted_returns["date_dt"])

    # ============================================================
    # A1: AP-pruning static, no TC
    # ============================================================
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
        rf=rf_series,
    )

    print(f"Selected AP-pruning params: {sel}", flush=True)

    bt_a1.to_csv(out_dir / "backtest_A1_ap_pruning_static_no_tc.csv", index=False)
    w_a1.to_csv(out_dir / "weights_A1_ap_pruning_static_no_tc.csv", index=False)
    diag_a1.to_csv(out_dir / "diagnostics_A1_ap_pruning_static_no_tc.csv", index=False)

    selected_candidates = w_a1["candidate"].tolist()

    # ============================================================
    # A2: same AP-pruning selection, stock-level TC ex-post
    # ============================================================
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
        rf=rf_series,
    )

    bt_a2.to_csv(out_dir / "backtest_A2_ap_pruning_static_stock_level_tc.csv", index=False)
    w_a2.to_csv(out_dir / "weights_A2_ap_pruning_static_stock_level_tc.csv", index=False)
    diag_a2.to_csv(out_dir / "diagnostics_A2_ap_pruning_static_stock_level_tc.csv", index=False)

    # ============================================================
    # B: rolling TC-aware, portfolio-level turnover
    # ============================================================
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

    # ============================================================
    # C: rolling TC-aware, stock-level turnover
    # ============================================================
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

    # ============================================================
    # Align dates + benchmark + combined outputs
    # ============================================================
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


if __name__ == "__main__":
    main()