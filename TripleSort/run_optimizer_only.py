from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


def _bootstrap_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_bootstrap_imports()

from config import (
    CHUNK_DIR, DEFAULT_CHARS, DEFAULT_Y_MIN, DEFAULT_Y_MAX,
    N_TRAIN_VALID, CV_N, STATIC_MU0_GRID, STATIC_LAMBDA_L2_GRID,
    STATIC_K, STATIC_K_MIN, STATIC_K_MAX, LONG_ONLY,
    ROLLING_WINDOW, TC_COST, TC_LAMBDA_L2, TC_LAMBDA_TC,
    TC_ETA, TC_LONG_ONLY, OUTPUT_DIR, TS32_DIR, TS64_DIR,
)
from data_io import (
    load_triplesort_excess_returns,
    load_yearly_chunks,
    load_yahoo_monthly_benchmark,
)
from optimizer import static_paper_style_optimize, rolling_tc_optimize
from metrics import performance_metrics, add_wealth_drawdown
from plots import make_all_plots


TS_SPECS = [
    ("TS32", TS32_DIR, (2, 4, 4)),
    ("TS64", TS64_DIR, (4, 4, 4)),
]


def _a1_selected_params(w_a1: pd.DataFrame, diag_a1: pd.DataFrame):
    best_lam2 = float(diag_a1["best_lambda_l2"].dropna().iloc[0])
    best_lam0 = float(diag_a1["best_lambda0"].dropna().iloc[0])

    weight_col = "weight" if "weight" in w_a1.columns else "weight_trade"
    k = int((w_a1[weight_col].abs() > 1e-12).sum())

    return best_lam0, best_lam2, k


def _run_one_triplesort(
    ts_name: str,
    ts_dir: Path,
    ts_bins: tuple[int, int, int],
    subdir: str,
    panel: pd.DataFrame,
    combined_out_dir: Path,
) -> list[pd.DataFrame]:
    print(f"\n================ {ts_name} ================", flush=True)
    print(f"Using bins {ts_bins}", flush=True)
    print(f"Loading triple-sort candidate returns ({ts_name}) for {subdir}...", flush=True)

    portfolio_csv = ts_dir / subdir / "excess_ports.csv"
    if not portfolio_csv.exists():
        raise FileNotFoundError(
            f"Missing {portfolio_csv}. Run the portfolio generation step for {ts_name} first."
        )

    returns = load_triplesort_excess_returns(portfolio_csv)

    ts_out_dir = combined_out_dir / ts_name
    ts_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running A1: {ts_name} static no TC — grid search...", flush=True)
    bt_a1, w_a1, diag_a1 = static_paper_style_optimize(
        returns,
        n_train_valid=N_TRAIN_VALID,
        panel=None,
        n_bins=ts_bins,
        cv_n=CV_N,
        long_only=LONG_ONLY,
        method_name=f"{ts_name} static (no TC)",
        cost_per_turnover=0.0,
        use_stock_level_turnover=False,
        mu0_grid=STATIC_MU0_GRID,
        lambda_l2_grid=STATIC_LAMBDA_L2_GRID,
        k_target=STATIC_K,
        k_min=STATIC_K_MIN,
        k_max=STATIC_K_MAX,
    )

    best_lam0, best_lam2, best_k = _a1_selected_params(w_a1, diag_a1)

    print(f"Running A2: {ts_name} same A1 selection + ex-post stock-level TC...", flush=True)
    bt_a2, w_a2, diag_a2 = static_paper_style_optimize(
        returns,
        n_train_valid=N_TRAIN_VALID,
        panel=panel,
        n_bins=ts_bins,
        cv_n=CV_N,
        long_only=LONG_ONLY,
        method_name=f"{ts_name} static + stock-level TC",
        cost_per_turnover=TC_COST,
        use_stock_level_turnover=True,
        mu0_grid=[best_lam0],
        lambda_l2_grid=[best_lam2],
        k_target=best_k,
        k_min=best_k,
        k_max=best_k,
    )

    print(f"Running B: {ts_name} rolling TC-aware portfolio-level TC...", flush=True)
    bt_b, w_b = rolling_tc_optimize(
        returns,
        window=ROLLING_WINDOW,
        panel=None,
        n_bins=ts_bins,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        long_only=TC_LONG_ONLY,
        method_name=f"{ts_name} rolling TC-aware + portfolio-level TC",
        turnover_mode="portfolio",
    )

    print(f"Running C: {ts_name} rolling TC-aware stock-level TC...", flush=True)
    bt_c, w_c = rolling_tc_optimize(
        returns,
        window=ROLLING_WINDOW,
        panel=panel,
        n_bins=ts_bins,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        long_only=TC_LONG_ONLY,
        method_name=f"{ts_name} rolling TC-aware + stock-level TC",
        turnover_mode="stock",
    )

    bt_a1.to_csv(ts_out_dir / f"backtest_A1_{ts_name}_static_no_tc.csv", index=False)
    w_a1.to_csv(ts_out_dir / f"weights_A1_{ts_name}_static_no_tc.csv", index=False)
    diag_a1.to_csv(ts_out_dir / f"diagnostics_A1_{ts_name}_static_no_tc.csv", index=False)

    bt_a2.to_csv(ts_out_dir / f"backtest_A2_{ts_name}_static_stock_level_tc.csv", index=False)
    w_a2.to_csv(ts_out_dir / f"weights_A2_{ts_name}_static_stock_level_tc.csv", index=False)
    diag_a2.to_csv(ts_out_dir / f"diagnostics_A2_{ts_name}_static_stock_level_tc.csv", index=False)

    bt_b.to_csv(ts_out_dir / f"backtest_B_{ts_name}_rolling_tc_portfolio_level_tc.csv", index=False)
    w_b.to_csv(ts_out_dir / f"weights_B_{ts_name}_rolling_tc_portfolio_level_tc.csv", index=False)

    bt_c.to_csv(ts_out_dir / f"backtest_C_{ts_name}_rolling_tc_stock_level_tc.csv", index=False)
    w_c.to_csv(ts_out_dir / f"weights_C_{ts_name}_rolling_tc_stock_level_tc.csv", index=False)

    return [bt_a1, bt_a2, bt_b, bt_c]


def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Loading stock-month panel for stock-level turnover...", flush=True)
    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    all_bt_pieces: list[pd.DataFrame] = []

    for ts_name, ts_dir, ts_bins in TS_SPECS:
        pieces = _run_one_triplesort(
            ts_name=ts_name,
            ts_dir=ts_dir,
            ts_bins=ts_bins,
            subdir=subdir,
            panel=panel,
            combined_out_dir=out_dir,
        )
        all_bt_pieces.extend(pieces)

    common_start = max(bt["date_dt"].min() for bt in all_bt_pieces)
    common_end = min(bt["date_dt"].max() for bt in all_bt_pieces)

    def _trim(bt: pd.DataFrame) -> pd.DataFrame:
        return bt[
            (bt["date_dt"] >= common_start)
            & (bt["date_dt"] <= common_end)
        ].copy()

    all_bt_pieces = [_trim(bt) for bt in all_bt_pieces]

    try:
        bt_sp = load_yahoo_monthly_benchmark(
            ticker="SPY",
            start_date=common_start,
            end_date=common_end,
            method_name="S&P 500 (SPY adjusted close)",
        )

        common_dates = sorted(set.intersection(*[
            set(bt["date_dt"]) for bt in all_bt_pieces
        ]))

        bt_sp = bt_sp[bt_sp["date_dt"].isin(common_dates)].copy()
        all_bt_pieces.append(bt_sp)

    except Exception as e:
        print(f"WARNING: Could not load SPY benchmark: {e}", flush=True)

    backtest = pd.concat(all_bt_pieces, ignore_index=True)
    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)

    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown.csv", index=False)

    metrics = performance_metrics(backtest)
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)

    print("\n================ SUMMARY METRICS ================", flush=True)
    print(metrics.to_string(index=False), flush=True)

    make_all_plots(backtest, plot_dir)

    print(f"\nDone. Combined outputs saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()