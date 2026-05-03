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
    CHUNK_DIR,
    DEFAULT_CHARS,
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
    TS64_DIR,
)

from check_portfolios import check_all

from data_io import (
    load_triplesort_excess_returns,
    load_yearly_chunks,
    load_yahoo_monthly_benchmark,
)

from optimizer import static_paper_style_optimize, rolling_tc_optimize
from metrics import performance_metrics, add_wealth_drawdown


TS_SPECS = [
    ("TS32", TS32_DIR, (2, 4, 4)),
    ("TS64", TS64_DIR, (4, 4, 4)),
]


def _a1_selected_params(w_a1: pd.DataFrame, diag_a1: pd.DataFrame):
    best_lam2 = float(diag_a1["best_lambda_l2"].dropna().iloc[0])
    best_lam0 = float(diag_a1["best_lambda0"].dropna().iloc[0])

    weight_col = "weight" if "weight" in w_a1.columns else "weight_trade"
    best_k = int((w_a1[weight_col].abs() > 1e-12).sum())

    return best_lam0, best_lam2, best_k


def _final_backtest_output_path() -> Path:
    """
    Save final combined TS output into the project-level backtest folder.

    If this script is in:
        forest-through-the-trees/TripleSort/run_all.py

    this writes to:
        forest-through-the-trees/backtest/backtest_comparison_ts.csv
    """
    here = Path(__file__).resolve().parent

    if here.name.lower() == "backtest":
        backtest_dir = here
    else:
        backtest_dir = here.parent / "backtest"

    backtest_dir.mkdir(parents=True, exist_ok=True)
    return backtest_dir / "backtest_comparison_ts.csv"


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
            f"Missing {portfolio_csv}. Run/check the triple-sort portfolio "
            f"generation step for {ts_name} first."
        )

    returns = load_triplesort_excess_returns(portfolio_csv)

    ts_out_dir = combined_out_dir / ts_name
    ts_out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # A1: static no TC
    # ============================================================
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

    print(
        f"{ts_name} selected params: "
        f"lambda0={best_lam0}, lambda_l2={best_lam2}, k={best_k}",
        flush=True,
    )

    # ============================================================
    # A2: same static selection + stock-level TC ex-post
    # ============================================================
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

    # ============================================================
    # B: rolling TC-aware, portfolio-level turnover
    # ============================================================
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

    # ============================================================
    # C: rolling TC-aware, stock-level turnover
    # ============================================================
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

    # ============================================================
    # Save individual TS outputs
    # ============================================================
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
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Triple-sort building/checking step
    # ============================================================
    print("Checking/building triple-sort portfolio parity vs Data/ ...", flush=True)
    check_all()

    # ============================================================
    # 2. Load stock panel for stock-level turnover
    # ============================================================
    print("Loading stock-month panel for stock-level turnover...", flush=True)

    panel = load_yearly_chunks(
        CHUNK_DIR,
        chars,
        DEFAULT_Y_MIN,
        DEFAULT_Y_MAX,
    )

    # ============================================================
    # 3. Run TS32 and TS64
    # ============================================================
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

    if not all_bt_pieces:
        raise RuntimeError("No backtest pieces were generated.")

    # ============================================================
    # 4. Align all TS32/TS64 variants to common window
    # ============================================================
    common_start = max(bt["date_dt"].min() for bt in all_bt_pieces)
    common_end = min(bt["date_dt"].max() for bt in all_bt_pieces)

    print(f"\nCommon test window: {common_start.date()} to {common_end.date()}", flush=True)

    def _trim(bt: pd.DataFrame) -> pd.DataFrame:
        return bt[
            bt["date_dt"].ge(common_start)
            & bt["date_dt"].le(common_end)
        ].copy()

    all_bt_pieces = [_trim(bt) for bt in all_bt_pieces]

    # ============================================================
    # 5. Add S&P 500 benchmark
    # ============================================================
    try:
        print("\nLoading S&P 500 benchmark from Yahoo Finance using SPY...", flush=True)

        bt_spy = load_yahoo_monthly_benchmark(
            ticker="SPY",
            start_date=common_start,
            end_date=common_end,
            method_name="S&P 500 (SPY adjusted close)",
        )

        common_dates = sorted(
            set.intersection(*[set(bt["date_dt"]) for bt in all_bt_pieces])
        )

        bt_spy = bt_spy[bt_spy["date_dt"].isin(common_dates)].copy()
        all_bt_pieces.append(bt_spy)

    except Exception as e:
        print(f"WARNING: Could not load SPY benchmark: {e}", flush=True)

    # ============================================================
    # 6. Save combined outputs
    # ============================================================
    backtest = pd.concat(all_bt_pieces, ignore_index=True)
    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)

    comparison_path = out_dir / "backtest_comparison_lagged_trade.csv"
    backtest.to_csv(comparison_path, index=False)

    enriched = add_wealth_drawdown(backtest)

    enriched_internal_path = out_dir / "backtest_comparison_with_wealth_drawdown_lagged_trade.csv"
    enriched.to_csv(enriched_internal_path, index=False)

    final_output_path = _final_backtest_output_path()
    enriched.to_csv(final_output_path, index=False)

    metrics = performance_metrics(backtest)
    metrics_path = out_dir / "summary_metrics_comparison_lagged_trade.csv"
    metrics.to_csv(metrics_path, index=False)

    print("\n================ SUMMARY METRICS ================", flush=True)
    print(metrics.to_string(index=False), flush=True)

    print("\nDone.", flush=True)
    print(f"Combined raw comparison saved to: {comparison_path}", flush=True)
    print(f"Internal enriched comparison saved to: {enriched_internal_path}", flush=True)
    print(f"Final TS backtest saved to: {final_output_path}", flush=True)
    print(f"Summary metrics saved to: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()