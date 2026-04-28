from __future__ import annotations

from pathlib import Path

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


# =============================================================================
# RF helpers
# =============================================================================

def _load_rf() -> np.ndarray | None:
    """
    Load monthly risk-free rates from rf_factor.csv and convert to decimal.

    R Step3_RmRf_Combine_Trees.R does:
        port_ret[, i] = port_ret[, i] - rf / 100
    meaning rf_factor.csv stores values as percentage points, e.g. 0.45 means
    0.45% per month. We divide by 100 unless the values already look decimal.
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

    if float(np.nanmedian(np.abs(rf_raw))) < 0.01:
        print(
            f"Loaded rf series ({len(rf_raw)} months) — values appear already "
            "in decimal form, using as-is.",
            flush=True,
        )
        return rf_raw

    print(
        f"Loaded rf series ({len(rf_raw)} months) — dividing by 100 to convert "
        "from percentage points to decimal.",
        flush=True,
    )
    return rf_raw / 100.0


def _align_rf_to_lagged_dates(
    rf: np.ndarray | None,
    original_candidate_dates: pd.Series,
    lagged_candidate_dates: pd.Series,
) -> np.ndarray | None:
    """
    The lagged candidate matrix drops the first month because month t uses
    stock weights from t-1. Optimizer.py aligns rf by row position, not date, so
    we must pass an rf array already aligned to the lagged candidate dates.
    """
    if rf is None:
        return None

    original_dates = pd.to_datetime(original_candidate_dates).reset_index(drop=True)
    lagged_dates = pd.to_datetime(lagged_candidate_dates).reset_index(drop=True)

    n = min(len(rf), len(original_dates))
    rf_by_date = pd.Series(np.asarray(rf[:n], dtype=float), index=original_dates.iloc[:n])
    aligned = rf_by_date.reindex(lagged_dates)

    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        print(
            f"WARNING: {missing} lagged candidate months do not have matching rf. "
            "Filling missing rf with 0.",
            flush=True,
        )
        aligned = aligned.fillna(0.0)

    return aligned.to_numpy(dtype=float)


# =============================================================================
# Lagged trading-return reconstruction
# =============================================================================

def _load_stock_weights_file(stock_weights_dir: Path, yy: int, mm: int) -> pd.DataFrame:
    """Load one monthly stock-weight file. Supports parquet, pkl, and csv."""
    candidates = [
        stock_weights_dir / f"{int(yy)}_{int(mm):02d}.parquet",
        stock_weights_dir / f"{int(yy):04d}_{int(mm):02d}.parquet",
        stock_weights_dir / f"{int(yy)}_{int(mm):02d}.pkl",
        stock_weights_dir / f"{int(yy):04d}_{int(mm):02d}.pkl",
        stock_weights_dir / f"{int(yy)}_{int(mm):02d}.csv",
        stock_weights_dir / f"{int(yy):04d}_{int(mm):02d}.csv",
    ]

    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".pkl":
                return pd.read_pickle(path, compression="gzip")
            if path.suffix == ".csv":
                return pd.read_csv(path)

    return pd.DataFrame()


def _write_stock_weights_file(df: pd.DataFrame, out_dir: Path, yy: int, mm: int) -> None:
    """Write lagged stock weights using the AP/RM optimizer's expected format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{int(yy)}_{int(mm):02d}.parquet"
    df.to_parquet(out_path, index=False)


def _standardize_stock_weight_columns(sw: pd.DataFrame) -> pd.DataFrame:
    sw = sw.copy()

    required = {"node_id", "permno", "tilt_stock_w"}
    missing = required - set(sw.columns)
    if missing:
        raise ValueError(f"Stock-weight file missing required columns: {sorted(missing)}")

    sw["node_id"] = sw["node_id"].astype(str)
    sw["permno"] = sw["permno"].astype(str)
    sw["tilt_stock_w"] = sw["tilt_stock_w"].astype(float)

    return sw


def build_lagged_candidate_matrix_and_weights(
    panel: pd.DataFrame,
    stock_weights_dir: Path,
    original_candidate_matrix: pd.DataFrame,
    out_candidate_path: Path,
    out_lagged_stock_weights_dir: Path,
) -> pd.DataFrame:
    """
    Build a trading-correct candidate return matrix without rebuilding trees.

    Existing candidate matrix has approximately:
        R_{p,t}^{old} = sum_i w_{i,p,t} r_{i,t}

    This function reconstructs:
        R_{p,t}^{lagged} = sum_i w_{i,p,t-1} r_{i,t}

    using saved stock weights from t-1 and realized stock returns from t.

    It also writes a lagged stock-weight directory where file t contains the
    stock weights from t-1 but relabeled to month t, so A2/C stock-level turnover
    is computed from the same tradable timing convention.
    """
    if out_candidate_path.exists() and out_lagged_stock_weights_dir.exists():
        print(f"Loading cached lagged candidate matrix: {out_candidate_path}", flush=True)
        out = pd.read_csv(out_candidate_path)
        if "date_dt" in out.columns:
            out["date_dt"] = pd.to_datetime(out["date_dt"])
        return out

    print("Building lagged candidate matrix from saved t-1 stock weights...", flush=True)
    print(f"  Source stock weights: {stock_weights_dir}", flush=True)
    print(f"  Output candidate CSV: {out_candidate_path}", flush=True)
    print(f"  Output lagged weights: {out_lagged_stock_weights_dir}", flush=True)

    panel = panel.copy()
    panel["permno"] = panel["permno"].astype(str)
    panel["yy"] = panel["yy"].astype(int)
    panel["mm"] = panel["mm"].astype(int)
    panel["date_dt"] = pd.to_datetime(panel["date_dt"])

    ret_panel = panel[["yy", "mm", "date", "date_dt", "permno", "ret"]].copy()
    ret_panel["ret"] = pd.to_numeric(ret_panel["ret"], errors="coerce")

    months = (
        ret_panel[["yy", "mm", "date", "date_dt"]]
        .drop_duplicates()
        .sort_values(["yy", "mm"])
        .reset_index(drop=True)
    )

    # Preserve the candidate universe/order from the existing CSV. Missing nodes
    # in a month are filled with 0, same spirit as the original candidate matrix.
    original_cols = [c for c in original_candidate_matrix.columns if c.startswith("port_")]
    node_order = [c[len("port_"):] for c in original_cols]

    rows: list[dict] = []

    # Do NOT delete the whole output directory. On Windows/OneDrive this can
    # fail with PermissionError if Explorer/Excel/OneDrive is touching it.
    # We simply create the directory if needed and overwrite each monthly file
    # as it is written below.
    out_lagged_stock_weights_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(1, len(months)):
        prev = months.iloc[idx - 1]
        curr = months.iloc[idx]

        prev_sw = _load_stock_weights_file(
            stock_weights_dir,
            int(prev["yy"]),
            int(prev["mm"]),
        )
        if prev_sw.empty:
            print(
                f"  WARNING: missing previous stock weights for "
                f"{int(prev['yy'])}-{int(prev['mm']):02d}; skipping "
                f"{int(curr['yy'])}-{int(curr['mm']):02d}",
                flush=True,
            )
            continue

        prev_sw = _standardize_stock_weight_columns(prev_sw)

        curr_ret = ret_panel[
            (ret_panel["yy"].eq(int(curr["yy"])))
            & (ret_panel["mm"].eq(int(curr["mm"])))
        ][["permno", "ret"]].dropna(subset=["ret"]).rename(columns={"ret": "curr_ret"})

        # prev_sw may already contain an old same-month "ret" column from the
        # original stock-weight files. Rename current-month returns before the
        # merge so pandas does not create ret_x/ret_y and break the calculation.
        merged = prev_sw.merge(curr_ret, on="permno", how="inner")

        row = {
            "date": curr["date"],
            "date_dt": curr["date_dt"],
            "yy": int(curr["yy"]),
            "mm": int(curr["mm"]),
        }

        if merged.empty:
            # Keep month with zero candidate returns if no overlap, but warn.
            print(
                f"  WARNING: no permno overlap for lagged month "
                f"{int(curr['yy'])}-{int(curr['mm']):02d}; filling candidates with 0.",
                flush=True,
            )
            node_ret = pd.Series(dtype=float)
        else:
            merged["weighted_ret"] = merged["tilt_stock_w"] * merged["curr_ret"]
            node_ret = merged.groupby("node_id", sort=False)["weighted_ret"].sum()

        for node_id, col in zip(node_order, original_cols):
            row[col] = float(node_ret.get(node_id, 0.0))

        rows.append(row)

        # Write previous weights relabeled as current month weights. This lets
        # optimizer.py compute stock-level turnover/cost using tradable timing.
        lagged_sw = prev_sw.copy()
        lagged_sw["yy"] = int(curr["yy"])
        lagged_sw["mm"] = int(curr["mm"])
        lagged_sw["date"] = curr["date"]
        lagged_sw["date_dt"] = curr["date_dt"]
        _write_stock_weights_file(lagged_sw, out_lagged_stock_weights_dir, int(curr["yy"]), int(curr["mm"]))

        if idx == 1 or idx % 50 == 0:
            print(
                f"  Built lagged month {idx}/{len(months)-1}: "
                f"{int(curr['yy'])}-{int(curr['mm']):02d}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("Lagged candidate matrix construction produced zero rows.")

    lagged = pd.DataFrame(rows).sort_values(["yy", "mm"]).reset_index(drop=True)
    lagged.to_csv(out_candidate_path, index=False)

    print(
        f"Finished lagged matrix: {lagged.shape[0]} months, "
        f"{len(original_cols)} candidates.",
        flush=True,
    )

    return lagged


# =============================================================================
# Output helpers
# =============================================================================

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


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    original_candidate_path = out_dir / f"tilted_candidate_matrix_tau_{DEFAULT_TAU}.csv"
    stock_weights_path = out_dir / f"stock_weights_by_month_tau_{DEFAULT_TAU}"
    stock_panel_path = out_dir / "stock_panel_with_market_residual_momentum.csv"

    # New cached trading-correct files. These are cheap to build compared with
    # reconstructing all trees.
    lagged_candidate_path = out_dir / f"tilted_candidate_matrix_tau_{DEFAULT_TAU}_lagged_trade.csv"
    lagged_stock_weights_path = out_dir / f"stock_weights_by_month_tau_{DEFAULT_TAU}_lagged_trade"

    if not original_candidate_path.exists():
        raise FileNotFoundError(
            f"Missing candidate matrix: {original_candidate_path}. "
            "Run the full tree-building pipeline first."
        )

    if not stock_weights_path.exists():
        raise FileNotFoundError(
            f"Missing stock weights directory: {stock_weights_path}. "
            "Run the full tree-building pipeline first."
        )

    rf_series_original = _load_rf()

    print(f"Loading original candidate matrix for column/date reference: {original_candidate_path}", flush=True)
    original_candidates = pd.read_csv(original_candidate_path)
    if "date_dt" in original_candidates.columns:
        original_candidates["date_dt"] = pd.to_datetime(original_candidates["date_dt"])

    print("Loading stock_panel_with_market_residual_momentum.csv for lagged trading reconstruction...", flush=True)
    if not stock_panel_path.exists():
        raise FileNotFoundError(
            f"Missing stock panel CSV: {stock_panel_path}. This lagged runner avoids "
            "Data/data_chunk_files_quantile and uses the already-generated "
            "stock_panel_with_market_residual_momentum.csv file instead."
        )

    panel = pd.read_csv(stock_panel_path)
    if "date_dt" in panel.columns:
        panel["date_dt"] = pd.to_datetime(panel["date_dt"])
    elif {"yy", "mm"}.issubset(panel.columns):
        panel["date_dt"] = pd.to_datetime(
            dict(year=panel["yy"].astype(int), month=panel["mm"].astype(int), day=1)
        ) + pd.offsets.MonthEnd(0)
    else:
        raise ValueError("Panel must contain either date_dt or yy/mm columns.")

    if "date" not in panel.columns:
        panel["date"] = panel["date_dt"].dt.strftime("%Y%m%d").astype(int)

    tilted_returns = build_lagged_candidate_matrix_and_weights(
        panel=panel,
        stock_weights_dir=stock_weights_path,
        original_candidate_matrix=original_candidates,
        out_candidate_path=lagged_candidate_path,
        out_lagged_stock_weights_dir=lagged_stock_weights_path,
    )

    if "date_dt" in tilted_returns.columns:
        tilted_returns["date_dt"] = pd.to_datetime(tilted_returns["date_dt"])

    rf_series = _align_rf_to_lagged_dates(
        rf=rf_series_original,
        original_candidate_dates=original_candidates["date_dt"],
        lagged_candidate_dates=tilted_returns["date_dt"],
    )

    # ============================================================
    # A1: AP-pruning static, no TC
    # ============================================================
    print("\n[A1] AP-pruning static, no TC on lagged-trading candidate returns...", flush=True)

    bt_a1, w_a1, diag_a1, sel = ap_pruning_static_optimize(
        tilted_returns,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda0_grid=AP_LAMBDA0_GRID,
        lambda2_grid=AP_LAMBDA2_GRID,
        port_n=AP_PORT_N,
        kmin=AP_K_MIN,
        kmax=AP_K_MAX,
        method_name="AP-tree + RM AP-pruning (static, no TC, lagged trade)",
        cost_per_turnover=0.0,
        stock_weights=None,
        use_stock_level_turnover=False,
        rf=rf_series,
    )

    print(f"Selected AP-pruning params: {sel}", flush=True)

    bt_a1.to_csv(out_dir / "backtest_A1_ap_pruning_static_no_tc_lagged_trade.csv", index=False)
    w_a1.to_csv(out_dir / "weights_A1_ap_pruning_static_no_tc_lagged_trade.csv", index=False)
    diag_a1.to_csv(out_dir / "diagnostics_A1_ap_pruning_static_no_tc_lagged_trade.csv", index=False)

    selected_candidates = w_a1["candidate"].tolist()

    # ============================================================
    # A2: same AP-pruning selection, stock-level TC ex-post
    # ============================================================
    print("\n[A2] Same AP-pruning selection, stock-level TC ex-post, lagged trade...", flush=True)

    bt_a2, w_a2, diag_a2, _ = ap_pruning_static_optimize(
        tilted_returns,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda0_grid=[sel.lambda0],
        lambda2_grid=[sel.lambda2],
        port_n=AP_PORT_N,
        kmin=AP_K_MIN,
        kmax=AP_K_MAX,
        method_name="AP-tree + RM AP-pruning (static + stock-level TC, lagged trade)",
        cost_per_turnover=TC_COST,
        stock_weights=lagged_stock_weights_path,
        use_stock_level_turnover=USE_STOCK_LEVEL_TURNOVER,
        rf=rf_series,
    )

    bt_a2.to_csv(out_dir / "backtest_A2_ap_pruning_static_stock_level_tc_lagged_trade.csv", index=False)
    w_a2.to_csv(out_dir / "weights_A2_ap_pruning_static_stock_level_tc_lagged_trade.csv", index=False)
    diag_a2.to_csv(out_dir / "diagnostics_A2_ap_pruning_static_stock_level_tc_lagged_trade.csv", index=False)

    # ============================================================
    # B: rolling TC-aware, portfolio-level turnover
    # ============================================================
    print("\n[B] TC-aware rolling ablation, portfolio-level turnover, lagged trade...", flush=True)

    bt_b, w_b = rolling_tc_optimize(
        tilted_returns,
        window=ROLLING_WINDOW,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        method_name="AP-tree + RM rolling TC-aware (portfolio-level TC, lagged trade)",
        turnover_mode="portfolio",
        stock_weights=None,
        selected_candidates=selected_candidates,
        long_only=TC_LONG_ONLY,
        rf=rf_series,
    )

    bt_b.to_csv(out_dir / "backtest_B_rolling_tc_portfolio_level_tc_lagged_trade.csv", index=False)
    w_b.to_csv(out_dir / "weights_B_rolling_tc_portfolio_level_tc_lagged_trade.csv", index=False)

    # ============================================================
    # C: rolling TC-aware, stock-level turnover
    # ============================================================
    print("\n[C] TC-aware rolling ablation, stock-level turnover, lagged trade...", flush=True)

    bt_c, w_c = rolling_tc_optimize(
        tilted_returns,
        window=ROLLING_WINDOW,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        method_name="AP-tree + RM rolling TC-aware (stock-level TC, lagged trade)",
        turnover_mode="stock",
        stock_weights=lagged_stock_weights_path,
        selected_candidates=selected_candidates,
        long_only=TC_LONG_ONLY,
        rf=rf_series,
    )

    bt_c.to_csv(out_dir / "backtest_C_rolling_tc_stock_level_tc_lagged_trade.csv", index=False)
    w_c.to_csv(out_dir / "weights_C_rolling_tc_stock_level_tc_lagged_trade.csv", index=False)

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

    backtest.to_csv(out_dir / "backtest_comparison_lagged_trade.csv", index=False)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown_lagged_trade.csv", index=False)

    metrics = performance_metrics(backtest)
    metrics.to_csv(out_dir / "summary_metrics_comparison_lagged_trade.csv", index=False)

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)

    print("\nMaking plots...", flush=True)
    make_all_plots(backtest, plot_dir)

    print(f"\nDone. Outputs saved to: {out_dir}", flush=True)
    print(f"Lagged candidate matrix saved to: {lagged_candidate_path}", flush=True)
    print(f"Lagged stock weights saved to: {lagged_stock_weights_path}", flush=True)
    print(f"Plots saved to: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
