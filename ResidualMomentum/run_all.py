from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
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
    stock weights from t-1. optimizer.py aligns rf by row position, not date,
    so we pass an rf array already aligned to the lagged candidate dates.
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
    """
    Load one monthly stock-weight file.

    Supports parquet, pkl, and csv because different runs may save different
    formats.
    """
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
    """
    Write lagged stock weights using the optimizer's expected monthly format.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{int(yy)}_{int(mm):02d}.parquet"
    df.to_parquet(out_path, index=False)


def _standardize_stock_weight_columns(sw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the stock-weight file before lagged reconstruction.
    """
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
    Build tradable candidate returns using t-1 stock weights.

    Existing same-month candidate matrix is approximately:

        R_{p,t}^{old} = sum_i w_{i,p,t} r_{i,t}

    This reconstructs:

        R_{p,t}^{lagged} = sum_i w_{i,p,t-1} r_{i,t}

    It also writes a lagged stock-weight directory where file t contains the
    stock weights from t-1 relabeled to month t. That way A2/C stock-level
    turnover and transaction costs use the same tradable timing convention.
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

    if "date_dt" in panel.columns:
        panel["date_dt"] = pd.to_datetime(panel["date_dt"])
    else:
        panel["date_dt"] = pd.to_datetime(
            dict(
                year=panel["yy"].astype(int),
                month=panel["mm"].astype(int),
                day=1,
            )
        ) + pd.offsets.MonthEnd(0)

    if "date" not in panel.columns:
        panel["date"] = panel["date_dt"].dt.strftime("%Y%m%d").astype(int)

    ret_panel = panel[["yy", "mm", "date", "date_dt", "permno", "ret"]].copy()
    ret_panel["ret"] = pd.to_numeric(ret_panel["ret"], errors="coerce")

    months = (
        ret_panel[["yy", "mm", "date", "date_dt"]]
        .drop_duplicates()
        .sort_values(["yy", "mm"])
        .reset_index(drop=True)
    )

    original_cols = [c for c in original_candidate_matrix.columns if c.startswith("port_")]
    node_order = [c[len("port_"):] for c in original_cols]

    if not original_cols:
        raise ValueError(
            "Original candidate matrix has no columns starting with 'port_'. "
            "Check select_candidate_matrix(candidate_returns, prefix='tilt_')."
        )

    rows: list[dict] = []

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

        curr_ret = (
            ret_panel[
                ret_panel["yy"].eq(int(curr["yy"]))
                & ret_panel["mm"].eq(int(curr["mm"]))
            ][["permno", "ret"]]
            .dropna(subset=["ret"])
            .rename(columns={"ret": "curr_ret"})
        )

        merged = prev_sw.merge(curr_ret, on="permno", how="inner")

        row = {
            "date": curr["date"],
            "date_dt": curr["date_dt"],
            "yy": int(curr["yy"]),
            "mm": int(curr["mm"]),
        }

        if merged.empty:
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

        lagged_sw = prev_sw.copy()
        lagged_sw["yy"] = int(curr["yy"])
        lagged_sw["mm"] = int(curr["mm"])
        lagged_sw["date"] = curr["date"]
        lagged_sw["date_dt"] = curr["date_dt"]

        _write_stock_weights_file(
            lagged_sw,
            out_lagged_stock_weights_dir,
            int(curr["yy"]),
            int(curr["mm"]),
        )

        if idx == 1 or idx % 50 == 0:
            print(
                f"  Built lagged month {idx}/{len(months) - 1}: "
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
    """
    Align all backtest result frames to the same date range.
    """
    common_start = max(df["date_dt"].min() for df in dfs)
    common_end = min(df["date_dt"].max() for df in dfs)

    out = []

    for df in dfs:
        out.append(
            df[
                df["date_dt"].ge(common_start)
                & df["date_dt"].le(common_end)
            ].copy()
        )

    print(f"Common test window: {common_start.date()} to {common_end.date()}", flush=True)

    return out


def _final_backtest_output_path() -> Path:
    """
    Save the final AP + RM comparison CSV directly into the backtest folder.

    If this file is located inside:
        forest-through-the-trees/backtest/run_all.py

    then the output becomes:
        forest-through-the-trees/backtest/backtest_comparison_ap_rm.csv
    """
    backtest_dir = Path(__file__).resolve().parent
    backtest_dir.mkdir(parents=True, exist_ok=True)
    return backtest_dir / "backtest_comparison_ap_rm.csv"


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    stock_weights_path = out_dir / f"stock_weights_by_month_tau_{DEFAULT_TAU}"

    if stock_weights_path.exists():
        shutil.rmtree(stock_weights_path)

    stock_weights_path.mkdir(parents=True, exist_ok=True)

    rf_series_original = _load_rf()

    # ============================================================
    # 1. Load stock panel
    # ============================================================
    print(f"Loading stock-month yearly chunks for {subdir}...", flush=True)

    panel = load_yearly_chunks(
        CHUNK_DIR,
        chars,
        DEFAULT_Y_MIN,
        DEFAULT_Y_MAX,
    )

    # ============================================================
    # 2. Compute residual momentum signal
    # ============================================================
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

    stock_panel_path = out_dir / f"stock_panel_with_{signal_version}.csv"
    panel.to_csv(stock_panel_path, index=False)

    # ============================================================
    # 3. Build AP-tree candidate returns with residual momentum tilt
    # ============================================================
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

    candidate_returns_path = out_dir / f"candidate_returns_tau_{DEFAULT_TAU}.csv"
    node_stats_path = out_dir / f"node_stats_tau_{DEFAULT_TAU}.csv"
    tilted_candidate_path = out_dir / f"tilted_candidate_matrix_tau_{DEFAULT_TAU}.csv"

    candidate_returns.to_csv(candidate_returns_path, index=False)
    node_stats.to_csv(node_stats_path, index=False)

    original_candidates = select_candidate_matrix(candidate_returns, prefix="tilt_")

    if "date_dt" in original_candidates.columns:
        original_candidates["date_dt"] = pd.to_datetime(original_candidates["date_dt"])

    original_candidates.to_csv(tilted_candidate_path, index=False)

    print(f"Saved tilted candidate matrix to: {tilted_candidate_path}", flush=True)

    # ============================================================
    # 4. Build lagged-trade candidate matrix and lagged stock weights
    # ============================================================
    lagged_candidate_path = out_dir / f"tilted_candidate_matrix_tau_{DEFAULT_TAU}_lagged_trade.csv"
    lagged_stock_weights_path = out_dir / f"stock_weights_by_month_tau_{DEFAULT_TAU}_lagged_trade"

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
    # 5. A1: AP-pruning static, no TC
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

    bt_a1.to_csv(
        out_dir / "backtest_A1_ap_pruning_static_no_tc_lagged_trade.csv",
        index=False,
    )
    w_a1.to_csv(
        out_dir / "weights_A1_ap_pruning_static_no_tc_lagged_trade.csv",
        index=False,
    )
    diag_a1.to_csv(
        out_dir / "diagnostics_A1_ap_pruning_static_no_tc_lagged_trade.csv",
        index=False,
    )

    selected_candidates = w_a1["candidate"].tolist()

    # ============================================================
    # 6. A2: same AP-pruning selection, stock-level TC ex-post
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

    bt_a2.to_csv(
        out_dir / "backtest_A2_ap_pruning_static_stock_level_tc_lagged_trade.csv",
        index=False,
    )
    w_a2.to_csv(
        out_dir / "weights_A2_ap_pruning_static_stock_level_tc_lagged_trade.csv",
        index=False,
    )
    diag_a2.to_csv(
        out_dir / "diagnostics_A2_ap_pruning_static_stock_level_tc_lagged_trade.csv",
        index=False,
    )

    # ============================================================
    # 7. B: rolling TC-aware, portfolio-level turnover
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

    bt_b.to_csv(
        out_dir / "backtest_B_rolling_tc_portfolio_level_tc_lagged_trade.csv",
        index=False,
    )
    w_b.to_csv(
        out_dir / "weights_B_rolling_tc_portfolio_level_tc_lagged_trade.csv",
        index=False,
    )

    # ============================================================
    # 8. C: rolling TC-aware, stock-level turnover
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

    bt_c.to_csv(
        out_dir / "backtest_C_rolling_tc_stock_level_tc_lagged_trade.csv",
        index=False,
    )
    w_c.to_csv(
        out_dir / "weights_C_rolling_tc_stock_level_tc_lagged_trade.csv",
        index=False,
    )

    # ============================================================
    # 9. Align dates + benchmark + combined outputs
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

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)

    print("\nDone.", flush=True)
    print(f"Tree/candidate outputs saved to: {out_dir}", flush=True)
    print(f"Lagged candidate matrix saved to: {lagged_candidate_path}", flush=True)
    print(f"Lagged stock weights saved to: {lagged_stock_weights_path}", flush=True)
    print(f"Internal enriched comparison saved to: {enriched_internal_path}", flush=True)
    print(f"Final AP + RM backtest saved to: {final_output_path}", flush=True)


if __name__ == "__main__":
    main()