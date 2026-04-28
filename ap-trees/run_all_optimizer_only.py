from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CHUNK_DIR,
    OUTPUT_DIR,
    FACTOR_DIR,
    DEFAULT_CHARS,
    DEFAULT_Y_MIN,
    DEFAULT_Y_MAX,
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
from data_io import load_yahoo_monthly_benchmark, load_yearly_chunks
from optimizer import ap_pruning_static_optimize, rolling_tc_optimize
from metrics import performance_metrics, add_wealth_drawdown
from plots import make_all_plots


# ============================================================
# Utilities
# ============================================================

def _load_rf() -> np.ndarray | None:
    rf_path = FACTOR_DIR / "rf_factor.csv"

    if not rf_path.exists():
        print(
            f"WARNING: rf_factor.csv not found at {rf_path}. "
            "Using raw returns instead of excess returns.",
            flush=True,
        )
        return None

    rf_raw = pd.read_csv(rf_path, header=None).squeeze().astype(float).to_numpy()

    if float(np.median(np.abs(rf_raw))) < 0.01:
        print(
            f"Loaded rf series ({len(rf_raw)} months) — values appear already in "
            "decimal form, using as-is.",
            flush=True,
        )
        return rf_raw

    print(
        f"Loaded rf series ({len(rf_raw)} months) — dividing by 100 to convert "
        "from percentage points to decimal.",
        flush=True,
    )
    return rf_raw / 100.0


def _align_rf_to_candidate_dates(
    rf: np.ndarray | None,
    full_panel: pd.DataFrame,
    candidate_returns: pd.DataFrame,
) -> np.ndarray | None:
    """
    The lagged trading candidate matrix starts one month after the original panel
    because returns at t are paired with stock weights from t-1. The optimizer
    only accepts an rf vector by row position, so align rf to the first candidate
    month before passing it in.
    """
    if rf is None:
        return None

    months = (
        full_panel[["yy", "mm"]]
        .drop_duplicates()
        .sort_values(["yy", "mm"])
        .reset_index(drop=True)
    )
    month_to_idx = {
        (int(r.yy), int(r.mm)): i for i, r in months.iterrows()
    }

    first = candidate_returns[["yy", "mm"]].iloc[0]
    start_key = (int(first["yy"]), int(first["mm"]))
    start_idx = month_to_idx.get(start_key, 0)

    rf = np.asarray(rf, dtype=float).flatten()
    if start_idx >= len(rf):
        print(
            "WARNING: rf start index is beyond rf length. Using unshifted rf.",
            flush=True,
        )
        return rf

    aligned = rf[start_idx : start_idx + len(candidate_returns)]
    if len(aligned) < len(candidate_returns):
        aligned = np.concatenate(
            [aligned, np.zeros(len(candidate_returns) - len(aligned))]
        )

    print(
        f"Aligned rf to lagged candidate matrix: start offset={start_idx}, "
        f"length={len(aligned)}",
        flush=True,
    )
    return aligned


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


def _parse_yyyy_mm_from_weight_file(path: Path) -> tuple[int, int]:
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse yy/mm from stock-weight file: {path}")
    return int(parts[0]), int(parts[1])


def _read_stock_weights(path: Path) -> pd.DataFrame:
    try:
        return pd.read_pickle(path, compression="gzip")
    except Exception:
        return pd.read_pickle(path)


def _write_stock_weights(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path, compression="gzip")


def build_lagged_candidate_matrix_and_weights(
    panel: pd.DataFrame,
    stock_weights_dir: Path,
    out_candidate_csv: Path,
    out_lagged_stock_weights_dir: Path,
) -> pd.DataFrame:
    """
    Cheap timing fix for AP-Trees without rebuilding trees.

    Existing AP-tree stock-weight files store stock weights formed at month t.
    The unlagged candidate return matrix effectively uses:
        R_{p,t}^{unlagged} = sum_i w_{i,p,t} * r_{i,t}

    This function reconstructs a tradable lagged matrix:
        R_{p,t}^{lagged} = sum_i w_{i,p,t-1} * r_{i,t}

    It also writes a parallel lagged stock-weight directory indexed by the
    current month t, but containing the t-1 formation weights. That lets the
    existing optimizer compute stock-level turnover/costs without changes.
    """
    if out_candidate_csv.exists() and out_lagged_stock_weights_dir.exists():
        print(f"Loading cached lagged candidate matrix: {out_candidate_csv}", flush=True)
        out = pd.read_csv(out_candidate_csv)
        if "date_dt" in out.columns:
            out["date_dt"] = pd.to_datetime(out["date_dt"])
        return out

    files = sorted(stock_weights_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No .pkl stock-weight files found in {stock_weights_dir}. "
            "This AP-Trees lagged fix needs the saved stock_weights_by_month folder."
        )

    panel = panel.copy()
    panel["permno"] = panel["permno"].astype(str)
    panel["yy"] = panel["yy"].astype(int)
    panel["mm"] = panel["mm"].astype(int)
    if "date_dt" in panel.columns:
        panel["date_dt"] = pd.to_datetime(panel["date_dt"])

    required = {"date", "date_dt", "yy", "mm", "permno", "ret"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")

    month_meta = (
        panel[["date", "date_dt", "yy", "mm"]]
        .drop_duplicates(subset=["yy", "mm"])
        .sort_values(["yy", "mm"])
        .reset_index(drop=True)
    )
    month_keys = [(int(r.yy), int(r.mm)) for r in month_meta.itertuples(index=False)]

    returns_by_month = {
        (int(yy), int(mm)): g[["permno", "ret"]]
        .rename(columns={"ret": "curr_ret"})
        .assign(permno=lambda x: x["permno"].astype(str))
        for (yy, mm), g in panel.groupby(["yy", "mm"], sort=True)
    }

    file_by_month = {_parse_yyyy_mm_from_weight_file(f): f for f in files}

    print("Building lagged AP-tree candidate matrix from saved t-1 stock weights...", flush=True)
    print(f"  Source stock weights: {stock_weights_dir}", flush=True)
    print(f"  Output candidate CSV: {out_candidate_csv}", flush=True)
    print(f"  Output lagged weights: {out_lagged_stock_weights_dir}", flush=True)

    out_lagged_stock_weights_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    built = 0

    for idx in range(1, len(month_keys)):
        prev_key = month_keys[idx - 1]
        curr_key = month_keys[idx]
        curr_yy, curr_mm = curr_key

        prev_file = file_by_month.get(prev_key)
        curr_ret = returns_by_month.get(curr_key)
        if prev_file is None or curr_ret is None:
            continue

        if built == 0 or built % 25 == 0:
            print(
                f"  Lagged reconstruction month {idx}/{len(month_keys)-1}: "
                f"using weights {prev_key[0]}-{prev_key[1]:02d} "
                f"for returns {curr_yy}-{curr_mm:02d}",
                flush=True,
            )

        sw_prev = _read_stock_weights(prev_file).copy()
        if sw_prev.empty:
            continue

        sw_prev["permno"] = sw_prev["permno"].astype(str)
        sw_prev["node_id"] = sw_prev["node_id"].astype(str)

        # Plain AP-Trees should use value-weighted node weights. If a legacy
        # file only has tilt_stock_w, fall back to it; otherwise mirror base to
        # tilt so the existing optimizer's default stock_weight_col still works.
        if "base_stock_w" in sw_prev.columns:
            weight_col = "base_stock_w"
        elif "tilt_stock_w" in sw_prev.columns:
            weight_col = "tilt_stock_w"
            sw_prev["base_stock_w"] = sw_prev["tilt_stock_w"].astype(float)
        else:
            raise ValueError(
                f"Stock-weight file {prev_file} has neither base_stock_w nor tilt_stock_w."
            )

        if "tilt_stock_w" not in sw_prev.columns:
            sw_prev["tilt_stock_w"] = sw_prev["base_stock_w"].astype(float)

        merged = sw_prev.merge(curr_ret, on="permno", how="inner")
        if merged.empty:
            continue

        merged["weighted_ret"] = merged[weight_col].astype(float) * merged["curr_ret"].astype(float)
        node_rets = merged.groupby("node_id", sort=False)["weighted_ret"].sum()

        meta = month_meta[(month_meta["yy"] == curr_yy) & (month_meta["mm"] == curr_mm)].iloc[0]
        row = {
            "date": int(meta["date"]),
            "date_dt": pd.to_datetime(meta["date_dt"]),
            "yy": int(curr_yy),
            "mm": int(curr_mm),
        }
        row.update({f"port_{node_id}": float(val) for node_id, val in node_rets.items()})
        rows.append(row)

        # Save lagged stock weights under the current month name so stock-level
        # TC uses the same t-1 formation weights as the lagged return matrix.
        lagged_sw = sw_prev.copy()
        lagged_sw = lagged_sw.merge(curr_ret, on="permno", how="left")
        lagged_sw["ret"] = lagged_sw["curr_ret"]
        lagged_sw = lagged_sw.drop(columns=["curr_ret"], errors="ignore")
        lagged_sw["date"] = int(meta["date"])
        lagged_sw["date_dt"] = pd.to_datetime(meta["date_dt"])
        lagged_sw["yy"] = int(curr_yy)
        lagged_sw["mm"] = int(curr_mm)

        out_file = out_lagged_stock_weights_dir / f"{curr_yy:04d}_{curr_mm:02d}.pkl"
        _write_stock_weights(lagged_sw, out_file)

        built += 1

    if not rows:
        raise RuntimeError(
            "Unable to build lagged AP-tree candidate matrix. Check that stock_weights_by_month "
            "files and yearly panel months overlap."
        )

    out = pd.DataFrame(rows).sort_values(["yy", "mm"]).reset_index(drop=True)
    out.to_csv(out_candidate_csv, index=False)
    print(
        f"Built lagged AP-tree candidate matrix: {out_candidate_csv} "
        f"({len(out)} months, {len([c for c in out.columns if c.startswith('port_')])} candidates)",
        flush=True,
    )
    return out


def _save_selected_params(out_dir: Path, sel, suffix: str) -> None:
    path = out_dir / f"selected_params_A1_static_no_tc{suffix}.json"
    payload = {
        "lambda0": float(sel.lambda0),
        "lambda2": float(sel.lambda2),
        "k": int(sel.k),
        "val_sharpe": float(sel.val_sharpe),
    }
    path.write_text(json.dumps(payload, indent=2))


def _load_selected_params(out_dir: Path, suffix: str):
    path = out_dir / f"selected_params_A1_static_no_tc{suffix}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ============================================================
# Main
# ============================================================

def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots_lagged_trade"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    original_candidate_path = out_dir / "ap_tree_candidate_matrix.csv"
    lagged_candidate_path = out_dir / "ap_tree_candidate_matrix_lagged_trade.csv"
    stock_weights_path = out_dir / "stock_weights_by_month"
    lagged_stock_weights_path = out_dir / "stock_weights_by_month_lagged_trade"

    if not stock_weights_path.exists():
        raise FileNotFoundError(
            f"Missing stock weights directory: {stock_weights_path}. "
            "Run the full AP-tree-building pipeline first."
        )

    rf_series_raw = _load_rf()

    # Plain AP-trees has no stock_panel_with_market_residual_momentum.csv.
    # The correct return source is the yearly chunk panel loaded here.
    print("Loading yearly chunk panel for lagged AP-tree trading reconstruction...", flush=True)
    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    if original_candidate_path.exists():
        print(
            f"Found original unlagged candidate matrix for reference only: {original_candidate_path}",
            flush=True,
        )

    candidate_returns = build_lagged_candidate_matrix_and_weights(
        panel=panel,
        stock_weights_dir=stock_weights_path,
        out_candidate_csv=lagged_candidate_path,
        out_lagged_stock_weights_dir=lagged_stock_weights_path,
    )

    if "date_dt" in candidate_returns.columns:
        candidate_returns["date_dt"] = pd.to_datetime(candidate_returns["date_dt"])

    rf_series = _align_rf_to_candidate_dates(rf_series_raw, panel, candidate_returns)

    suffix = "_lagged_trade"

    # ============================================================
    # A1: Static AP-pruning, no TC
    # ============================================================
    a1_path = out_dir / f"backtest_A1_static_no_tc{suffix}.csv"
    w_a1_path = out_dir / f"weights_A1_static_no_tc{suffix}.csv"
    diag_a1_path = out_dir / f"diagnostics_A1_static_no_tc{suffix}.csv"

    sel = None

    if a1_path.exists() and w_a1_path.exists():
        print(
            "\n[A1] Loading lagged-trade checkpoint "
            f"(delete {a1_path.name} and {w_a1_path.name} to re-run)...",
            flush=True,
        )
        bt_a1 = pd.read_csv(a1_path)
        bt_a1["date_dt"] = pd.to_datetime(bt_a1["date_dt"])
        w_a1 = pd.read_csv(w_a1_path)

        sel_payload = _load_selected_params(out_dir, suffix)
        if sel_payload is not None:
            from optimizer import SelectedParams
            sel = SelectedParams(
                lambda0=sel_payload["lambda0"],
                lambda2=sel_payload["lambda2"],
                k=sel_payload["k"],
                val_sharpe=sel_payload["val_sharpe"],
            )
    else:
        print("\n[A1] AP-pruning static, no TC on lagged-trading candidate returns...", flush=True)
        bt_a1, w_a1, diag_a1, sel = ap_pruning_static_optimize(
            candidate_returns,
            n_train_valid=N_TRAIN_VALID,
            cv_n=CV_N,
            lambda0_grid=AP_LAMBDA0_GRID,
            lambda2_grid=AP_LAMBDA2_GRID,
            port_n=AP_PORT_N,
            kmin=AP_K_MIN,
            kmax=AP_K_MAX,
            method_name="AP-Trees baseline (static, no TC, lagged trade)",
            cost_per_turnover=0.0,
            stock_weights=lagged_stock_weights_path,
            use_stock_level_turnover=False,
            rf=rf_series,
        )

        print(f"Selected AP-pruning params: {sel}", flush=True)

        bt_a1.to_csv(a1_path, index=False)
        w_a1.to_csv(w_a1_path, index=False)
        diag_a1.to_csv(diag_a1_path, index=False)
        _save_selected_params(out_dir, sel, suffix)

    selected_candidates = w_a1["candidate"].tolist()

    # ============================================================
    # A2: same AP-pruning selection, stock-level TC ex-post
    # ============================================================
    a2_path = out_dir / f"backtest_A2_static_stock_level_tc{suffix}.csv"
    w_a2_path = out_dir / f"weights_A2_static_stock_level_tc{suffix}.csv"
    diag_a2_path = out_dir / f"diagnostics_A2_static_stock_level_tc{suffix}.csv"

    if a2_path.exists():
        print(
            "\n[A2] Loading lagged-trade checkpoint "
            f"(delete {a2_path.name} to re-run)...",
            flush=True,
        )
        bt_a2 = pd.read_csv(a2_path)
        bt_a2["date_dt"] = pd.to_datetime(bt_a2["date_dt"])
    else:
        if sel is None:
            raise RuntimeError(
                "A2 needs A1 selected params. Delete A1/A2 lagged checkpoint files "
                "and rerun, or make sure the lagged selected-params JSON exists."
            )

        print("\n[A2] Same AP-pruning selection, stock-level TC ex-post on lagged trade weights...", flush=True)
        bt_a2, w_a2, diag_a2, _ = ap_pruning_static_optimize(
            candidate_returns,
            n_train_valid=N_TRAIN_VALID,
            cv_n=CV_N,
            lambda0_grid=[sel.lambda0],
            lambda2_grid=[sel.lambda2],
            port_n=AP_PORT_N,
            kmin=sel.k,
            kmax=sel.k,
            method_name="AP-Trees static + stock-level TC (lagged trade)",
            cost_per_turnover=TC_COST,
            stock_weights=lagged_stock_weights_path,
            use_stock_level_turnover=USE_STOCK_LEVEL_TURNOVER,
            rf=rf_series,
        )

        bt_a2.to_csv(a2_path, index=False)
        w_a2.to_csv(w_a2_path, index=False)
        diag_a2.to_csv(diag_a2_path, index=False)

    # ============================================================
    # B: rolling TC-aware, portfolio-level turnover
    # ============================================================
    b_path = out_dir / f"backtest_B_rolling_tc_portfolio_level_tc{suffix}.csv"
    w_b_path = out_dir / f"weights_B_rolling_tc_portfolio_level_tc{suffix}.csv"

    if b_path.exists():
        print(
            "\n[B] Loading lagged-trade checkpoint "
            f"(delete {b_path.name} to re-run)...",
            flush=True,
        )
        bt_b = pd.read_csv(b_path)
        bt_b["date_dt"] = pd.to_datetime(bt_b["date_dt"])
    else:
        print("\n[B] TC-aware rolling ablation, portfolio-level turnover on lagged candidate returns...", flush=True)
        bt_b, w_b = rolling_tc_optimize(
            candidate_returns,
            window=ROLLING_WINDOW,
            lambda_l2=TC_LAMBDA_L2,
            lambda_tc=TC_LAMBDA_TC,
            eta=TC_ETA,
            cost_per_turnover=TC_COST,
            method_name="AP-Trees rolling TC-aware (portfolio-level TC, lagged trade)",
            turnover_mode="portfolio",
            stock_weights=None,
            selected_candidates=selected_candidates,
            long_only=TC_LONG_ONLY,
            rf=rf_series,
        )

        bt_b.to_csv(b_path, index=False)
        w_b.to_csv(w_b_path, index=False)

    # ============================================================
    # C: rolling TC-aware, stock-level turnover
    # ============================================================
    c_path = out_dir / f"backtest_C_rolling_tc_stock_level_tc{suffix}.csv"
    w_c_path = out_dir / f"weights_C_rolling_tc_stock_level_tc{suffix}.csv"

    if c_path.exists():
        print(
            "\n[C] Loading lagged-trade checkpoint "
            f"(delete {c_path.name} to re-run)...",
            flush=True,
        )
        bt_c = pd.read_csv(c_path)
        bt_c["date_dt"] = pd.to_datetime(bt_c["date_dt"])
    else:
        print("\n[C] TC-aware rolling ablation, stock-level turnover on lagged stock weights...", flush=True)
        bt_c, w_c = rolling_tc_optimize(
            candidate_returns,
            window=ROLLING_WINDOW,
            lambda_l2=TC_LAMBDA_L2,
            lambda_tc=TC_LAMBDA_TC,
            eta=TC_ETA,
            cost_per_turnover=TC_COST,
            method_name="AP-Trees rolling TC-aware (stock-level TC, lagged trade)",
            turnover_mode="stock",
            stock_weights=lagged_stock_weights_path,
            selected_candidates=selected_candidates,
            long_only=TC_LONG_ONLY,
            rf=rf_series,
        )

        bt_c.to_csv(c_path, index=False)
        w_c.to_csv(w_c_path, index=False)

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

    # Main combined file now reflects the lagged-trade results.
    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)
    backtest.to_csv(out_dir / f"backtest_comparison{suffix}.csv", index=False)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown.csv", index=False)
    enriched.to_csv(out_dir / f"backtest_comparison_with_wealth_drawdown{suffix}.csv", index=False)

    metrics = performance_metrics(backtest)
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)
    metrics.to_csv(out_dir / f"summary_metrics_comparison{suffix}.csv", index=False)

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)

    print("\nMaking plots...", flush=True)
    make_all_plots(backtest, plot_dir)

    print(f"\nDone. Outputs saved to: {out_dir}", flush=True)
    print(f"Plots saved to: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
