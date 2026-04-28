from __future__ import annotations

import json
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


def _restore_candidate_matrix_from_stock_weights(
    candidate_path,
    stock_weights_path,
    chars: list[str],
) -> pd.DataFrame:
    files = sorted(stock_weights_path.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(
            f"Missing stock-weight checkpoint files in {stock_weights_path}."
        )

    print(
        f"Candidate matrix missing. Reconstructing from {len(files)} monthly stock-weight files...",
        flush=True,
    )

    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)[
        ["date", "date_dt", "yy", "mm", "permno", "ret"]
    ].copy()
    panel["permno"] = panel["permno"].astype(str)

    month_returns = {
        key: grp[["permno", "ret"]].set_index("permno")["ret"]
        for key, grp in panel.groupby(["yy", "mm"], sort=True)
    }

    month_meta = (
        panel[["date", "date_dt", "yy", "mm"]]
        .drop_duplicates(subset=["yy", "mm"])
        .sort_values(["yy", "mm"])
        .set_index(["yy", "mm"])
    )

    rows = []

    for idx, path in enumerate(files, start=1):
        yy = int(path.stem.split("_")[0])
        mm = int(path.stem.split("_")[1])

        if idx == 1 or idx % 25 == 0:
            print(
                f"  Restoring candidate matrix month {idx}/{len(files)} ({yy}-{mm:02d})",
                flush=True,
            )

        if (yy, mm) not in month_returns:
            continue

        sw = pd.read_pickle(path, compression="gzip")
        sw["permno"] = sw["permno"].astype(str)
        sw["ret"] = sw["permno"].map(month_returns[(yy, mm)]).fillna(0.0)
        sw["weighted_ret"] = sw["base_stock_w"].astype(float) * sw["ret"].astype(float)

        port_ret = sw.groupby("node_id", sort=False)["weighted_ret"].sum()
        meta = month_meta.loc[(yy, mm)]

        row = {
            "date": int(meta["date"]),
            "date_dt": pd.to_datetime(meta["date_dt"]),
            "yy": yy,
            "mm": mm,
        }
        row.update({f"port_{node_id}": value for node_id, value in port_ret.items()})
        rows.append(row)

    if not rows:
        raise ValueError("Unable to reconstruct any candidate-matrix rows from stock weights.")

    out = pd.DataFrame(rows).sort_values(["yy", "mm"]).reset_index(drop=True)
    out.to_csv(candidate_path, index=False)
    print(f"Restored candidate matrix: {candidate_path}", flush=True)
    return out


def _save_selected_params(out_dir, sel) -> None:
    path = out_dir / "selected_params_A1_static_no_tc.json"
    payload = {
        "lambda0": float(sel.lambda0),
        "lambda2": float(sel.lambda2),
        "k": int(sel.k),
        "val_sharpe": float(sel.val_sharpe),
    }
    path.write_text(json.dumps(payload, indent=2))


def _load_selected_params(out_dir):
    path = out_dir / "selected_params_A1_static_no_tc.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = out_dir / "ap_tree_candidate_matrix.csv"
    stock_weights_path = out_dir / "stock_weights_by_month"

    if not stock_weights_path.exists():
        raise FileNotFoundError(
            f"Missing stock weights directory: {stock_weights_path}. "
            "Run the full tree-building pipeline first."
        )

    rf_series = _load_rf()

    if candidate_path.exists():
        print(f"Loading existing candidate matrix: {candidate_path}", flush=True)
        candidate_returns = pd.read_csv(candidate_path)
        if "date_dt" in candidate_returns.columns:
            candidate_returns["date_dt"] = pd.to_datetime(candidate_returns["date_dt"])
    else:
        candidate_returns = _restore_candidate_matrix_from_stock_weights(
            candidate_path,
            stock_weights_path,
            chars,
        )

    # ============================================================
    # A1: Static AP-pruning, no TC
    # ============================================================
    a1_path = out_dir / "backtest_A1_static_no_tc.csv"
    w_a1_path = out_dir / "weights_A1_static_no_tc.csv"
    diag_a1_path = out_dir / "diagnostics_A1_static_no_tc.csv"

    sel = None

    if a1_path.exists() and w_a1_path.exists():
        print(
            "\n[A1] Loading from checkpoint "
            "(delete backtest_A1_static_no_tc.csv and weights_A1_static_no_tc.csv to re-run)...",
            flush=True,
        )
        bt_a1 = pd.read_csv(a1_path)
        bt_a1["date_dt"] = pd.to_datetime(bt_a1["date_dt"])
        w_a1 = pd.read_csv(w_a1_path)

        sel_payload = _load_selected_params(out_dir)
        if sel_payload is not None:
            from optimizer import SelectedParams
            sel = SelectedParams(
                lambda0=sel_payload["lambda0"],
                lambda2=sel_payload["lambda2"],
                k=sel_payload["k"],
                val_sharpe=sel_payload["val_sharpe"],
            )
    else:
        print("\n[A1] AP-pruning static, no TC...", flush=True)
        bt_a1, w_a1, diag_a1, sel = ap_pruning_static_optimize(
            candidate_returns,
            n_train_valid=N_TRAIN_VALID,
            cv_n=CV_N,
            lambda0_grid=AP_LAMBDA0_GRID,
            lambda2_grid=AP_LAMBDA2_GRID,
            port_n=AP_PORT_N,
            kmin=AP_K_MIN,
            kmax=AP_K_MAX,
            method_name="AP-Trees baseline (static, no TC)",
            cost_per_turnover=0.0,
            stock_weights=stock_weights_path,
            use_stock_level_turnover=USE_STOCK_LEVEL_TURNOVER,
            rf=rf_series,
        )

        print(f"Selected AP-pruning params: {sel}", flush=True)

        bt_a1.to_csv(a1_path, index=False)
        w_a1.to_csv(w_a1_path, index=False)
        diag_a1.to_csv(diag_a1_path, index=False)
        _save_selected_params(out_dir, sel)

    selected_candidates = w_a1["candidate"].tolist()

    # ============================================================
    # A2: same AP-pruning selection, stock-level TC ex-post
    # ============================================================
    a2_path = out_dir / "backtest_A2_static_stock_level_tc.csv"
    w_a2_path = out_dir / "weights_A2_static_stock_level_tc.csv"
    diag_a2_path = out_dir / "diagnostics_A2_static_stock_level_tc.csv"

    if a2_path.exists():
        print(
            "\n[A2] Loading from checkpoint "
            "(delete backtest_A2_static_stock_level_tc.csv to re-run)...",
            flush=True,
        )
        bt_a2 = pd.read_csv(a2_path)
        bt_a2["date_dt"] = pd.to_datetime(bt_a2["date_dt"])
    else:
        if sel is None:
            raise RuntimeError(
                "A2 needs A1 selected params to reuse the same AP-pruning selection. "
                "Delete A1/A2 checkpoint files and rerun, or make sure "
                "selected_params_A1_static_no_tc.json exists."
            )

        print("\n[A2] Same AP-pruning selection, stock-level TC ex-post...", flush=True)
        bt_a2, w_a2, diag_a2, _ = ap_pruning_static_optimize(
            candidate_returns,
            n_train_valid=N_TRAIN_VALID,
            cv_n=CV_N,
            lambda0_grid=[sel.lambda0],
            lambda2_grid=[sel.lambda2],
            port_n=AP_PORT_N,
            kmin=sel.k,
            kmax=sel.k,
            method_name="AP-Trees static + stock-level TC",
            cost_per_turnover=TC_COST,
            stock_weights=stock_weights_path,
            use_stock_level_turnover=USE_STOCK_LEVEL_TURNOVER,
            rf=rf_series,
        )

        bt_a2.to_csv(a2_path, index=False)
        w_a2.to_csv(w_a2_path, index=False)
        diag_a2.to_csv(diag_a2_path, index=False)

    # ============================================================
    # B: rolling TC-aware, portfolio-level turnover
    # ============================================================
    b_path = out_dir / "backtest_B_rolling_tc_portfolio_level_tc.csv"
    w_b_path = out_dir / "weights_B_rolling_tc_portfolio_level_tc.csv"

    if b_path.exists():
        print(
            "\n[B] Loading from checkpoint "
            "(delete backtest_B_rolling_tc_portfolio_level_tc.csv to re-run)...",
            flush=True,
        )
        bt_b = pd.read_csv(b_path)
        bt_b["date_dt"] = pd.to_datetime(bt_b["date_dt"])
    else:
        print("\n[B] TC-aware rolling ablation, portfolio-level turnover...", flush=True)
        bt_b, w_b = rolling_tc_optimize(
            candidate_returns,
            window=ROLLING_WINDOW,
            lambda_l2=TC_LAMBDA_L2,
            lambda_tc=TC_LAMBDA_TC,
            eta=TC_ETA,
            cost_per_turnover=TC_COST,
            method_name="AP-Trees rolling TC-aware (portfolio-level TC)",
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
    c_path = out_dir / "backtest_C_rolling_tc_stock_level_tc.csv"
    w_c_path = out_dir / "weights_C_rolling_tc_stock_level_tc.csv"

    if c_path.exists():
        print(
            "\n[C] Loading from checkpoint "
            "(delete backtest_C_rolling_tc_stock_level_tc.csv to re-run)...",
            flush=True,
        )
        bt_c = pd.read_csv(c_path)
        bt_c["date_dt"] = pd.to_datetime(bt_c["date_dt"])
    else:
        print("\n[C] TC-aware rolling ablation, stock-level turnover...", flush=True)
        bt_c, w_c = rolling_tc_optimize(
            candidate_returns,
            window=ROLLING_WINDOW,
            lambda_l2=TC_LAMBDA_L2,
            lambda_tc=TC_LAMBDA_TC,
            eta=TC_ETA,
            cost_per_turnover=TC_COST,
            method_name="AP-Trees rolling TC-aware (stock-level TC)",
            turnover_mode="stock",
            stock_weights=stock_weights_path,
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