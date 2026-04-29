from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd


def _bootstrap_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_bootstrap_imports()

from config import (  # noqa: E402
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
    FACTOR_DIR,
)
from data_io import load_yearly_chunks, load_yahoo_monthly_benchmark  # noqa: E402
from optimizer import static_paper_style_optimize, rolling_tc_optimize  # noqa: E402
from metrics import performance_metrics, add_wealth_drawdown  # noqa: E402
from plots import make_all_plots  # noqa: E402
from utils import ntile_r  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_rf() -> pd.DataFrame | None:
    """Load monthly risk-free rate as decimal, indexed by yy/mm."""
    rf_path = FACTOR_DIR / "rf_factor.csv"
    if not rf_path.exists():
        print(f"WARNING: rf_factor.csv not found at {rf_path}. Using raw returns.", flush=True)
        return None

    rf = pd.read_csv(rf_path, header=None).squeeze().astype(float).reset_index(drop=True)
    if float(np.nanmedian(np.abs(rf))) >= 0.01:
        rf = rf / 100.0

    months = []
    yy, mm = DEFAULT_Y_MIN, 1
    for val in rf:
        months.append({"yy": yy, "mm": mm, "rf": float(val)})
        mm += 1
        if mm == 13:
            mm = 1
            yy += 1

    return pd.DataFrame(months)


def _month_key_df(panel: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [c for c in ["date", "date_dt", "yy", "mm"] if c in panel.columns]
    return (
        panel[meta_cols]
        .drop_duplicates(subset=["yy", "mm"])
        .sort_values(["yy", "mm"])
        .reset_index(drop=True)
    )


def _bucket_id_for_panel(
    df: pd.DataFrame,
    n_bins: tuple[int, int, int],
    feat_cols: tuple[str, str, str] = ("LME", "OP", "Investment"),
) -> pd.Series:
    b1 = ntile_r(df[feat_cols[0]].astype(float), n_bins[0])
    b2 = ntile_r(df[feat_cols[1]].astype(float), n_bins[1])
    b3 = ntile_r(df[feat_cols[2]].astype(float), n_bins[2])

    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    ok = b1.notna() & b2.notna() & b3.notna()
    if not ok.any():
        return out

    n2, n3 = n_bins[1], n_bins[2]
    out.loc[ok] = (
        (b1.loc[ok].astype(int) - 1) * (n2 * n3)
        + (b2.loc[ok].astype(int) - 1) * n3
        + (b3.loc[ok].astype(int) - 1)
        + 1
    ).astype(int)
    return out


def build_lagged_triplesort_returns_and_panel(
    panel: pd.DataFrame,
    n_bins: tuple[int, int, int],
    rf_df: pd.DataFrame | None,
    out_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build lagged triple-sort candidate returns without regenerating portfolio files.

    For month t, form buckets/value weights using month t-1 characteristics and size,
    then apply those stock weights to month t returns:

        R_{p,t} = sum_i w_{i,p,t-1} r_{i,t}.

    Also returns a lagged panel whose row (yy, mm) contains current ret_t but
    formation variables size/chars from t-1, so optimizer stock-level TC uses the
    same timing convention.
    """
    panel = panel.copy().sort_values(["permno", "yy", "mm"]).reset_index(drop=True)
    panel["permno"] = panel["permno"].astype(str)
    panel["date_dt"] = pd.to_datetime(panel["date_dt"])

    # Current returns/meta at month t.
    curr_cols = ["permno", "yy", "mm", "date", "date_dt", "ret"]
    curr = panel[curr_cols].copy()
    curr = curr.rename(columns={"ret": "curr_ret"})

    # Lagged formation variables from t-1, aligned onto month t by permno.
    lag_cols = ["size", "LME", "OP", "Investment"]
    lagged = panel[["permno", *lag_cols]].copy()
    lagged[lag_cols] = lagged.groupby(panel["permno"])[lag_cols].shift(1)

    lagged_panel = pd.concat([curr, lagged[lag_cols]], axis=1)
    lagged_panel = lagged_panel.dropna(subset=["curr_ret", *lag_cols]).copy()
    lagged_panel = lagged_panel.rename(columns={"curr_ret": "ret"})
    lagged_panel["permno"] = lagged_panel["permno"].astype(str)

    n_ports = int(n_bins[0] * n_bins[1] * n_bins[2])
    lagged_panel["bucket_id"] = _bucket_id_for_panel(lagged_panel, n_bins=n_bins)
    lagged_panel = lagged_panel[lagged_panel["bucket_id"].notna()].copy()
    lagged_panel["bucket_id"] = lagged_panel["bucket_id"].astype(int)

    # Value weights inside each lagged bucket, using size_{t-1}.
    lagged_panel = lagged_panel[lagged_panel["size"].astype(float) > 0].copy()
    lagged_panel["sum_size_bucket"] = lagged_panel.groupby(["yy", "mm", "bucket_id"])["size"].transform("sum")
    lagged_panel = lagged_panel[lagged_panel["sum_size_bucket"] > 0].copy()
    lagged_panel["base_w"] = lagged_panel["size"].astype(float) / lagged_panel["sum_size_bucket"].astype(float)
    lagged_panel["weighted_ret"] = lagged_panel["base_w"] * lagged_panel["ret"].astype(float)

    port_rets = (
        lagged_panel.groupby(["yy", "mm", "bucket_id"], sort=True)["weighted_ret"]
        .sum()
        .reset_index()
    )

    meta = _month_key_df(lagged_panel)
    rows = []
    for _, mrow in meta.iterrows():
        yy, mm = int(mrow["yy"]), int(mrow["mm"])
        row = {
            "date": int(mrow["date"]),
            "date_dt": pd.to_datetime(mrow["date_dt"]),
            "yy": yy,
            "mm": mm,
        }
        sub = port_rets[(port_rets["yy"] == yy) & (port_rets["mm"] == mm)]
        vals = dict(zip(sub["bucket_id"].astype(int), sub["weighted_ret"].astype(float)))
        for k in range(1, n_ports + 1):
            val = vals.get(k, 0.0)
            row[f"port_V{k}"] = float(val)
        rows.append(row)

    returns = pd.DataFrame(rows).sort_values(["yy", "mm"]).reset_index(drop=True)

    # Match original triple-sort excess-return convention by subtracting rf_t.
    if rf_df is not None:
        returns = returns.merge(rf_df, on=["yy", "mm"], how="left")
        port_cols = [c for c in returns.columns if c.startswith("port_")]
        returns["rf"] = returns["rf"].fillna(0.0)
        returns[port_cols] = returns[port_cols].subtract(returns["rf"], axis=0)
        returns = returns.drop(columns=["rf"])

    # Optimizer stock-level turnover helpers expect the panel columns but do not
    # need bucket_id/base_w/sum_size_bucket/weighted_ret.
    lagged_panel_for_tc = lagged_panel.drop(
        columns=["bucket_id", "sum_size_bucket", "base_w", "weighted_ret"],
        errors="ignore",
    ).copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(out_csv, index=False)
    print(f"Saved lagged {n_bins} candidate matrix: {out_csv}", flush=True)

    return returns, lagged_panel_for_tc


def _a1_selected_params(w_a1: pd.DataFrame, diag_a1: pd.DataFrame):
    best_lam2 = float(diag_a1["best_lambda_l2"].dropna().iloc[0])
    best_lam0 = float(diag_a1["best_lambda0"].dropna().iloc[0])
    k = int((w_a1["weight"].abs() > 1e-12).sum())
    return best_lam0, best_lam2, k


def _run_one_spec(
    label: str,
    n_bins: tuple[int, int, int],
    returns: pd.DataFrame,
    panel_lagged: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run A1/A2/B/C for one triple-sort spec, e.g. TS32 or TS64."""
    print(f"\n================ {label} ================", flush=True)

    print(f"Running A1: Triple Sort {label} static no TC — grid search...", flush=True)
    bt_a1, w_a1, diag_a1 = static_paper_style_optimize(
        returns,
        n_train_valid=N_TRAIN_VALID,
        panel=None,
        n_bins=n_bins,
        cv_n=CV_N,
        long_only=LONG_ONLY,
        method_name=f"Triple Sort {label} static (no TC, lagged trade)",
        cost_per_turnover=0.0,
        use_stock_level_turnover=False,
        mu0_grid=STATIC_MU0_GRID,
        lambda_l2_grid=STATIC_LAMBDA_L2_GRID,
        k_target=STATIC_K,
        k_min=STATIC_K_MIN,
        k_max=STATIC_K_MAX,
    )

    best_lam0, best_lam2, best_k = _a1_selected_params(w_a1, diag_a1)

    print(f"Running A2: Triple Sort {label} same A1 selection + ex-post stock-level TC...", flush=True)
    bt_a2, w_a2, diag_a2 = static_paper_style_optimize(
        returns,
        n_train_valid=N_TRAIN_VALID,
        panel=panel_lagged,
        n_bins=n_bins,
        cv_n=CV_N,
        long_only=LONG_ONLY,
        method_name=f"Triple Sort {label} static + stock-level TC (lagged trade)",
        cost_per_turnover=TC_COST,
        use_stock_level_turnover=True,
        mu0_grid=[best_lam0],
        lambda_l2_grid=[best_lam2],
        k_target=best_k,
        k_min=best_k,
        k_max=best_k,
    )

    print(f"Running B: Triple Sort {label} rolling TC-aware portfolio-level TC...", flush=True)
    bt_b, w_b = rolling_tc_optimize(
        returns,
        window=ROLLING_WINDOW,
        panel=None,
        n_bins=n_bins,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        long_only=TC_LONG_ONLY,
        method_name=f"Triple Sort {label} rolling TC-aware + portfolio-level TC (lagged trade)",
        turnover_mode="portfolio",
    )

    print(f"Running C: Triple Sort {label} rolling TC-aware stock-level TC...", flush=True)
    bt_c, w_c = rolling_tc_optimize(
        returns,
        window=ROLLING_WINDOW,
        panel=panel_lagged,
        n_bins=n_bins,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        eta=TC_ETA,
        cost_per_turnover=TC_COST,
        long_only=TC_LONG_ONLY,
        method_name=f"Triple Sort {label} rolling TC-aware + stock-level TC (lagged trade)",
        turnover_mode="stock",
    )

    # Save per-spec outputs.
    prefix = label.lower()
    bt_a1.to_csv(out_dir / f"backtest_A1_{prefix}_static_no_tc_lagged_trade.csv", index=False)
    w_a1.to_csv(out_dir / f"weights_A1_{prefix}_static_no_tc_lagged_trade.csv", index=False)
    diag_a1.to_csv(out_dir / f"diagnostics_A1_{prefix}_static_no_tc_lagged_trade.csv", index=False)

    bt_a2.to_csv(out_dir / f"backtest_A2_{prefix}_static_stock_level_tc_lagged_trade.csv", index=False)
    w_a2.to_csv(out_dir / f"weights_A2_{prefix}_static_stock_level_tc_lagged_trade.csv", index=False)
    diag_a2.to_csv(out_dir / f"diagnostics_A2_{prefix}_static_stock_level_tc_lagged_trade.csv", index=False)

    bt_b.to_csv(out_dir / f"backtest_B_{prefix}_rolling_tc_portfolio_level_tc_lagged_trade.csv", index=False)
    w_b.to_csv(out_dir / f"weights_B_{prefix}_rolling_tc_portfolio_level_tc_lagged_trade.csv", index=False)

    bt_c.to_csv(out_dir / f"backtest_C_{prefix}_rolling_tc_stock_level_tc_lagged_trade.csv", index=False)
    w_c.to_csv(out_dir / f"weights_C_{prefix}_rolling_tc_stock_level_tc_lagged_trade.csv", index=False)

    pieces = [bt_a1, bt_a2, bt_b, bt_c]
    return pd.concat(pieces, ignore_index=True), {
        "bt_a1": bt_a1,
        "bt_a2": bt_a2,
        "bt_b": bt_b,
        "bt_c": bt_c,
    }


def _trim_to_common_window(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    common_start = max(df["date_dt"].min() for df in dfs if not df.empty)
    common_end = min(df["date_dt"].max() for df in dfs if not df.empty)
    out = []
    for df in dfs:
        out.append(df[(df["date_dt"] >= common_start) & (df["date_dt"] <= common_end)].copy())
    print(f"Common comparison window: {common_start.date()} to {common_end.date()}", flush=True)
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)
    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots_lagged_trade"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Loading stock-month panel for lagged triple-sort reconstruction...", flush=True)
    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)
    rf_df = _load_rf()

    specs = [
        ("TS32", (2, 4, 4), TS32_DIR),
        ("TS64", (4, 4, 4), TS64_DIR),
    ]

    all_backtests: list[pd.DataFrame] = []

    for label, n_bins, ts_dir in specs:
        # We do not need to read the old excess_ports.csv for returns, but checking
        # its presence helps catch missing TS32/TS64 generation outputs.
        old_csv = ts_dir / subdir / "excess_ports.csv"
        if not old_csv.exists():
            print(f"WARNING: {old_csv} not found. Still reconstructing from panel for {label}.", flush=True)

        lagged_csv = out_dir / f"triple_sort_{label.lower()}_candidate_matrix_lagged_trade.csv"
        print(f"\nBuilding lagged candidate returns for {label} ({n_bins})...", flush=True)
        returns, panel_lagged = build_lagged_triplesort_returns_and_panel(
            panel=panel,
            n_bins=n_bins,
            rf_df=rf_df,
            out_csv=lagged_csv,
        )

        bt_spec, _ = _run_one_spec(
            label=label,
            n_bins=n_bins,
            returns=returns,
            panel_lagged=panel_lagged,
            out_dir=out_dir,
        )
        all_backtests.append(bt_spec)

    # Align TS32 and TS64 outputs to one common window.
    all_backtests = _trim_to_common_window(all_backtests)
    pieces = all_backtests.copy()

    # Add benchmark once, aligned to common TS window.
    try:
        common_start = max(df["date_dt"].min() for df in pieces)
        common_end = min(df["date_dt"].max() for df in pieces)
        print("\nLoading S&P 500 benchmark from Yahoo Finance using SPY...", flush=True)
        bt_sp = load_yahoo_monthly_benchmark(
            ticker="SPY",
            start_date=common_start,
            end_date=common_end,
            method_name="S&P 500 (SPY adjusted close)",
        )
        ref_dates = set(pieces[0]["date_dt"].unique())
        bt_sp = bt_sp[bt_sp["date_dt"].isin(ref_dates)].copy()
        pieces.append(bt_sp)
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
    print(f"Plots saved to: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
