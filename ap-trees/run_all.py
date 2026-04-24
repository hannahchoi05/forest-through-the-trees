"""
Pure AP-Trees backtest for the Forest Through the Trees project.

Runs the 4 variants specified in the methodology PDF:
  A1: Static optimizer, no transaction costs
  A2: Static optimizer, stock-level transaction costs
  B : Rolling TC-aware optimizer, portfolio-level turnover (for cost accounting)
  C : Rolling TC-aware optimizer, stock-level turnover (for cost accounting)

Output schema matches ResidualMomentum/outputs/*/backtest_comparison.csv:
  date, date_dt, yy, mm, method, gross_ret, turnover_raw, turnover, cost, net_ret
"""
from __future__ import annotations

import pandas as pd

from config import (
    CHUNK_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
    DEFAULT_CHARS,
    DEFAULT_Y_MIN,
    DEFAULT_Y_MAX,
    DEFAULT_TREE_DEPTH,
    DEFAULT_Q_NUM,
    DEFAULT_TAU,
    N_TRAIN_VALID,
    CV_N,
    STATIC_LAMBDA_L1,
    STATIC_LAMBDA_L2,
    STATIC_MU0,
    LONG_ONLY,
    ROLLING_WINDOW,
    TC_COST,
    TC_LAMBDA_L1,
    TC_LAMBDA_L2,
    TC_LAMBDA_TC,
    TC_MU0,
    DEDUPLICATE_CANDIDATES,
    RUN_FULL_TREE_SET,
)
from data_io import load_yearly_chunks
from tree_portfolios import (
    build_all_ap_tree_candidate_returns,
    select_candidate_matrix,
)
from optimizer import static_paper_style_optimize, rolling_tc_optimize
from metrics import performance_metrics, add_wealth_drawdown
from plots import make_all_plots


METHOD_A1 = "AP-Trees baseline (static, no TC)"
METHOD_A2 = "AP-Trees static + stock-level TC"
METHOD_B  = "AP-Trees rolling TC-aware + portfolio-level TC"
METHOD_C  = "AP-Trees rolling TC-aware + stock-level TC"


def main() -> None:
    chars = DEFAULT_CHARS
    subdir = "_".join(chars)

    out_dir = OUTPUT_DIR / subdir
    plot_dir = PLOT_DIR / subdir
    stock_weights_dir = out_dir / "stock_weights_by_month"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    stock_weights_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load CRSP/Compustat panel
    # ------------------------------------------------------------------
    print(f"Loading stock-month yearly chunks for {subdir}...", flush=True)
    panel = load_yearly_chunks(CHUNK_DIR, chars, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    # Pure AP-Trees has no tilt signal. Add a dummy column so the shared tree
    # builder is happy; with tau=0 this column has no effect on the output.
    panel["residual_mom"] = 0.0

    # ------------------------------------------------------------------
    # 2. Build AP-Tree candidate portfolios
    # ------------------------------------------------------------------
    candidate_path = out_dir / "candidate_returns.csv"
    node_stats_path = out_dir / "node_stats.csv"

    reuse_tree_build = (
        candidate_path.exists()
        and node_stats_path.exists()
        and any(stock_weights_dir.glob("*.pkl"))
    )

    if reuse_tree_build:
        print("Reusing existing tree-build checkpoint from disk...", flush=True)
        candidate_returns = pd.read_csv(candidate_path)
        if "date_dt" in candidate_returns.columns:
            candidate_returns["date_dt"] = pd.to_datetime(candidate_returns["date_dt"])
        node_stats = pd.read_csv(node_stats_path)
        if "date_dt" in node_stats.columns:
            node_stats["date_dt"] = pd.to_datetime(node_stats["date_dt"])
        stock_weights = stock_weights_dir
    else:
        tree_mode = "full AP-tree set (paper-faithful)" if RUN_FULL_TREE_SET else "single-tree smoke test"
        print(f"Building {tree_mode}, tau={DEFAULT_TAU} (pure value-weighted)...", flush=True)

        candidate_returns, stock_weights, node_stats = build_all_ap_tree_candidate_returns(
            panel=panel,
            chars=chars,
            tree_depth=DEFAULT_TREE_DEPTH,
            tau=DEFAULT_TAU,        # 0.0 => baseline_ret == tilt_ret
            q_num=DEFAULT_Q_NUM,
            signal_col="residual_mom",
            deduplicate=DEDUPLICATE_CANDIDATES,
            run_full_tree_set=RUN_FULL_TREE_SET,
            stock_weights_dir=stock_weights_dir,
        )
        stock_weights = stock_weights_dir

    print(f"Candidate returns shape: {candidate_returns.shape}", flush=True)
    if isinstance(stock_weights, pd.DataFrame):
        print(f"Stock weights shape:     {stock_weights.shape}", flush=True)
    else:
        print(f"Stock weights streamed:  {stock_weights_dir}", flush=True)

    candidate_returns.to_csv(out_dir / "candidate_returns.csv", index=False)
    node_stats.to_csv(out_dir / "node_stats.csv", index=False)

    if isinstance(stock_weights, pd.DataFrame):
        stock_weights.to_csv(out_dir / "stock_weights.csv", index=False)

    # Read the baseline_ columns (identical to tilt_ at tau=0) and rename them
    # to the port_ prefix the optimizer expects.
    ap_tree_candidates = select_candidate_matrix(candidate_returns, prefix="baseline_")
    ap_tree_candidates.to_csv(out_dir / "ap_tree_candidate_matrix.csv", index=False)

    # ------------------------------------------------------------------
    # Variant A1: Static optimizer, no transaction cost
    # ------------------------------------------------------------------
    print("\n[A1] Static optimizer, no transaction cost...", flush=True)
    bt_a1, w_a1, diag_a1 = static_paper_style_optimize(
        ap_tree_candidates,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda_l1=STATIC_LAMBDA_L1,
        lambda_l2=STATIC_LAMBDA_L2,
        mu0=STATIC_MU0,
        long_only=LONG_ONLY,
        method_name=METHOD_A1,
        cost_per_turnover=0.0,
        stock_weights=None,
        use_stock_level_turnover=False,
    )
    bt_a1.to_csv(out_dir / "backtest_A1_static_no_tc.csv", index=False)
    w_a1.to_csv(out_dir / "weights_A1_static_no_tc.csv", index=False)
    diag_a1.to_csv(out_dir / "diagnostics_A1_static_no_tc.csv", index=False)

    # ------------------------------------------------------------------
    # Variant A2: Static optimizer, stock-level transaction cost
    # ------------------------------------------------------------------
    print("\n[A2] Static optimizer, stock-level transaction cost...", flush=True)
    bt_a2, w_a2, diag_a2 = static_paper_style_optimize(
        ap_tree_candidates,
        n_train_valid=N_TRAIN_VALID,
        cv_n=CV_N,
        lambda_l1=STATIC_LAMBDA_L1,
        lambda_l2=STATIC_LAMBDA_L2,
        mu0=STATIC_MU0,
        long_only=LONG_ONLY,
        method_name=METHOD_A2,
        cost_per_turnover=TC_COST,
        stock_weights=stock_weights_dir,
        use_stock_level_turnover=True,
    )
    bt_a2.to_csv(out_dir / "backtest_A2_static_stock_level_tc.csv", index=False)
    w_a2.to_csv(out_dir / "weights_A2_static_stock_level_tc.csv", index=False)
    diag_a2.to_csv(out_dir / "diagnostics_A2_static_stock_level_tc.csv", index=False)

    # ------------------------------------------------------------------
    # Variant B: Rolling TC-aware optimizer, portfolio-level turnover
    # ------------------------------------------------------------------
    print("\n[B] Rolling TC-aware optimizer, portfolio-level turnover...", flush=True)
    bt_b, w_b = rolling_tc_optimize(
        ap_tree_candidates,
        window=ROLLING_WINDOW,
        lambda_l1=TC_LAMBDA_L1,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        cost_per_turnover=TC_COST,
        mu0=TC_MU0,
        long_only=LONG_ONLY,
        method_name=METHOD_B,
        stock_weights=None,
        use_stock_level_turnover=False,
    )
    bt_b.to_csv(out_dir / "backtest_B_rolling_tc_portfolio_level_tc.csv", index=False)
    w_b.to_csv(out_dir / "weights_B_rolling_tc_portfolio_level_tc.csv", index=False)

    # ------------------------------------------------------------------
    # Variant C: Rolling TC-aware optimizer, stock-level turnover
    # ------------------------------------------------------------------
    print("\n[C] Rolling TC-aware optimizer, stock-level turnover...", flush=True)
    bt_c, w_c = rolling_tc_optimize(
        ap_tree_candidates,
        window=ROLLING_WINDOW,
        lambda_l1=TC_LAMBDA_L1,
        lambda_l2=TC_LAMBDA_L2,
        lambda_tc=TC_LAMBDA_TC,
        cost_per_turnover=TC_COST,
        mu0=TC_MU0,
        long_only=LONG_ONLY,
        method_name=METHOD_C,
        stock_weights=stock_weights_dir,
        use_stock_level_turnover=True,
    )
    bt_c.to_csv(out_dir / "backtest_C_rolling_tc_stock_level_tc.csv", index=False)
    w_c.to_csv(out_dir / "weights_C_rolling_tc_stock_level_tc.csv", index=False)

    # ------------------------------------------------------------------
    # Align all variants to a common test window
    # ------------------------------------------------------------------
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
    print(f"\nCommon test window: {common_start.date()} to {common_end.date()}", flush=True)

    def _clip(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["date_dt"] >= common_start) & (df["date_dt"] <= common_end)].copy()

    bt_a1 = _clip(bt_a1)
    bt_a2 = _clip(bt_a2)
    bt_b = _clip(bt_b)
    bt_c = _clip(bt_c)

    # ------------------------------------------------------------------
    # Combine into the single CSV the teammate expects
    # ------------------------------------------------------------------
    backtest = pd.concat([bt_a1, bt_a2, bt_b, bt_c], ignore_index=True)
    backtest = backtest.sort_values(["method", "yy", "mm"]).reset_index(drop=True)

    # Column order matches ResidualMomentum/outputs/*/backtest_comparison.csv
    schema = ["date", "date_dt", "yy", "mm", "method",
              "gross_ret", "turnover_raw", "turnover", "cost", "net_ret"]
    backtest = backtest[[c for c in schema if c in backtest.columns]]

    backtest.to_csv(out_dir / "backtest_comparison.csv", index=False)
    print(f"\nbacktest_comparison.csv -> {out_dir / 'backtest_comparison.csv'}", flush=True)

    enriched = add_wealth_drawdown(backtest)
    enriched.to_csv(out_dir / "backtest_comparison_with_wealth_drawdown.csv", index=False)

    metrics = performance_metrics(backtest)
    metrics.to_csv(out_dir / "summary_metrics_comparison.csv", index=False)

    print("\nSummary metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)

    print("\nMaking plots...", flush=True)
    make_all_plots(backtest, plot_dir)

    print(f"\nDone. Outputs saved to: {out_dir}", flush=True)
    print(f"Plots saved to:          {plot_dir}", flush=True)


if __name__ == "__main__":
    main()