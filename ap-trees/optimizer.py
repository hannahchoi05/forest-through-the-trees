"""
AP-Trees optimizer — implements the AP-pruning algorithm from
Bryzgalova, Pelger & Zhu (2020) Section II and Appendix A.4.

Key functions
-------------
ap_pruning_static_optimize  : paper-faithful AP-pruning with LARS lasso path,
                               mean shrinkage (lambda0) and ridge (lambda2),
                               cross-validated over grids. Returns 4-tuple
                               (backtest_df, weights_df, diagnostics_df, SelectedParams).

rolling_tc_optimize         : rolling-window TC-aware optimizer for Variants B/C.
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path

from utils import annualized_sharpe


# ── Named tuple for selected hyper-params ─────────────────────────────────
SelectedParams = namedtuple("SelectedParams", ["lambda0", "lambda2", "k", "val_sharpe"])


# ══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════

def _candidate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("port_")]


def _meta_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["date", "date_dt", "yy", "mm"] if c in df.columns]


def _depth_from_col(col: str) -> int:
    """
    Extract node depth from a port_ column name.
    Format: port_T{seq}_N{node_path}
    Depth = len(node_path) - 1  (root "1" is depth 0).
    """
    try:
        node_part = col.split("_N")[-1]
        return len(node_part) - 1
    except Exception:
        return 0


def _depth_scale(col: str) -> float:
    """Paper Appendix A.4: multiply each portfolio by sqrt(1 / 2^depth)."""
    d = _depth_from_col(col)
    return float(np.sqrt(1.0 / (2 ** d)))


def _load_month_stock_weights(stock_weights_dir: Path, yy: int, mm: int) -> pd.DataFrame:
    """Load one month's stock-weight file from the streaming directory."""
    path = stock_weights_dir / f"{int(yy):04d}_{int(mm):02d}.pkl"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_pickle(path, compression="gzip")


def _stock_weights_for_month(
    stock_weights_source,          # Path or None
    meta: dict,
    candidate_weights: pd.Series,  # index = port_ col names, values = optimizer weights
    stock_weight_col: str = "tilt_stock_w",
) -> pd.Series:
    """
    Aggregate per-stock final weights:
        W_i,t = sum_p optimizer_w_p * stock_w_{i,p,t}
    Works whether stock_weights_source is a Path (directory) or None.
    """
    if stock_weights_source is None:
        return pd.Series(dtype=float)

    sw = _load_month_stock_weights(stock_weights_source, int(meta["yy"]), int(meta["mm"]))
    if sw.empty:
        return pd.Series(dtype=float)

    # Strip the port_ prefix to get node_ids
    cw = candidate_weights.copy()
    cw.index = [c[len("port_"):] if c.startswith("port_") else c for c in cw.index]
    cw = cw[cw.abs() > 1e-12]
    if cw.empty:
        return pd.Series(dtype=float)

    wmap = cw.rename("optimizer_w").reset_index()
    wmap.columns = ["node_id", "optimizer_w"]
    wmap["node_id"] = wmap["node_id"].astype(str)

    sw["node_id"] = sw["node_id"].astype(str)
    merged = sw.merge(wmap, on="node_id", how="inner")
    if merged.empty:
        return pd.Series(dtype=float)

    merged["final_w"] = merged["optimizer_w"].astype(float) * merged[stock_weight_col].astype(float)
    out = merged.groupby("permno")["final_w"].sum()
    out = out[out.abs() > 1e-14]
    total = out.sum()
    if abs(total) > 1e-12:
        out = out / total
    return out


def _stock_basis_matrix(
    stock_weights_dir: Path,
    meta: dict,
    candidates: list[str],
    stock_weight_col: str = "tilt_stock_w",
) -> tuple[np.ndarray | None, list | None]:
    """
    Build the (K, N_stocks) basis matrix B for one month, where
    B[k, i] = stock-i weight inside basis-asset k (the candidate's stock holdings).

    Returned alongside the permno list so the caller can construct
    stock-level weight vectors via stock_w = B.T @ w_candidate.
    Returns (None, None) if no stock-weights file is available for this month.
    """
    sw = _load_month_stock_weights(stock_weights_dir, int(meta["yy"]), int(meta["mm"]))
    if sw.empty:
        return None, None

    cand_ids = [c[len("port_"):] if c.startswith("port_") else c for c in candidates]
    sw = sw.copy()
    sw["node_id"] = sw["node_id"].astype(str)
    sw = sw[sw["node_id"].isin(cand_ids)]
    if sw.empty:
        return None, None

    pivot = sw.pivot_table(
        index="node_id",
        columns="permno",
        values=stock_weight_col,
        fill_value=0.0,
        aggfunc="sum",
    )
    pivot = pivot.reindex(cand_ids, fill_value=0.0)
    return pivot.to_numpy(), pivot.columns.tolist()


def _stock_turnover(curr: pd.Series, prev: pd.Series | None) -> float:
    if prev is None or prev.empty:
        return float(curr.abs().sum())
    idx = curr.index.union(prev.index)
    return float((curr.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)).abs().sum())


def _align_rf(rf: np.ndarray | None, n: int) -> np.ndarray:
    """Return an rf array of length n (pad with zeros if rf is too short)."""
    if rf is None:
        return np.zeros(n)
    if len(rf) >= n:
        return rf[:n]
    return np.concatenate([rf, np.zeros(n - len(rf))])


# ══════════════════════════════════════════════════════════════════════════
# Core AP-Pruning routine (paper Appendix A.4)
# ══════════════════════════════════════════════════════════════════════════

def _ap_prune(
    x_est: np.ndarray,    # (T, N) excess returns used for estimation
    lambda0: float,
    lambda2: float,
    kmin: int,
    kmax: int,
) -> dict[int, np.ndarray]:
    """
    Run the LARS-based AP-pruning for one (lambda0, lambda2) pair.

    Returns a dict  {K: w}  of long-only normalized weight vectors for each
    achievable K in [kmin, kmax].
    """
    T, N = x_est.shape
    mu_hat = x_est.mean(axis=0)
    sigma_hat = np.cov(x_est, rowvar=False, ddof=1)

    # Regularise sigma
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.T)
    sigma_hat += 1e-8 * np.eye(N)

    # Mean shrinkage
    mu_bar = float(mu_hat.mean())
    mu_shrunk = mu_hat + lambda0 * mu_bar * np.ones(N)

    # Eigendecomposition — sort descending to match R's eigen() convention
    eigvals, eigvecs = np.linalg.eigh(sigma_hat)
    order   = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals, 1e-12)
    sqrt_ev = np.sqrt(eigvals)

    sigma_half = eigvecs @ np.diag(sqrt_ev) @ eigvecs.T

    # mu_tilde uses only eigenvalues above threshold (avoids inverting near-zeros)
    keep        = eigvals > 1e-10
    V_k         = eigvecs[:, keep]
    inv_sqrt_k  = 1.0 / np.sqrt(eigvals[keep])
    mu_tilde    = V_k @ (inv_sqrt_k * (V_k.T @ mu_shrunk))

    # Augmented system
    X_aug = np.vstack([sigma_half, np.sqrt(lambda2) * np.eye(N)])
    y_aug = np.concatenate([mu_tilde, np.zeros(N)])

    # LARS lasso path
    try:
        _, _, coef_path = lars_path(X_aug, y_aug, method="lasso", max_iter=N)
    except Exception:
        return {}

    results: dict[int, np.ndarray] = {}

    for K in range(kmin, min(kmax, N) + 1):
        w = None
        for step in range(coef_path.shape[1]):
            nz = int(np.sum(np.abs(coef_path[:, step]) > 1e-8))
            if nz >= K:
                w = coef_path[:, step].copy()
                break
        if w is None:
            w = coef_path[:, -1].copy()

        # Normalize by abs(sum(w)) — matches R's b = b / abs(sum(b)).
        # Do NOT clip to zero: LARS SDF weights can be negative (short positions
        # in the portfolio combination). The underlying sorted portfolios are
        # long-only by construction; the SDF weights combining them are not.
        denom = abs(float(w.sum()))
        if denom < 1e-12:
            continue
        results[K] = w / denom

    return results


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════

def ap_pruning_static_optimize(
    returns_df: pd.DataFrame,
    n_train_valid: int,
    cv_n: int = 3,
    lambda0_grid: list[float] | None = None,
    lambda2_grid: list[float] | None = None,
    port_n: int = 40,
    kmin: int = 10,
    kmax: int = 40,
    method_name: str = "AP-tree AP-pruning (static, no TC)",
    cost_per_turnover: float = 0.0,
    stock_weights=None,
    use_stock_level_turnover: bool = True,
    rf: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SelectedParams]:
    """
    Paper-faithful AP-pruning with cross-validation over (lambda0, lambda2, K).
    """
    if lambda0_grid is None:
        lambda0_grid = [0.0, 0.15, 0.30, 0.45, 0.60, 0.90]
    if lambda2_grid is None:
        lambda2_grid = [1e-5, 1e-6, 1e-7, 1e-8]

    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)
    cols = _candidate_cols(df)
    meta_c = _meta_cols(df)

    if n_train_valid >= len(df):
        raise ValueError("n_train_valid must be < number of months in returns_df.")

    # ── Depth-scale portfolio returns (Appendix A.4) ──────────────────────
    x_raw = df[cols].astype(float).fillna(0.0).to_numpy()
    scales = np.array([_depth_scale(c) for c in cols])

    # Subtract rf FIRST, then depth-scale
    rf_arr = _align_rf(rf, len(df))
    x_excess_unscaled = x_raw - rf_arr[:, np.newaxis]
    x_scaled_excess   = x_excess_unscaled * scales[np.newaxis, :]

    # Train / valid / test split
    n_valid = n_train_valid // cv_n
    n_train = n_train_valid - n_valid

    x_tr  = x_scaled_excess[:n_train]
    x_val = x_scaled_excess[n_train:n_train_valid]
    x_te  = x_scaled_excess[n_train_valid:]

    N = len(cols)

    # Grid search
    best_sr    = -np.inf
    best_sel   = SelectedParams(lambda0=0.0, lambda2=1e-6, k=kmin, val_sharpe=-np.inf)
    best_w     = np.ones(N) / N

    print(f"  AP-pruning grid search: {len(lambda0_grid)} x {len(lambda2_grid)} "
          f"param combos, K in [{kmin},{kmax}]...", flush=True)

    for l0 in lambda0_grid:
        for l2 in lambda2_grid:
            k_weights = _ap_prune(x_tr, l0, l2, kmin, kmax)
            for K, w in k_weights.items():
                val_ret = x_val @ w
                sr = annualized_sharpe(pd.Series(val_ret))
                if np.isfinite(sr) and sr > best_sr:
                    best_sr  = sr
                    best_sel = SelectedParams(lambda0=l0, lambda2=l2, k=K, val_sharpe=sr)
                    best_w   = w.copy()

    print(f"  Best params: lambda0={best_sel.lambda0}, lambda2={best_sel.lambda2:.2e}, "
          f"K={best_sel.k}, val_SR={best_sel.val_sharpe:.3f}", flush=True)

    # Re-estimate on full train+valid with best params
    k_weights_full = _ap_prune(
        x_scaled_excess[:n_train_valid],
        best_sel.lambda0, best_sel.lambda2,
        kmin=best_sel.k, kmax=best_sel.k,
    )
    final_w = k_weights_full[best_sel.k] if best_sel.k in k_weights_full else best_w

    # Test-period gross returns — use RAW (unscaled, with rf) portfolio returns.
    # Depth-scaling and rf subtraction are only for estimation (mu/sigma/LARS).
    # Matching R: sdf_ret = ports %*% (b / adj_w), where b already has adj_w
    # baked in, so b / adj_w removes the scaling → raw port returns.
    # For the backtest wealth series we use raw returns so $1 grows correctly.
    x_raw_te = x_raw[n_train_valid:]
    gross = x_raw_te @ final_w

    # Transaction costs
    result = df.iloc[n_train_valid:][meta_c].copy().reset_index(drop=True)
    result["method"] = method_name
    result["gross_ret"] = gross

    candidate_w = pd.Series(final_w, index=cols)
    sw_dir = Path(stock_weights) if isinstance(stock_weights, (str, Path)) and stock_weights is not None else None

    turnover_stock, costs = [], []
    prev_sw = None

    for _, row in result.iterrows():
        meta = {c: row[c] for c in meta_c}
        if use_stock_level_turnover and sw_dir is not None:
            curr_sw = _stock_weights_for_month(sw_dir, meta, candidate_w)
            to = _stock_turnover(curr_sw, prev_sw)
            prev_sw = curr_sw
        else:
            to = 0.05
        turnover_stock.append(to)
        costs.append(to * cost_per_turnover)

    turnover_portfolio = np.zeros(len(result), dtype=float)
    if len(turnover_portfolio) > 0:
        turnover_portfolio[0] = 1.0

    result["turnover_raw"] = turnover_portfolio
    result["turnover"] = turnover_stock
    result["turnover_portfolio"] = turnover_portfolio
    result["turnover_stock"] = turnover_stock
    result["cost"]         = costs
    result["net_ret"]      = result["gross_ret"] - result["cost"]

    # Weights dataframe
    active_mask = np.abs(final_w) > 1e-8
    weights_df = pd.DataFrame({
        "candidate": [cols[i] for i in range(N) if active_mask[i]],
        "weight":    final_w[active_mask],
    }).sort_values("weight", ascending=False).reset_index(drop=True)

    # Diagnostics
    tr_ret  = x_tr  @ final_w
    val_ret = x_val @ final_w
    te_ret  = x_te  @ final_w

    diag = pd.DataFrame({
        "sample":            ["train", "valid", "test"],
        "n_months":          [n_train, n_valid, len(te_ret)],
        "mean_monthly":      [tr_ret.mean(), val_ret.mean(), te_ret.mean()],
        "std_monthly":       [tr_ret.std(ddof=1), val_ret.std(ddof=1), te_ret.std(ddof=1)],
        "sharpe_ann":        [annualized_sharpe(pd.Series(tr_ret)),
                              annualized_sharpe(pd.Series(val_ret)),
                              annualized_sharpe(pd.Series(te_ret))],
    })

    return result.reset_index(drop=True), weights_df, diag, best_sel


# ══════════════════════════════════════════════════════════════════════════
# Rolling TC-aware optimizer (Variants B and C)
# ══════════════════════════════════════════════════════════════════════════

def rolling_tc_optimize(
    returns_df: pd.DataFrame,
    window: int = 120,
    lambda_l2: float = 1e-3,
    lambda_tc: float = 0.0025,
    eta: float = 0.15,
    cost_per_turnover: float = 0.0025,
    method_name: str = "AP-tree rolling TC-aware",
    turnover_mode: str = "portfolio",
    stock_weights=None,
    selected_candidates: list[str] | None = None,
    long_only: bool = True,
    rf: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling-window TC-aware optimizer.

    Each month t:
      1. Estimate mu, sigma on x[t-window:t] (with mean shrinkage eta).
      2. Build the per-month basis matrix B (K x N_stocks) so we can express
         stock-level weights as B.T @ w.
      3. Solve:
         - portfolio mode:
             min_w  0.5*gamma * w^T sigma w - mu^T w
                  + 0.5 lambda_l2 ||w||^2
                  + lambda_tc ||w - w_{t-1}||_1
         - stock mode:
             min_w  0.5*gamma * w^T sigma w - mu^T w
                  + 0.5 lambda_l2 ||w||^2
                  + lambda_tc ||B^T w - prev_stock_w||_1
         s.t. sum(w)=1, w>=0 (if long_only)
      4. Both portfolio-level and stock-level turnover are *always* recorded
         in the output, regardless of which mode was used to charge cost.

    Returns (backtest_df, weights_df).
    """
    from scipy.optimize import minimize

    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)
    all_cols = _candidate_cols(df)
    meta_c   = _meta_cols(df)

    # Optionally restrict to a subset of candidates
    if selected_candidates is not None:
        cols = [c for c in all_cols if c in selected_candidates]
        if not cols:
            cols = all_cols
    else:
        cols = all_cols

    # Subtract rf first, then depth-scale (same convention as static optimizer).
    x_raw = df[cols].astype(float).fillna(0.0).to_numpy()
    rf_arr = _align_rf(rf, len(df))
    x_excess_unscaled = x_raw - rf_arr[:, np.newaxis]
    scales = np.array([_depth_scale(c) for c in cols])
    x_scaled = x_excess_unscaled * scales[np.newaxis, :]

    K = len(cols)
    w_prev = np.ones(K) / K
    prev_sw = None  # pd.Series of permno -> normalized stock weight

    sw_dir = Path(stock_weights) if isinstance(stock_weights, (str, Path)) and stock_weights is not None else None

    rows, w_rows = [], []

    for t in range(window, len(df)):
        hist = x_scaled[t - window:t]

        mu_hat = hist.mean(axis=0)
        mu_bar = float(mu_hat.mean())
        mu_shrunk = mu_hat + eta * mu_bar * np.ones(K)

        sigma = np.cov(hist, rowvar=False, ddof=1)
        sigma = 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(K)

        meta = df.iloc[t][meta_c].to_dict()

        # Build basis once per month (shared by penalty + measurement)
        B, permnos_curr = (None, None)
        prev_stock_aligned = None
        if sw_dir is not None:
            B, permnos_curr = _stock_basis_matrix(sw_dir, meta, cols)
            if B is not None:
                if prev_sw is not None and not prev_sw.empty:
                    prev_stock_aligned = prev_sw.reindex(permnos_curr, fill_value=0.0).to_numpy()
                else:
                    prev_stock_aligned = np.zeros(len(permnos_curr))

        use_stock_penalty = (turnover_mode == "stock"
                             and B is not None
                             and prev_stock_aligned is not None)

        # Solve QP — penalty matches turnover_mode, gamma scales risk aversion
        if use_stock_penalty:
            def obj(w):
                stock_w = B.T @ w
                return (0.5 * w @ sigma @ w
                        - np.dot(mu_shrunk, w)
                        + 0.5 * lambda_l2 * np.dot(w, w)
                        + lambda_tc * np.sum(np.abs(stock_w - prev_stock_aligned)))
        else:
            def obj(w):
                return (0.5 * w @ sigma @ w
                        - np.dot(mu_shrunk, w)
                        + 0.5 * lambda_l2 * np.dot(w, w)
                        + lambda_tc * np.sum(np.abs(w - w_prev)))

        TARGET_MONTHLY_VAR = (0.067 / np.sqrt(12)) ** 2  # A1's monthly variance
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "ineq", "fun": lambda w: TARGET_MONTHLY_VAR - w @ sigma @ w},
        ]
        bounds = [(0.0, 1.0)] * K if long_only else [(-2.0, 2.0)] * K
        x0 = np.clip(w_prev, 0.0, 1.0) if long_only else w_prev.copy()
        x0 = x0 / x0.sum() if x0.sum() > 1e-12 else np.ones(K) / K

        res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-10})
        w = res.x if res.success else x0
        w[np.abs(w) < 1e-12] = 0.0
        w = np.clip(w, 0.0, None) if long_only else w
        w = w / w.sum() if w.sum() > 1e-12 else x0

        # Use raw (unscaled, with rf included) returns for backtest gross return.
        # x_scaled is only for estimation; wealth plots must use raw port returns.
        gross = float(x_raw[t] @ w)
        raw_to = float(np.sum(np.abs(w - w_prev)))

        # Always compute stock-level turnover when basis is available
        turnover_stock_val = np.nan
        if B is not None:
            stock_w_curr = B.T @ w
            s = stock_w_curr.sum()
            if abs(s) > 1e-12:
                stock_w_curr = stock_w_curr / s
            curr_sw = pd.Series(stock_w_curr, index=permnos_curr)
            curr_sw = curr_sw[curr_sw.abs() > 1e-14]
            turnover_stock_val = _stock_turnover(curr_sw, prev_sw)
            prev_sw = curr_sw

        # Cost charged according to the requested mode
        if turnover_mode == "stock" and not np.isnan(turnover_stock_val):
            to = turnover_stock_val
        else:
            to = raw_to

        cost = cost_per_turnover * to
        rows.append({**meta,
                     "method": method_name,
                     "gross_ret": gross,
                     "turnover_raw": raw_to,
                     "turnover": to,
                     "turnover_portfolio": raw_to,
                     "turnover_stock": turnover_stock_val,
                     "cost": cost,
                     "net_ret": gross - cost})

        for c, wi in zip(cols, w):
            if abs(wi) > 1e-12:
                w_rows.append({**meta, "candidate": c, "weight": wi})

        w_prev = w

    return pd.DataFrame(rows), pd.DataFrame(w_rows)