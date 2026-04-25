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

    Algorithm (Appendix A.4):
      1. mu_shrunk = mu_hat + lambda0 * mean(mu_hat) * 1
      2. Eigen-decompose Sigma_hat -> Sigma^{1/2} and Sigma^{-1/2}
      3. mu_tilde = Sigma^{-1/2} @ mu_shrunk
      4. Augmented system X = [Sigma^{1/2}; sqrt(lambda2)*I], y = [mu_tilde; 0]
      5. LARS lasso path on (X, y) -> coef_path[N, n_steps]
      6. For each K: take the step with >= K non-zeros, clip to long-only, normalize.
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

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(sigma_hat)
    eigvals = np.maximum(eigvals, 1e-12)
    sqrt_ev = np.sqrt(eigvals)

    sigma_half     = eigvecs @ np.diag(sqrt_ev) @ eigvecs.T
    sigma_inv_half = eigvecs @ np.diag(1.0 / sqrt_ev) @ eigvecs.T

    mu_tilde = sigma_inv_half @ mu_shrunk

    # Augmented system
    X_aug = np.vstack([sigma_half, np.sqrt(lambda2) * np.eye(N)])
    y_aug = np.concatenate([mu_tilde, np.zeros(N)])

    # LARS lasso path
    try:
        _, _, coef_path = lars_path(X_aug, y_aug, method="lasso", max_iter=N)
    except Exception:
        return {}

    # coef_path: (N, n_steps), columns = solutions from sparse to dense
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

        # Long-only + normalize
        w = np.maximum(w, 0.0)
        total = w.sum()
        if total < 1e-12:
            continue
        results[K] = w / total

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
    stock_weights=None,          # None or Path to streaming directory
    use_stock_level_turnover: bool = True,
    rf: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SelectedParams]:
    """
    Paper-faithful AP-pruning with cross-validation over (lambda0, lambda2, K).

    Split: first n_train months = training, next n_valid = validation,
           remainder = test (out-of-sample).
    where n_train = n_train_valid * (cv_n-1)/cv_n, n_valid = n_train_valid/cv_n.

    Returns
    -------
    backtest_df   : monthly returns (test period), columns include gross_ret/net_ret
    weights_df    : selected candidates and their weights
    diagnostics   : train/valid/test Sharpe summary
    sel           : SelectedParams namedtuple with best lambda0, lambda2, k
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
    x_scaled = x_raw * scales[np.newaxis, :]

    # ── Subtract risk-free rate for estimation ────────────────────────────
    rf_arr = _align_rf(rf, len(df))
    x_excess = x_scaled - rf_arr[:, np.newaxis]

    # ── Train / valid / test split ────────────────────────────────────────
    n_valid = n_train_valid // cv_n
    n_train = n_train_valid - n_valid

    x_tr  = x_excess[:n_train]
    x_val = x_excess[n_train:n_train_valid]
    x_te  = x_excess[n_train_valid:]   # used for gross_ret (already excess)

    N = len(cols)

    # ── Grid search ──────────────────────────────────────────────────────
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

    # ── Re-estimate on full train+valid with best params ──────────────────
    k_weights_full = _ap_prune(
        x_excess[:n_train_valid],
        best_sel.lambda0, best_sel.lambda2,
        kmin=best_sel.k, kmax=best_sel.k,
    )
    if best_sel.k in k_weights_full:
        final_w = k_weights_full[best_sel.k]
    else:
        # fallback: use validation best
        final_w = best_w

    # ── Test-period gross returns ─────────────────────────────────────────
    gross = x_te @ final_w

    # ── Transaction costs ─────────────────────────────────────────────────
    result = df.iloc[n_train_valid:][meta_c].copy().reset_index(drop=True)
    result["method"] = method_name
    result["gross_ret"] = gross

    candidate_w = pd.Series(final_w, index=cols)
    sw_dir = Path(stock_weights) if isinstance(stock_weights, (str, Path)) and stock_weights is not None else None

    turnovers, costs = [], []
    prev_sw = None

    for _, row in result.iterrows():
        meta = {c: row[c] for c in meta_c}
        if use_stock_level_turnover and sw_dir is not None:
            curr_sw = _stock_weights_for_month(sw_dir, meta, candidate_w)
            to = _stock_turnover(curr_sw, prev_sw)
            prev_sw = curr_sw
        else:
            to = 0.05   # static proxy (same as teammate)
        turnovers.append(to)
        costs.append(to * cost_per_turnover)

    result["turnover_raw"] = turnovers
    result["turnover"]     = turnovers
    result["cost"]         = costs
    result["net_ret"]      = result["gross_ret"] - result["cost"]

    # ── Weights dataframe (selected portfolios only) ──────────────────────
    active_mask = np.abs(final_w) > 1e-8
    weights_df = pd.DataFrame({
        "candidate": [cols[i] for i in range(N) if active_mask[i]],
        "weight":    final_w[active_mask],
    }).sort_values("weight", ascending=False).reset_index(drop=True)

    # ── Diagnostics ───────────────────────────────────────────────────────
    tr_ret  = x_tr  @ final_w
    val_ret = x_val @ final_w
    te_ret  = gross

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
    eta: float = 0.15,               # mean shrinkage (lambda0 analog) in each window
    cost_per_turnover: float = 0.0025,
    method_name: str = "AP-tree rolling TC-aware",
    turnover_mode: str = "portfolio", # "portfolio" or "stock"
    stock_weights=None,              # None or Path
    selected_candidates: list[str] | None = None,
    long_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling-window TC-aware optimizer.

    Each month t:
      1. Estimate mu, sigma on x[t-window:t] (with mean shrinkage eta)
      2. Solve: min_w  1/2 w^T sigma w + 1/2 lambda_l2 ||w||^2 + lambda_tc ||w - w_{t-1}||_1
         s.t. sum(w)=1, w>=0 (if long_only)
      3. Turnover and TC computed at portfolio level ("portfolio") or
         by aggregating to stock level ("stock").

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

    # Depth-scale
    x_raw = df[cols].astype(float).fillna(0.0).to_numpy()
    scales = np.array([_depth_scale(c) for c in cols])
    x_scaled = x_raw * scales[np.newaxis, :]

    K = len(cols)
    w_prev = np.ones(K) / K
    prev_sw = None

    sw_dir = Path(stock_weights) if isinstance(stock_weights, (str, Path)) and stock_weights is not None else None

    rows, w_rows = [], []

    for t in range(window, len(df)):
        hist = x_scaled[t - window:t]

        mu_hat = hist.mean(axis=0)
        mu_bar = float(mu_hat.mean())
        mu_shrunk = mu_hat + eta * mu_bar * np.ones(K)

        sigma = np.cov(hist, rowvar=False, ddof=1)
        sigma = 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(K)

        # ── Solve QP ────────────────────────────────────────────────────
        def obj(w):
            return (0.5 * w @ sigma @ w
                    + 0.5 * lambda_l2 * np.dot(w, w)
                    + lambda_tc * np.sum(np.abs(w - w_prev)))

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * K if long_only else [(-2.0, 2.0)] * K
        x0 = np.clip(w_prev, 0.0, 1.0) if long_only else w_prev.copy()
        x0 /= x0.sum() if x0.sum() > 1e-12 else 1.0

        res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-10})
        w = res.x if res.success else x0
        w = np.clip(w, 0.0, None) if long_only else w
        w = w / w.sum() if w.sum() > 1e-12 else x0

        gross = float(x_scaled[t] @ w)
        raw_to = float(np.sum(np.abs(w - w_prev)))

        meta = df.iloc[t][meta_c].to_dict()
        cand_w = pd.Series(w, index=cols)

        if turnover_mode == "stock" and sw_dir is not None:
            curr_sw = _stock_weights_for_month(sw_dir, meta, cand_w)
            to = _stock_turnover(curr_sw, prev_sw)
            prev_sw = curr_sw
        else:
            to = raw_to

        cost = cost_per_turnover * to
        rows.append({**meta,
                     "method": method_name,
                     "gross_ret": gross,
                     "turnover_raw": raw_to,
                     "turnover": to,
                     "cost": cost,
                     "net_ret": gross - cost})

        for c, wi in zip(cols, w):
            if abs(wi) > 1e-12:
                w_rows.append({**meta, "candidate": c, "weight": wi})

        w_prev = w

    return pd.DataFrame(rows), pd.DataFrame(w_rows)
