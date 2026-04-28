"""
AP-Trees optimizer.

Implements:
1. Static AP-pruning / SDF recovery from Bryzgalova, Pelger, and Zhu.
2. Conversion from SDF weights to gross-exposure-normalized trading weights.
3. Rolling transaction-cost-aware trading-weight optimizer.

Important convention:
- AP-pruning estimates SDF weights on depth-scaled excess returns.
- For backtesting, SDF weights are converted to candidate-portfolio trading weights:
      w_trade = (b / adj_w) / sum_q |b_q / adj_w,q|
- Rolling TC-aware optimization directly optimizes trading weights on candidate portfolios.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path

from utils import annualized_sharpe


SelectedParams = namedtuple("SelectedParams", ["lambda0", "lambda2", "k", "val_sharpe"])


# =============================================================================
# Helpers
# =============================================================================

def _candidate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("port_")]


def _meta_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["date", "date_dt", "yy", "mm"] if c in df.columns]


def _depth_from_col(col: str) -> int:
    try:
        node_part = col.split("_N")[-1]
        return len(node_part) - 1
    except Exception:
        return 0


def _depth_scale(col: str) -> float:
    d = _depth_from_col(col)
    return float(np.sqrt(1.0 / (2 ** d)))


def _load_month_stock_weights(stock_weights_dir: Path, yy: int, mm: int) -> pd.DataFrame:
    path = stock_weights_dir / f"{int(yy):04d}_{int(mm):02d}.pkl"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_pickle(path, compression="gzip")


def _align_rf(rf: np.ndarray | None, n: int) -> np.ndarray:
    if rf is None:
        return np.zeros(n)
    rf = np.asarray(rf, dtype=float)
    if len(rf) >= n:
        return rf[:n]
    return np.concatenate([rf, np.zeros(n - len(rf))])


def _l1_normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    gross = float(np.abs(w).sum())
    if gross > 1e-12:
        return w / gross
    return w


def _budget_normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    s = float(w.sum())
    if abs(s) > 1e-12:
        return w / s
    return np.ones_like(w) / len(w)


def _stock_weights_for_month(
    stock_weights_source,
    meta: dict,
    candidate_weights: pd.Series,
    stock_weight_col: str = "tilt_stock_w",
) -> pd.Series:
    """
    Aggregate implied stock weights:
        W_i,t = sum_p w_p,t * s_i,p,t

    Important:
    Do NOT normalize by net exposure. That was the bug.
    Do NOT renormalize here either; turnover should follow the slide:
        TC(w_t, w_{t-1}) = ||M_t w_t - M_{t-1} w_{t-1}||_1
    """
    if stock_weights_source is None:
        return pd.Series(dtype=float)

    sw = _load_month_stock_weights(stock_weights_source, int(meta["yy"]), int(meta["mm"]))
    if sw.empty:
        return pd.Series(dtype=float)

    cw = candidate_weights.copy()
    cw.index = [c[len("port_"):] if c.startswith("port_") else c for c in cw.index]
    cw = cw[cw.abs() > 1e-12]

    if cw.empty:
        return pd.Series(dtype=float)

    wmap = cw.rename("optimizer_w").reset_index()
    wmap.columns = ["node_id", "optimizer_w"]
    wmap["node_id"] = wmap["node_id"].astype(str)

    sw = sw.copy()
    sw["node_id"] = sw["node_id"].astype(str)

    merged = sw.merge(wmap, on="node_id", how="inner")
    if merged.empty:
        return pd.Series(dtype=float)

    merged["final_w"] = (
        merged["optimizer_w"].astype(float)
        * merged[stock_weight_col].astype(float)
    )

    out = merged.groupby("permno")["final_w"].sum()
    out = out[out.abs() > 1e-14]
    return out


def _stock_basis_matrix(
    stock_weights_dir: Path,
    meta: dict,
    candidates: list[str],
    stock_weight_col: str = "tilt_stock_w",
) -> tuple[np.ndarray | None, list | None]:
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
    return float(
        (
            curr.reindex(idx, fill_value=0.0)
            - prev.reindex(idx, fill_value=0.0)
        ).abs().sum()
    )


# =============================================================================
# AP-pruning
# =============================================================================

def _ap_prune(
    x_est: np.ndarray,
    lambda0: float,
    lambda2: float,
    kmin: int,
    kmax: int,
) -> dict[int, np.ndarray]:
    """
    AP-pruning on depth-scaled excess returns.

    Returns SDF weights b in scaled-return space, normalized by abs(sum(b)).
    Negative weights are allowed.
    """
    _, n_assets = x_est.shape

    mu_hat = x_est.mean(axis=0)
    sigma_hat = np.cov(x_est, rowvar=False, ddof=1)

    sigma_hat = 0.5 * (sigma_hat + sigma_hat.T)
    sigma_hat += 1e-8 * np.eye(n_assets)

    mu_bar = float(mu_hat.mean())
    mu_shrunk = mu_hat + lambda0 * mu_bar * np.ones(n_assets)

    eigvals, eigvecs = np.linalg.eigh(sigma_hat)
    order = np.argsort(eigvals)[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals, 1e-12)

    sigma_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    keep = eigvals > 1e-10
    v_keep = eigvecs[:, keep]
    inv_sqrt = 1.0 / np.sqrt(eigvals[keep])
    mu_tilde = v_keep @ (inv_sqrt * (v_keep.T @ mu_shrunk))

    x_aug = np.vstack([sigma_half, np.sqrt(lambda2) * np.eye(n_assets)])
    y_aug = np.concatenate([mu_tilde, np.zeros(n_assets)])

    try:
        _, _, coef_path = lars_path(x_aug, y_aug, method="lasso", max_iter=n_assets)
    except Exception:
        return {}

    results: dict[int, np.ndarray] = {}

    for k in range(kmin, min(kmax, n_assets) + 1):
        w = None

        for step in range(coef_path.shape[1]):
            nz = int(np.sum(np.abs(coef_path[:, step]) > 1e-8))
            if nz >= k:
                w = coef_path[:, step].copy()
                break

        if w is None:
            w = coef_path[:, -1].copy()

        denom = abs(float(w.sum()))
        if denom < 1e-12:
            continue

        results[k] = w / denom

    return results


# =============================================================================
# Static AP-pruning optimizer
# =============================================================================

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
    Static AP-pruning.

    Estimation:
        uses depth-scaled excess returns.

    Backtest:
        converts SDF weights to gross-normalized trading weights and applies
        them to raw candidate returns.
    """
    if lambda0_grid is None:
        lambda0_grid = [0.0, 0.15, 0.30, 0.45, 0.60, 0.90]

    if lambda2_grid is None:
        lambda2_grid = [1e-5, 1e-6, 1e-7, 1e-8]

    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)

    cols = _candidate_cols(df)
    meta_c = _meta_cols(df)

    if not cols:
        raise ValueError("No candidate columns found. Expected columns starting with 'port_'.")

    if n_train_valid >= len(df):
        raise ValueError("n_train_valid must be < number of months in returns_df.")

    x_raw = df[cols].astype(float).fillna(0.0).to_numpy()

    rf_arr = _align_rf(rf, len(df))
    x_excess = x_raw - rf_arr[:, None]

    scales = np.array([_depth_scale(c) for c in cols])
    x_scaled_excess = x_excess * scales[None, :]

    n_valid = n_train_valid // cv_n
    n_train = n_train_valid - n_valid

    x_tr = x_scaled_excess[:n_train]
    x_val = x_scaled_excess[n_train:n_train_valid]

    n_assets = len(cols)

    best_sr = -np.inf
    best_sel = SelectedParams(lambda0=0.0, lambda2=1e-6, k=kmin, val_sharpe=-np.inf)
    best_w = np.ones(n_assets) / n_assets

    print(
        f"  AP-pruning grid search: {len(lambda0_grid)} x {len(lambda2_grid)} "
        f"param combos, K in [{kmin},{kmax}]...",
        flush=True,
    )

    for l0 in lambda0_grid:
        for l2 in lambda2_grid:
            k_weights = _ap_prune(x_tr, l0, l2, kmin, kmax)

            for k, w in k_weights.items():
                val_ret = x_val @ w
                sr = annualized_sharpe(pd.Series(val_ret))

                if np.isfinite(sr) and sr > best_sr:
                    best_sr = sr
                    best_sel = SelectedParams(lambda0=l0, lambda2=l2, k=k, val_sharpe=sr)
                    best_w = w.copy()

    print(
        f"  Best params: lambda0={best_sel.lambda0}, "
        f"lambda2={best_sel.lambda2:.2e}, K={best_sel.k}, "
        f"val_SR={best_sel.val_sharpe:.3f}",
        flush=True,
    )

    k_weights_full = _ap_prune(
        x_scaled_excess[:n_train_valid],
        best_sel.lambda0,
        best_sel.lambda2,
        kmin=best_sel.k,
        kmax=best_sel.k,
    )

    sdf_w = k_weights_full.get(best_sel.k, best_w)

    # Convert SDF weights to candidate-portfolio trading weights.
    b_unscaled = sdf_w / scales
    w_trade = _l1_normalize(b_unscaled)

    x_raw_te = x_raw[n_train_valid:]
    gross = x_raw_te @ w_trade

    result = df.iloc[n_train_valid:][meta_c].copy().reset_index(drop=True)
    result["method"] = method_name
    result["gross_ret"] = gross

    candidate_w = pd.Series(w_trade, index=cols)
    sw_dir = (
        Path(stock_weights)
        if isinstance(stock_weights, (str, Path)) and stock_weights is not None
        else None
    )

    turnover_stock = []
    costs = []
    prev_sw = None

    for _, row in result.iterrows():
        meta = {c: row[c] for c in meta_c}

        if use_stock_level_turnover and sw_dir is not None:
            curr_sw = _stock_weights_for_month(sw_dir, meta, candidate_w)
            to = _stock_turnover(curr_sw, prev_sw)
            prev_sw = curr_sw
        else:
            to = 0.0

        turnover_stock.append(to)
        costs.append(cost_per_turnover * to)

    turnover_portfolio = np.zeros(len(result), dtype=float)
    if len(turnover_portfolio) > 0:
        # First month establishes the position.
        turnover_portfolio[0] = float(np.abs(w_trade).sum())

    result["turnover_raw"] = turnover_portfolio
    result["turnover"] = turnover_stock
    result["turnover_portfolio"] = turnover_portfolio
    result["turnover_stock"] = turnover_stock
    result["cost"] = costs
    result["net_ret"] = result["gross_ret"] - result["cost"]

    active_mask = np.abs(w_trade) > 1e-8

    weights_df = pd.DataFrame({
        "candidate": [cols[i] for i in range(n_assets) if active_mask[i]],
        "weight": w_trade[active_mask],
    }).sort_values("weight", ascending=False).reset_index(drop=True)

    # Diagnostics should use the same trading-return convention as the backtest.
    tr_ret = x_raw[:n_train] @ w_trade
    val_ret = x_raw[n_train:n_train_valid] @ w_trade
    te_ret = x_raw[n_train_valid:] @ w_trade

    diag = pd.DataFrame({
        "sample": ["train", "valid", "test"],
        "n_months": [n_train, n_valid, len(te_ret)],
        "mean_monthly": [tr_ret.mean(), val_ret.mean(), te_ret.mean()],
        "std_monthly": [
            tr_ret.std(ddof=1),
            val_ret.std(ddof=1),
            te_ret.std(ddof=1),
        ],
        "sharpe_ann": [
            annualized_sharpe(pd.Series(tr_ret)),
            annualized_sharpe(pd.Series(val_ret)),
            annualized_sharpe(pd.Series(te_ret)),
        ],
    })

    return result.reset_index(drop=True), weights_df, diag, best_sel


# =============================================================================
# Rolling transaction-cost-aware optimizer
# =============================================================================

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
    long_only: bool = False,
    rf: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling TC-aware optimizer.

    This is the trading/backtest extension, not the original static SDF
    evaluation from the paper.

    If long_only=True:
        s.t. sum(w) = 1, w >= 0

    If long_only=False:
        s.t. sum(abs(w)) = 1

    The long-short case matches the gross-exposure-normalized trading-weight
    convention used in the slides.
    """
    from scipy.optimize import minimize

    if turnover_mode not in {"portfolio", "stock"}:
        raise ValueError("turnover_mode must be either 'portfolio' or 'stock'.")

    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)

    all_cols = _candidate_cols(df)
    meta_c = _meta_cols(df)

    if not all_cols:
        raise ValueError("No candidate columns found. Expected columns starting with 'port_'.")

    if selected_candidates is not None:
        cols = [c for c in all_cols if c in selected_candidates]
        if not cols:
            cols = all_cols
    else:
        cols = all_cols

    x_raw = df[cols].astype(float).fillna(0.0).to_numpy()

    rf_arr = _align_rf(rf, len(df))
    x_excess = x_raw - rf_arr[:, None]

    n_assets = len(cols)

    if long_only:
        w_prev = np.ones(n_assets) / n_assets
    else:
        w_prev = np.ones(n_assets) / n_assets
        w_prev = _l1_normalize(w_prev)

    prev_sw = None

    sw_dir = (
        Path(stock_weights)
        if isinstance(stock_weights, (str, Path)) and stock_weights is not None
        else None
    )

    rows = []
    w_rows = []

    for t in range(window, len(df)):
        hist = x_excess[t - window:t]

        mu_hat = hist.mean(axis=0)

        sigma = np.cov(hist, rowvar=False, ddof=1)
        sigma = 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(n_assets)

        meta = df.iloc[t][meta_c].to_dict()

        B = None
        permnos_curr = None
        prev_stock_aligned = None

        if sw_dir is not None:
            B, permnos_curr = _stock_basis_matrix(sw_dir, meta, cols)

            if B is not None:
                if prev_sw is not None and not prev_sw.empty:
                    prev_stock_aligned = prev_sw.reindex(
                        permnos_curr,
                        fill_value=0.0,
                    ).to_numpy()
                else:
                    prev_stock_aligned = np.zeros(len(permnos_curr))

        use_stock_penalty = (
            turnover_mode == "stock"
            and B is not None
            and prev_stock_aligned is not None
        )

        if use_stock_penalty:
            def obj(w):
                stock_w = B.T @ w
                return (
                    0.5 * w @ sigma @ w
                    - eta * np.dot(mu_hat, w)
                    + 0.5 * lambda_l2 * np.dot(w, w)
                    + lambda_tc * np.sum(np.abs(stock_w - prev_stock_aligned))
                )
        else:
            def obj(w):
                return (
                    0.5 * w @ sigma @ w
                    - eta * np.dot(mu_hat, w)
                    + 0.5 * lambda_l2 * np.dot(w, w)
                    + lambda_tc * np.sum(np.abs(w - w_prev))
                )

        if long_only:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            ]
            bounds = [(0.0, 1.0)] * n_assets
            x0 = np.clip(w_prev, 0.0, 1.0)
            x0 = _budget_normalize(x0)
        else:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0},
            ]
            bounds = [(-1.0, 1.0)] * n_assets
            x0 = _l1_normalize(w_prev)

        res = minimize(
            obj,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )

        if res.success and np.all(np.isfinite(res.x)):
            w = res.x.copy()
        else:
            w = x0.copy()

        w[np.abs(w) < 1e-12] = 0.0

        if long_only:
            w = np.clip(w, 0.0, None)
            w = _budget_normalize(w)
        else:
            w = _l1_normalize(w)

        gross = float(x_raw[t] @ w)

        turnover_portfolio = float(np.sum(np.abs(w - w_prev)))

        turnover_stock_val = np.nan

        if B is not None and permnos_curr is not None:
            stock_w_curr = B.T @ w
            curr_sw = pd.Series(stock_w_curr, index=permnos_curr)
            curr_sw = curr_sw[curr_sw.abs() > 1e-14]

            turnover_stock_val = _stock_turnover(curr_sw, prev_sw)
            prev_sw = curr_sw

        if turnover_mode == "stock" and np.isfinite(turnover_stock_val):
            charged_turnover = turnover_stock_val
        else:
            charged_turnover = turnover_portfolio

        cost = cost_per_turnover * charged_turnover

        rows.append({
            **meta,
            "method": method_name,
            "gross_ret": gross,
            "turnover_raw": turnover_portfolio,
            "turnover": charged_turnover,
            "turnover_portfolio": turnover_portfolio,
            "turnover_stock": turnover_stock_val,
            "cost": cost,
            "net_ret": gross - cost,
        })

        for c, wi in zip(cols, w):
            if abs(wi) > 1e-12:
                w_rows.append({
                    **meta,
                    "candidate": c,
                    "weight": wi,
                })

        w_prev = w.copy()

    return pd.DataFrame(rows), pd.DataFrame(w_rows)