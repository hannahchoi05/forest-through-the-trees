from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import ntile_r


def _candidate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("port_")]


def _meta_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["date", "date_dt", "yy", "mm"] if c in df.columns]


def _normalize_gross_exposure(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    gross = float(np.sum(np.abs(w)))
    if not np.isfinite(gross) or gross < 1e-12:
        return np.zeros_like(w)
    return w / gross


def solve_portfolio_qp(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray | None = None,
    lambda_l1: float = 0.0,
    lambda_l2: float = 1e-3,
    lambda_tc: float = 0.0,
    mu0: float | None = None,
    long_only: bool = True,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    k = len(mu)

    if w_prev is None:
        w_prev = np.ones(k) / k
    else:
        w_prev = np.asarray(w_prev, dtype=float)

    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(k)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    def obj(w: np.ndarray) -> float:
        return float(
            0.5 * w @ sigma @ w
            + lambda_l1 * np.sum(np.abs(w))
            + 0.5 * lambda_l2 * np.sum(w * w)
            + lambda_tc * np.sum(np.abs(w - w_prev))
        )

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if mu0 is not None:
        constraints.append({"type": "ineq", "fun": lambda w: float(w @ mu - mu0)})

    if long_only:
        bounds = [(0.0, 1.0) for _ in range(k)]
        x0 = np.clip(w_prev, 0.0, 1.0)
    else:
        bounds = [(-1.0, 1.0) for _ in range(k)]
        x0 = w_prev.copy()

    if abs(x0.sum()) < 1e-12:
        x0 = np.ones(k) / k
    else:
        x0 = x0 / x0.sum()

    res = minimize(
        obj,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )

    if not res.success:
        # Freeze previous portfolio instead of resetting to equal weights.
        return _normalize_gross_exposure(w_prev)

    w = np.asarray(res.x, dtype=float)
    if np.any(~np.isfinite(w)):
        return _normalize_gross_exposure(w_prev)

    w[np.abs(w) < 1e-12] = 0.0

    # Keep static optimizer in "budget" convention (sum(w)=1) when feasible.
    if abs(w.sum()) < 1e-12:
        return _normalize_gross_exposure(w_prev)
    return w / w.sum()


def solve_tc_mean_variance_qp(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray,
    eta: float,
    lambda_l2: float,
    lambda_tc: float,
    turnover_mode: str,
    long_only: bool,
    stock_matrix: np.ndarray | None = None,
    prev_stock_vec: np.ndarray | None = None,
) -> np.ndarray:
    """
    Rolling TC-aware ablation objective:
        min_w 0.5 w'Σw - eta μ'w + 0.5 λ2||w||_2^2 + λtc * turnover(w)

    Returned weights are gross-exposure-normalized to produce stable, investable
    wealth series (especially for long/short configurations).
    """
    if turnover_mode not in {"portfolio", "stock"}:
        raise ValueError("turnover_mode must be 'portfolio' or 'stock'.")

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)

    k = len(mu)

    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(k)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    if turnover_mode == "stock" and stock_matrix is not None and stock_matrix.size > 0:
        M = np.asarray(stock_matrix, dtype=float)
        if prev_stock_vec is None:
            prev_stock_vec = np.zeros(M.shape[0])
        else:
            prev_stock_vec = np.asarray(prev_stock_vec, dtype=float)

        def tc_penalty(w: np.ndarray) -> float:
            return float(np.sum(np.abs(M @ w - prev_stock_vec)))
    else:

        def tc_penalty(w: np.ndarray) -> float:
            return float(np.sum(np.abs(w - w_prev)))

    def obj(w: np.ndarray) -> float:
        return float(
            0.5 * w @ sigma @ w
            - float(eta) * float(w @ mu)
            + 0.5 * float(lambda_l2) * float(np.sum(w * w))
            + float(lambda_tc) * tc_penalty(w)
        )

    # Constrain net budget; gross exposure is normalized after solve.
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if long_only:
        bounds = [(0.0, 1.0) for _ in range(k)]
        x0 = np.clip(w_prev, 0.0, 1.0)
    else:
        bounds = [(-1.0, 1.0) for _ in range(k)]
        x0 = w_prev.copy()

    if abs(x0.sum()) <= 1e-12:
        x0 = np.ones(k) / k
    else:
        x0 = x0 / x0.sum()

    res = minimize(
        obj,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )

    if not res.success:
        return _normalize_gross_exposure(w_prev)

    w = np.asarray(res.x, dtype=float)
    if np.any(~np.isfinite(w)):
        return _normalize_gross_exposure(w_prev)

    w[np.abs(w) < 1e-12] = 0.0
    return _normalize_gross_exposure(w)


def _final_stock_weights_for_month(
    month_panel: pd.DataFrame,
    candidate_weights: np.ndarray,
    n_bins: tuple[int, int, int],
    feat_cols: tuple[str, str, str] = ("LME", "OP", "Investment"),
    size_col: str = "size",
    permno_col: str = "permno",
) -> pd.Series:
    df = month_panel[[permno_col, size_col, *feat_cols]].copy()
    df[permno_col] = df[permno_col].astype(str)
    df[size_col] = df[size_col].astype(float)

    df = df[df[size_col].notna() & (df[size_col] > 0)].copy()
    if df.empty:
        return pd.Series(dtype=float)

    b1 = ntile_r(df[feat_cols[0]].astype(float), n_bins[0])
    b2 = ntile_r(df[feat_cols[1]].astype(float), n_bins[1])
    b3 = ntile_r(df[feat_cols[2]].astype(float), n_bins[2])

    ok = b1.notna() & b2.notna() & b3.notna()
    df = df.loc[ok].copy()
    if df.empty:
        return pd.Series(dtype=float)

    b1 = b1.loc[df.index].astype(int)
    b2 = b2.loc[df.index].astype(int)
    b3 = b3.loc[df.index].astype(int)

    n2, n3 = n_bins[1], n_bins[2]
    bucket_id = (b1 - 1) * (n2 * n3) + (b2 - 1) * n3 + (b3 - 1) + 1
    df["bucket_id"] = bucket_id.astype(int)

    sum_size = df.groupby("bucket_id")[size_col].transform("sum")
    df["base_w"] = df[size_col] / sum_size

    w = np.asarray(candidate_weights, dtype=float)
    df["final_w"] = df["base_w"] * df["bucket_id"].map(lambda k: float(w[int(k) - 1]))

    out = df.groupby(permno_col)["final_w"].sum()
    out = out[out.abs() > 1e-14]

    total = float(out.sum())
    if abs(total) > 1e-12:
        out = out / total

    return out


def _stock_weight_matrix_for_month(
    month_panel: pd.DataFrame,
    n_bins: tuple[int, int, int],
    feat_cols: tuple[str, str, str] = ("LME", "OP", "Investment"),
    size_col: str = "size",
    permno_col: str = "permno",
) -> tuple[np.ndarray, pd.Index]:
    """
    Return M (n_stocks x n_ports) such that final_stock_weights = M @ w_candidate.
    Column j contains within-bucket value weights for bucket j.
    """
    df = month_panel[[permno_col, size_col, *feat_cols]].copy()
    df[permno_col] = df[permno_col].astype(str)
    df[size_col] = df[size_col].astype(float)

    df = df[df[size_col].notna() & (df[size_col] > 0)].copy()
    if df.empty:
        return np.zeros((0, n_bins[0] * n_bins[1] * n_bins[2])), pd.Index([])

    b1 = ntile_r(df[feat_cols[0]].astype(float), n_bins[0])
    b2 = ntile_r(df[feat_cols[1]].astype(float), n_bins[1])
    b3 = ntile_r(df[feat_cols[2]].astype(float), n_bins[2])

    ok = b1.notna() & b2.notna() & b3.notna()
    df = df.loc[ok].copy()
    if df.empty:
        return np.zeros((0, n_bins[0] * n_bins[1] * n_bins[2])), pd.Index([])

    b1 = b1.loc[df.index].astype(int)
    b2 = b2.loc[df.index].astype(int)
    b3 = b3.loc[df.index].astype(int)

    n2, n3 = n_bins[1], n_bins[2]
    bucket_id = (b1 - 1) * (n2 * n3) + (b2 - 1) * n3 + (b3 - 1) + 1
    df["bucket_id"] = bucket_id.astype(int)

    sum_size = df.groupby("bucket_id")[size_col].transform("sum")
    df["base_w"] = df[size_col] / sum_size

    permnos = pd.Index(df[permno_col].astype(str).tolist())
    n_ports = n_bins[0] * n_bins[1] * n_bins[2]
    M = np.zeros((len(df), n_ports), dtype=float)

    # bucket_id is 1..n_ports; map to 0-indexed column.
    cols = (df["bucket_id"].to_numpy(dtype=int) - 1).clip(0, n_ports - 1)
    rows = np.arange(len(df), dtype=int)
    M[rows, cols] = df["base_w"].to_numpy(dtype=float)

    return M, permnos


def _stock_turnover(curr: pd.Series, prev: pd.Series | None) -> float:
    if prev is None or prev.empty:
        return float(curr.abs().sum())
    idx = curr.index.union(prev.index)
    curr_aligned = curr.reindex(idx, fill_value=0.0)
    prev_aligned = prev.reindex(idx, fill_value=0.0)
    return float((curr_aligned - prev_aligned).abs().sum())


def static_paper_style_optimize(
    returns_df: pd.DataFrame,
    n_train_valid: int,
    panel: pd.DataFrame | None = None,
    n_bins: tuple[int, int, int] = (2, 4, 4),
    cv_n: int = 3,
    lambda_l1: float = 0.0,
    lambda_l2: float = 1e-3,
    mu0: float | None = None,
    long_only: bool = True,
    method_name: str = "Triple Sort static",
    cost_per_turnover: float = 0.0,
    use_stock_level_turnover: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)
    cols = _candidate_cols(df)

    if n_train_valid >= len(df):
        raise ValueError("n_train_valid must be smaller than number of months.")

    x = df[cols].astype(float).fillna(0.0)
    train_valid = x.iloc[:n_train_valid]
    test = x.iloc[n_train_valid:]

    mu = train_valid.mean(axis=0).to_numpy()
    sigma = np.cov(train_valid.to_numpy(), rowvar=False, ddof=1)

    w = solve_portfolio_qp(
        mu=mu,
        sigma=sigma,
        w_prev=None,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        lambda_tc=0.0,
        mu0=mu0,
        long_only=long_only,
    )

    gross = test.to_numpy() @ w

    meta_cols = _meta_cols(df)
    result = df.iloc[n_train_valid:][meta_cols].copy()
    result["method"] = method_name
    result["gross_ret"] = gross

    turnovers: list[float] = []
    costs: list[float] = []
    prev_stock_w: pd.Series | None = None

    if use_stock_level_turnover:
        if panel is None:
            raise ValueError("panel is required when use_stock_level_turnover=True")
        panel_g = panel.groupby(["yy", "mm"], sort=False)

    for _, row in result.iterrows():
        if use_stock_level_turnover:
            m = panel_g.get_group((int(row["yy"]), int(row["mm"])))
            curr_stock_w = _final_stock_weights_for_month(
                m,
                candidate_weights=w,
                n_bins=n_bins,
            )
            turnover = _stock_turnover(curr_stock_w, prev_stock_w)
            prev_stock_w = curr_stock_w
        else:
            turnover = 0.0

        turnovers.append(turnover)
        costs.append(float(cost_per_turnover) * turnover)

    result["turnover_raw"] = turnovers
    result["turnover"] = turnovers
    result["cost"] = costs
    result["net_ret"] = result["gross_ret"] - result["cost"]

    weights = pd.DataFrame({"candidate": cols, "weight": w})
    weights = (
        weights[weights["weight"].abs() > 1e-12]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    def _safe_sr(xr: np.ndarray) -> float:
        xr = np.asarray(xr, dtype=float)
        xr = xr[np.isfinite(xr)]
        if len(xr) < 2:
            return np.nan
        sd = xr.std(ddof=1)
        if not np.isfinite(sd) or sd < 1e-12:
            return np.nan
        return float(xr.mean() / sd)

    n_valid = int(n_train_valid / cv_n)
    n_train = n_train_valid - n_valid

    cv_train_ret = x.iloc[:n_train].to_numpy() @ w
    cv_valid_ret = x.iloc[n_train:n_train_valid].to_numpy() @ w
    full_train_valid_ret = x.iloc[:n_train_valid].to_numpy() @ w
    test_ret = x.iloc[n_train_valid:].to_numpy() @ w

    # Triple Sort has no SDF-vs-trade distinction; keep both columns equal.
    diag = pd.DataFrame(
        {
            "sample": ["cv_train", "cv_valid", "full_train_valid", "test"],
            "start_row": [0, n_train, 0, n_train_valid],
            "end_row_exclusive": [n_train, n_train_valid, n_train_valid, len(df)],
            "sharpe_sdf_monthly": [
                _safe_sr(cv_train_ret),
                _safe_sr(cv_valid_ret),
                _safe_sr(full_train_valid_ret),
                _safe_sr(test_ret),
            ],
            "sharpe_trade_monthly": [
                _safe_sr(cv_train_ret),
                _safe_sr(cv_valid_ret),
                _safe_sr(full_train_valid_ret),
                _safe_sr(test_ret),
            ],
        }
    )

    return result.reset_index(drop=True), weights, diag


def rolling_tc_optimize(
    returns_df: pd.DataFrame,
    window: int,
    panel: pd.DataFrame | None = None,
    n_bins: tuple[int, int, int] = (2, 4, 4),
    lambda_l2: float = 1e-3,
    lambda_tc: float = 0.0025,
    eta: float = 1.0,
    cost_per_turnover: float = 0.0025,
    long_only: bool = True,
    method_name: str = "Triple Sort rolling TC-aware",
    turnover_mode: str = "portfolio",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)
    cols = _candidate_cols(df)

    x = df[cols].astype(float).fillna(0.0).reset_index(drop=True)
    k = len(cols)

    w_prev = np.ones(k) / k
    prev_stock_w: pd.Series | None = None

    rows: list[dict] = []
    w_rows: list[dict] = []

    meta_cols = _meta_cols(df)

    if turnover_mode not in {"portfolio", "stock"}:
        raise ValueError("turnover_mode must be 'portfolio' or 'stock'.")

    if turnover_mode == "stock":
        if panel is None:
            raise ValueError("panel is required when turnover_mode='stock'")
        panel_g = panel.groupby(["yy", "mm"], sort=False)

    for t in range(window, len(df)):
        hist = x.iloc[t - window : t]

        mu = hist.mean(axis=0).to_numpy()
        sigma = np.cov(hist.to_numpy(), rowvar=False, ddof=1)

        meta = df.iloc[t][meta_cols].to_dict()

        stock_matrix = None
        prev_stock_vec = None
        permnos = pd.Index([])

        if turnover_mode == "stock":
            m = panel_g.get_group((int(meta["yy"]), int(meta["mm"])))
            stock_matrix, permnos = _stock_weight_matrix_for_month(m, n_bins=n_bins)
            if len(permnos) > 0:
                if prev_stock_w is None:
                    prev_stock_vec = np.zeros(len(permnos))
                else:
                    prev_stock_vec = prev_stock_w.reindex(permnos, fill_value=0.0).to_numpy()

        w = solve_tc_mean_variance_qp(
            mu=mu,
            sigma=sigma,
            w_prev=w_prev,
            lambda_tc=lambda_tc,
            lambda_l2=lambda_l2,
            eta=eta,
            long_only=long_only,
            turnover_mode=turnover_mode,
            stock_matrix=stock_matrix,
            prev_stock_vec=prev_stock_vec,
        )

        gross = float(x.iloc[t].to_numpy() @ w)
        raw_turnover = float(np.sum(np.abs(w - w_prev)))

        if turnover_mode == "stock":
            curr_stock_w = _final_stock_weights_for_month(
                panel_g.get_group((int(meta["yy"]), int(meta["mm"]))),
                candidate_weights=w,
                n_bins=n_bins,
            )
            turnover = _stock_turnover(curr_stock_w, prev_stock_w)
            prev_stock_w = curr_stock_w
        else:
            turnover = raw_turnover

        cost = float(cost_per_turnover) * turnover
        net = gross - cost

        rows.append(
            {
                **meta,
                "method": method_name,
                "gross_ret": gross,
                "turnover_raw": raw_turnover,
                "turnover": turnover,
                "cost": cost,
                "net_ret": net,
            }
        )

        for c, wi in zip(cols, w):
            if abs(wi) > 1e-12:
                w_rows.append({**meta, "candidate": c, "weight_trade": float(wi)})

        w_prev = w

    return pd.DataFrame(rows), pd.DataFrame(w_rows)
