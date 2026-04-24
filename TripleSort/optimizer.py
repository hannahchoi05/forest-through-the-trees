from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import ntile_r


def _candidate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("port_")]


def _meta_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["date_dt", "yy", "mm"] if c in df.columns]


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
        bounds = [(-2.0, 2.0) for _ in range(k)]
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
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not res.success:
        return np.ones(k) / k

    w = np.asarray(res.x, dtype=float)
    w[np.abs(w) < 1e-12] = 0.0
    if abs(w.sum()) < 1e-12:
        return np.ones(k) / k
    return w / w.sum()


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

    n_valid = int(n_train_valid / cv_n)
    n_train = n_train_valid - n_valid

    train_ret = x.iloc[:n_train].to_numpy() @ w
    valid_ret = x.iloc[n_train:n_train_valid].to_numpy() @ w
    test_ret = gross

    diag = pd.DataFrame(
        {
            "sample": ["train", "valid", "test"],
            "start_row": [0, n_train, n_train_valid],
            "end_row_exclusive": [n_train, n_train_valid, len(df)],
            "mean_monthly": [np.mean(train_ret), np.mean(valid_ret), np.mean(test_ret)],
            "std_monthly": [
                np.std(train_ret, ddof=1),
                np.std(valid_ret, ddof=1),
                np.std(test_ret, ddof=1),
            ],
        }
    )
    diag["sharpe_ann"] = np.sqrt(12.0) * diag["mean_monthly"] / diag["std_monthly"]

    return result.reset_index(drop=True), weights, diag


def rolling_tc_optimize(
    returns_df: pd.DataFrame,
    window: int,
    panel: pd.DataFrame | None = None,
    n_bins: tuple[int, int, int] = (2, 4, 4),
    lambda_l1: float = 0.0,
    lambda_l2: float = 1e-3,
    lambda_tc: float = 0.0025,
    cost_per_turnover: float = 0.0025,
    mu0: float | None = None,
    long_only: bool = True,
    method_name: str = "Triple Sort rolling TC-aware",
    use_stock_level_turnover: bool = False,
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

    if use_stock_level_turnover:
        if panel is None:
            raise ValueError("panel is required when use_stock_level_turnover=True")
        panel_g = panel.groupby(["yy", "mm"], sort=False)

    for t in range(window, len(df)):
        hist = x.iloc[t - window : t]

        mu = hist.mean(axis=0).to_numpy()
        sigma = np.cov(hist.to_numpy(), rowvar=False, ddof=1)

        w = solve_portfolio_qp(
            mu=mu,
            sigma=sigma,
            w_prev=w_prev,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_tc=lambda_tc,
            mu0=mu0,
            long_only=long_only,
        )

        gross = float(x.iloc[t].to_numpy() @ w)
        raw_turnover = float(np.sum(np.abs(w - w_prev)))

        meta = df.iloc[t][meta_cols].to_dict()

        if use_stock_level_turnover:
            m = panel_g.get_group((int(meta["yy"]), int(meta["mm"])))
            curr_stock_w = _final_stock_weights_for_month(
                m,
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
                w_rows.append({**meta, "candidate": c, "weight": float(wi)})

        w_prev = w

    return pd.DataFrame(rows), pd.DataFrame(w_rows)

