from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


STATIC_TURNOVER_PROXY = 0.05


def _candidate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("port_")]


def _meta_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["date", "date_dt", "yy", "mm"] if c in df.columns]


def _candidate_to_node_id(candidate: str) -> str:
    if candidate.startswith("port_"):
        return candidate[len("port_"):]
    return candidate


def _prep_stock_weights(
    stock_weights: pd.DataFrame | str | Path | None,
) -> pd.DataFrame | Path | None:
    if stock_weights is None:
        return None

    if isinstance(stock_weights, (str, Path)):
        sw_path = Path(stock_weights)
        if not sw_path.exists():
            raise FileNotFoundError(f"stock_weights path not found: {sw_path}")
        return sw_path

    if stock_weights.empty:
        return None

    sw = stock_weights.copy()

    if "date_dt" in sw.columns:
        sw["date_dt"] = pd.to_datetime(sw["date_dt"])

    sw["node_id"] = sw["node_id"].astype(str)
    sw["permno"] = sw["permno"].astype(str)

    return sw


def _month_stock_weights(
    stock_weights: pd.DataFrame | Path | None,
    meta: dict,
    candidate_weights: pd.Series,
    stock_weight_col: str = "tilt_stock_w",
) -> pd.Series:
    """
    Convert optimizer weights on AP-tree candidate portfolios into final stock weights.

    final_stock_weight_i,t =
        sum_p optimizer_weight_p,t * stock_weight_i,p,t

    Note: for pure AP-Trees (tau=0), base_stock_w == tilt_stock_w exactly, so the
    default of "tilt_stock_w" still produces the correct value-weighted result.
    """
    if stock_weights is None:
        return pd.Series(dtype=float)

    if isinstance(stock_weights, Path):
        yy = int(meta["yy"])
        mm = int(meta["mm"])
        month_file = stock_weights / f"{yy:04d}_{mm:02d}.pkl"
        if not month_file.exists():
            return pd.Series(dtype=float)

        m = pd.read_pickle(month_file, compression="gzip")
        if m.empty:
            return pd.Series(dtype=float)

        m["node_id"] = m["node_id"].astype(str)
        m["permno"] = m["permno"].astype(str)
    else:
        sw = stock_weights

        if "date_dt" in sw.columns and "date_dt" in meta:
            date_val = pd.to_datetime(meta["date_dt"])
            m = sw[sw["date_dt"].eq(date_val)].copy()
        elif "yy" in sw.columns and "mm" in sw.columns:
            m = sw[
                (sw["yy"].astype(int).eq(int(meta["yy"])))
                & (sw["mm"].astype(int).eq(int(meta["mm"])))
            ].copy()
        else:
            raise ValueError("stock_weights must contain date_dt or yy/mm columns.")

    if m.empty:
        return pd.Series(dtype=float)

    cw = candidate_weights.copy()
    cw.index = [_candidate_to_node_id(c) for c in cw.index]
    cw = cw[cw.abs() > 1e-12]

    if cw.empty:
        return pd.Series(dtype=float)

    wmap = cw.rename("optimizer_w").reset_index()
    wmap = wmap.rename(columns={"index": "node_id"})
    wmap["node_id"] = wmap["node_id"].astype(str)

    m = m.merge(wmap, on="node_id", how="inner")

    if m.empty:
        return pd.Series(dtype=float)

    m["final_stock_w"] = m["optimizer_w"].astype(float) * m[stock_weight_col].astype(float)

    out = m.groupby("permno")["final_stock_w"].sum()
    out = out[out.abs() > 1e-14]

    total = out.sum()
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


def static_paper_style_optimize(
    returns_df: pd.DataFrame,
    n_train_valid: int,
    cv_n: int = 3,
    lambda_l1: float = 0.0,
    lambda_l2: float = 1e-3,
    mu0: float | None = None,
    long_only: bool = True,
    method_name: str = "Static paper-style",
    cost_per_turnover: float = 0.0025,
    stock_weights: pd.DataFrame | str | Path | None = None,
    use_stock_level_turnover: bool = True,
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

    candidate_w = pd.Series(w, index=cols)
    sw = _prep_stock_weights(stock_weights)

    turnovers = []
    costs = []
    stock_weight_rows = []
    prev_stock_w = None

    for _, row in result.iterrows():
        meta = {c: row[c] for c in meta_cols}

        if use_stock_level_turnover and sw is not None:
            curr_stock_w = _month_stock_weights(
                stock_weights=sw,
                meta=meta,
                candidate_weights=candidate_w,
                stock_weight_col="tilt_stock_w",
            )
            turnover = _stock_turnover(curr_stock_w, prev_stock_w)

            for permno, wi in curr_stock_w.items():
                stock_weight_rows.append({**meta, "permno": permno, "stock_weight": wi})

            prev_stock_w = curr_stock_w
        else:
            turnover = STATIC_TURNOVER_PROXY

        turnovers.append(turnover)
        costs.append(turnover * cost_per_turnover)

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
            "mean_monthly": [
                np.mean(train_ret),
                np.mean(valid_ret),
                np.mean(test_ret),
            ],
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
    lambda_l1: float = 0.0,
    lambda_l2: float = 1e-3,
    lambda_tc: float = 0.0025,
    cost_per_turnover: float = 0.0025,
    mu0: float | None = None,
    long_only: bool = True,
    method_name: str = "Rolling TC-aware",
    stock_weights: pd.DataFrame | str | Path | None = None,
    use_stock_level_turnover: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)
    cols = _candidate_cols(df)

    x = df[cols].astype(float).fillna(0.0).reset_index(drop=True)
    k = len(cols)

    w_prev = np.ones(k) / k
    prev_stock_w = None

    rows = []
    w_rows = []
    stock_weight_rows = []

    sw = _prep_stock_weights(stock_weights)
    meta_cols = _meta_cols(df)

    for t in range(window, len(df)):
        hist = x.iloc[t - window:t]

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
        candidate_w = pd.Series(w, index=cols)

        if use_stock_level_turnover and sw is not None:
            curr_stock_w = _month_stock_weights(
                stock_weights=sw,
                meta=meta,
                candidate_weights=candidate_w,
                stock_weight_col="tilt_stock_w",
            )
            turnover = _stock_turnover(curr_stock_w, prev_stock_w)

            for permno, wi in curr_stock_w.items():
                stock_weight_rows.append({**meta, "permno": permno, "stock_weight": wi})

            prev_stock_w = curr_stock_w
        else:
            turnover = raw_turnover

        cost = cost_per_turnover * turnover
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
                w_rows.append({**meta, "candidate": c, "weight": wi})

        w_prev = w

    weights = pd.DataFrame(w_rows)

    if stock_weight_rows:
        final_stock_weights = pd.DataFrame(stock_weight_rows)
        weights = pd.concat(
            [weights, final_stock_weights.assign(candidate="__FINAL_STOCK_WEIGHT__")],
            ignore_index=True,
            sort=False,
        )

    return pd.DataFrame(rows), weights