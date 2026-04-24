from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _candidate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("port_")]


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
    """
    Solve:
        min 0.5 w'Σw + λ1||w||1 + 0.5λ2||w||2^2 + λtc||w-w_prev||1
        s.t. 1'w = 1, optionally μ'w >= μ0.

    For long_only=True, ||w||1 is constant because sum(w)=1, but we keep the term
    for consistency with the methodology.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    k = len(mu)
    if w_prev is None:
        w_prev = np.ones(k) / k
    else:
        w_prev = np.asarray(w_prev, dtype=float)

    # Numerical ridge for stable covariance optimization.
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

    bounds = [(0.0, 1.0) for _ in range(k)] if long_only else [(-2.0, 2.0) for _ in range(k)]
    x0 = np.clip(w_prev, 0, 1) if long_only else w_prev.copy()
    if abs(x0.sum()) < 1e-12:
        x0 = np.ones(k) / k
    else:
        x0 = x0 / x0.sum()

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success:
        # Fallback: minimum-variance-ish equal weighted feasible portfolio.
        return np.ones(k) / k
    w = np.asarray(res.x, dtype=float)
    w[np.abs(w) < 1e-12] = 0.0
    return w / w.sum()


def static_paper_style_optimize(
    returns_df: pd.DataFrame,
    n_train_valid: int,
    cv_n: int = 3,
    lambda_l1: float = 0.0,
    lambda_l2: float = 1e-3,
    mu0: float | None = None,
    long_only: bool = True,
    method_name: str = "Static paper-style + residual momentum tilt",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Paper-faithful no-rolling optimizer.

    Uses a fixed train/validation/test split analogous to lasso_valid_full.R:
      - first n_train_valid months are train+validation
      - last fold within train_valid is validation, if wanted for diagnostics
      - remaining months are test
      - final weights are fit once on all train+valid and applied unchanged to test.
    """
    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)
    cols = _candidate_cols(df)
    if n_train_valid >= len(df):
        raise ValueError("n_train_valid must be smaller than number of months.")

    x = df[cols].astype(float).fillna(0.0)
    train_valid = x.iloc[:n_train_valid]
    test = x.iloc[n_train_valid:]

    mu = train_valid.mean(axis=0).to_numpy()
    sigma = np.cov(train_valid.to_numpy(), rowvar=False, ddof=1)
    w = solve_portfolio_qp(mu, sigma, None, lambda_l1, lambda_l2, 0.0, mu0, long_only)

    gross = test.to_numpy() @ w
    result = df.iloc[n_train_valid:][[c for c in ["date", "date_dt", "yy", "mm"] if c in df.columns]].copy()
    result["method"] = method_name
    result["gross_ret"] = gross
    result["turnover"] = 0.0
    result["cost"] = 0.0
    result["net_ret"] = result["gross_ret"]

    weights = pd.DataFrame({"candidate": cols, "weight": w})
    weights = weights[weights["weight"].abs() > 1e-12].sort_values("weight", ascending=False).reset_index(drop=True)

    # Optional split diagnostics: same fixed blocks, not rolling.
    n_valid = int(n_train_valid / cv_n)
    n_train = n_train_valid - n_valid
    train_ret = x.iloc[:n_train].to_numpy() @ w
    valid_ret = x.iloc[n_train:n_train_valid].to_numpy() @ w
    test_ret = gross
    diag = pd.DataFrame({
        "sample": ["train", "valid", "test"],
        "start_row": [0, n_train, n_train_valid],
        "end_row_exclusive": [n_train, n_train_valid, len(df)],
        "mean_monthly": [np.mean(train_ret), np.mean(valid_ret), np.mean(test_ret)],
        "std_monthly": [np.std(train_ret, ddof=1), np.std(valid_ret, ddof=1), np.std(test_ret, ddof=1)],
    })
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
    method_name: str = "Rolling TC-aware + residual momentum tilt",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling transaction-cost-aware optimizer from the PDF framework.
    At month t, estimate moments from the prior window and apply w_t to x_{t+1}.
    """
    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)
    cols = _candidate_cols(df)
    x = df[cols].astype(float).fillna(0.0).reset_index(drop=True)
    k = len(cols)
    w_prev = np.ones(k) / k
    rows, w_rows = [], []

    for t in range(window, len(df)):
        hist = x.iloc[t - window:t]
        mu = hist.mean(axis=0).to_numpy()
        sigma = np.cov(hist.to_numpy(), rowvar=False, ddof=1)
        w = solve_portfolio_qp(mu, sigma, w_prev, lambda_l1, lambda_l2, lambda_tc, mu0, long_only)

        gross = float(x.iloc[t].to_numpy() @ w)
        turnover = float(np.sum(np.abs(w - w_prev)))
        cost = cost_per_turnover * turnover
        meta = df.iloc[t][[c for c in ["date", "date_dt", "yy", "mm"] if c in df.columns]].to_dict()
        rows.append({**meta, "method": method_name, "gross_ret": gross, "turnover": turnover, "cost": cost, "net_ret": gross - cost})
        for c, wi in zip(cols, w):
            if abs(wi) > 1e-12:
                w_rows.append({**meta, "candidate": c, "weight": wi})
        w_prev = w

    return pd.DataFrame(rows), pd.DataFrame(w_rows)
