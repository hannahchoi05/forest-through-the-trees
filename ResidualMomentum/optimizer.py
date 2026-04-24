
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from sklearn.linear_model import lars_path
except ImportError:
    lars_path = None


STATIC_TURNOVER_PROXY = 0.05


# ============================================================
# Helpers
# ============================================================

def _candidate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("port_")]


def _meta_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["date", "date_dt", "yy", "mm"] if c in df.columns]


def _candidate_to_node_id(candidate: str) -> str:
    if candidate.startswith("port_"):
        return candidate[len("port_"):]
    return candidate


def _node_depth_from_candidate(candidate: str) -> int:
    """
    Candidate names are expected to look like:
        port_T1111_N1
        port_T1111_N11
        port_T1111_N111
        ...

    Node path "1" has depth 0.
    Node path "11" has depth 1.
    Node path "11111" has depth 4.

    R AP_Pruning.R uses:
        adj_w = 1 / sqrt(2^depths)
    """
    node_id = _candidate_to_node_id(candidate)
    if "_N" not in node_id:
        return 0
    path = node_id.split("_N", 1)[1]
    return max(len(path) - 1, 0)


def _depth_adjustment(cols: list[str]) -> np.ndarray:
    depths = np.array([_node_depth_from_candidate(c) for c in cols], dtype=float)
    return 1.0 / np.sqrt(2.0 ** depths)


def _safe_sr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    return float(x.mean() / sd)


def _normalize_gross_exposure(w: np.ndarray) -> np.ndarray:
    """
    Convert SDF/path weights into an investable trading-weight convention.

    This is deliberately separate from the faithful R-style AP-pruning weights.
    Paper replication diagnostics use the original SDF weights.
    Wealth plots / TC / backtest returns use this gross-normalized version.
    """
    w = np.asarray(w, dtype=float)
    gross = float(np.sum(np.abs(w)))
    if not np.isfinite(gross) or gross < 1e-12:
        return np.zeros_like(w)
    return w / gross


def _prep_stock_weights(stock_weights):
    if stock_weights is None:
        return None

    if isinstance(stock_weights, (str, Path)):
        return Path(stock_weights)

    if isinstance(stock_weights, pd.DataFrame):
        if stock_weights.empty:
            return None

        sw = stock_weights.copy()

        if "date_dt" in sw.columns:
            sw["date_dt"] = pd.to_datetime(sw["date_dt"])

        sw["node_id"] = sw["node_id"].astype(str)
        sw["permno"] = sw["permno"].astype(str)

        return sw

    raise TypeError(f"Unsupported stock_weights type: {type(stock_weights)}")


def _load_month_stock_weights_from_dir(stock_weights_dir: Path, meta: dict) -> pd.DataFrame:
    file_path = stock_weights_dir / f"{int(meta['yy'])}_{int(meta['mm']):02d}.parquet"
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(file_path)


def _month_stock_weights(
    stock_weights,
    meta: dict,
    candidate_weights: pd.Series,
    stock_weight_col: str = "tilt_stock_w",
) -> pd.Series:
    """
    Convert optimizer weights on tree candidate portfolios into final stock weights:
        W_stock,t = sum_j w_candidate,j,t * W_stock_given_candidate,j,t
    """
    if stock_weights is None:
        return pd.Series(dtype=float)

    if isinstance(stock_weights, Path):
        m = _load_month_stock_weights_from_dir(stock_weights, meta)
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

    m["node_id"] = m["node_id"].astype(str)
    m["permno"] = m["permno"].astype(str)
    m = m.merge(wmap, on="node_id", how="inner")

    if m.empty:
        return pd.Series(dtype=float)

    m["final_stock_w"] = m["optimizer_w"].astype(float) * m[stock_weight_col].astype(float)

    out = m.groupby("permno")["final_stock_w"].sum()
    out = out[out.abs() > 1e-14]

    return out


def _stock_turnover(curr: pd.Series, prev: pd.Series | None) -> float:
    if prev is None or prev.empty:
        return float(curr.abs().sum())

    idx = curr.index.union(prev.index)
    curr_aligned = curr.reindex(idx, fill_value=0.0)
    prev_aligned = prev.reindex(idx, fill_value=0.0)

    return float((curr_aligned - prev_aligned).abs().sum())


def _stock_weight_matrix_for_month(
    stock_weights,
    meta: dict,
    cols: list[str],
    stock_weight_col: str = "tilt_stock_w",
) -> tuple[np.ndarray, pd.Index]:
    """
    Matrix M has shape (#stocks in month, #candidate portfolios).
    M @ w_candidate gives stock-level weights.
    """
    if stock_weights is None:
        return np.zeros((0, len(cols))), pd.Index([])

    if isinstance(stock_weights, Path):
        m = _load_month_stock_weights_from_dir(stock_weights, meta)
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
        return np.zeros((0, len(cols))), pd.Index([])

    node_order = [_candidate_to_node_id(c) for c in cols]
    node_to_col = {node: j for j, node in enumerate(node_order)}

    m["node_id"] = m["node_id"].astype(str)
    m["permno"] = m["permno"].astype(str)
    m = m[m["node_id"].isin(node_to_col)].copy()

    if m.empty:
        return np.zeros((0, len(cols))), pd.Index([])

    permnos = pd.Index(sorted(m["permno"].unique()))
    perm_to_row = {p: i for i, p in enumerate(permnos)}

    M = np.zeros((len(permnos), len(cols)), dtype=float)

    for row in m.itertuples(index=False):
        i = perm_to_row[getattr(row, "permno")]
        j = node_to_col[getattr(row, "node_id")]
        M[i, j] += float(getattr(row, stock_weight_col))

    return M, permnos


# ============================================================
# Faithful AP-pruning based on released R code
# ============================================================

@dataclass
class APPruningSelection:
    lambda0_index: int
    lambda2_index: int
    lambda0: float
    lambda2: float
    portsN: int
    cv_valid_sr_sdf: float
    full_train_sr_sdf: float
    full_test_sr_sdf: float
    full_train_sr_trade: float
    full_test_sr_trade: float


def _r_style_lasso_path(
    sigma_tilde: np.ndarray,
    mu_tilde: np.ndarray,
    lambda2: float,
    kmin: int,
    kmax: int,
) -> list[dict]:
    """
    Counterpart of lasso.R:

        yy = c(y, rep(0, p))
        XX = rbind(X, diag(sqrt(lambda2), p, p))

        lasso_obj = lars(XX, yy, type="lasso",
                         normalize=FALSE, intercept=FALSE)

        beta = coef(lasso_obj)
        K = apply(beta, 1, function(x) sum(x != 0))
        subset = K >= kmin & K <= kmax

    No candidate pre-cap.
    """
    if lars_path is None:
        raise ImportError("Install scikit-learn: pip install scikit-learn")

    p = sigma_tilde.shape[1]
    X_aug = np.vstack([sigma_tilde, np.sqrt(lambda2) * np.eye(p)])
    y_aug = np.concatenate([mu_tilde, np.zeros(p)])

    _, _, coefs = lars_path(
        X_aug,
        y_aug,
        method="lasso",
        verbose=False,
        max_iter=min(500, p),
    )

    rows: list[dict] = []

    for step in range(coefs.shape[1]):
        beta = np.asarray(coefs[:, step], dtype=float)
        beta[np.abs(beta) < 1e-12] = 0.0
        portsN = int(np.sum(beta != 0.0))

        if kmin <= portsN <= kmax:
            rows.append({"step": step, "beta": beta, "portsN": portsN})

    return rows


def _ap_pruning_run_one(
    ports_train_raw: np.ndarray,
    ports_valid_raw: np.ndarray | None,
    ports_test_raw: np.ndarray,
    adj_w: np.ndarray,
    lambda0: float,
    lambda2: float,
    kmin: int,
    kmax: int,
) -> list[dict]:
    """
    For one lambda0/lambda2 pair, this follows the R AP-pruning calculation.

    Important:
    - `weights_sdf` is the faithful R-style SDF/path weight:
          b = beta * adj_w
          b = b / abs(sum(b))
    - `weights_trade` is the investable plotting/TC version:
          b_trade = b / sum(abs(b))
      This is NOT used to select lambda0/lambda2; it is only used for wealth
      plots and transaction-cost analysis.
    """
    ports_train_adj = ports_train_raw * adj_w

    mu = ports_train_adj.mean(axis=0)
    sigma = np.cov(ports_train_adj, rowvar=False, ddof=1)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = 0.5 * (sigma + sigma.T)

    eigvals, eigvecs = np.linalg.eigh(sigma)
    keep = eigvals > 1e-10
    if keep.sum() == 0:
        return []

    D = eigvals[keep]
    V = eigvecs[:, keep]

    sigma_tilde = V @ np.diag(np.sqrt(D)) @ V.T
    mu_bar = float(mu.mean())
    mu_tilde = V @ np.diag(1.0 / np.sqrt(D)) @ V.T @ (mu + lambda0 * mu_bar)

    path_rows = _r_style_lasso_path(
        sigma_tilde=sigma_tilde,
        mu_tilde=mu_tilde,
        lambda2=lambda2,
        kmin=kmin,
        kmax=kmax,
    )

    out: list[dict] = []

    for row in path_rows:
        beta = row["beta"]

        # Faithful R-style SDF/path weight.
        b_sdf = beta * adj_w
        denom = abs(float(np.sum(b_sdf)))

        if not np.isfinite(denom) or denom < 1e-12:
            continue

        b_sdf = b_sdf / denom

        # Investable version for wealth plots / TC.
        b_trade = _normalize_gross_exposure(b_sdf)
        if np.sum(np.abs(b_trade)) < 1e-12:
            continue

        train_sdf_ret = ports_train_raw @ b_sdf
        test_sdf_ret = ports_test_raw @ b_sdf

        train_trade_ret = ports_train_raw @ b_trade
        test_trade_ret = ports_test_raw @ b_trade

        record = {
            "train_SR_sdf": _safe_sr(train_sdf_ret),
            "test_SR_sdf": _safe_sr(test_sdf_ret),
            "train_SR_trade": _safe_sr(train_trade_ret),
            "test_SR_trade": _safe_sr(test_trade_ret),
            "portsN": int(row["portsN"]),
            "weights_sdf": b_sdf,
            "weights_trade": b_trade,
            "step": row["step"],
        }

        if ports_valid_raw is not None:
            valid_sdf_ret = ports_valid_raw @ b_sdf
            valid_trade_ret = ports_valid_raw @ b_trade
            record["valid_SR_sdf"] = _safe_sr(valid_sdf_ret)
            record["valid_SR_trade"] = _safe_sr(valid_trade_ret)

        out.append(record)

    return out


def ap_pruning_static_optimize(
    returns_df: pd.DataFrame,
    n_train_valid: int,
    cv_n: int,
    lambda0_grid: list[float],
    lambda2_grid: list[float],
    port_n: int,
    kmin: int = 5,
    kmax: int = 50,
    method_name: str = "AP-pruning static",
    cost_per_turnover: float = 0.0,
    stock_weights=None,
    use_stock_level_turnover: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, APPruningSelection]:
    """
    Static AP-pruning with two outputs:
      1. Faithful SDF weights/diagnostics for replication.
      2. Gross-normalized trading weights for backtest, cumulative wealth plots, and TC.

    The hyperparameter selection still follows the R-code style:
      - exact portsN == port_n
      - validation Sharpe on SDF payoff, not trading-normalized payoff
      - first exact row, no nearest-K fallback
    """
    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)

    if "date_dt" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date_dt"])

    cols = _candidate_cols(df)
    meta_cols = _meta_cols(df)

    if n_train_valid >= len(df):
        raise ValueError("n_train_valid must be smaller than number of months.")

    raw = df[cols].astype(float).fillna(0.0).to_numpy()
    adj_w = _depth_adjustment(cols)

    n_valid = int(n_train_valid / cv_n)
    n_train = n_train_valid - n_valid

    ports_train_cv = raw[:n_train]
    ports_valid_cv = raw[n_train:n_train_valid]
    ports_train_valid = raw[:n_train_valid]
    ports_test = raw[n_train_valid:]

    cv_table = {}
    best_valid_sr = -np.inf
    best_i = None
    best_j = None

    for i, lambda0 in enumerate(lambda0_grid, start=1):
        for j, lambda2 in enumerate(lambda2_grid, start=1):
            rows = _ap_pruning_run_one(
                ports_train_raw=ports_train_cv,
                ports_valid_raw=ports_valid_cv,
                ports_test_raw=ports_test,
                adj_w=adj_w,
                lambda0=float(lambda0),
                lambda2=float(lambda2),
                kmin=kmin,
                kmax=kmax,
            )

            exact_rows = [r for r in rows if r["portsN"] == port_n]

            if not exact_rows:
                raise RuntimeError(
                    f"No lasso-path row with portsN == {port_n} for "
                    f"lambda0 index {i}, lambda2 index {j}. "
                    f"Exact portN is required."
                )

            first = exact_rows[0]
            valid_sr = float(first["valid_SR_sdf"])
            cv_table[(i, j)] = first

            if np.isfinite(valid_sr) and valid_sr > best_valid_sr:
                best_valid_sr = valid_sr
                best_i = i
                best_j = j

    if best_i is None or best_j is None:
        raise RuntimeError("No finite validation Sharpe found in AP-pruning CV.")

    lambda0_best = float(lambda0_grid[best_i - 1])
    lambda2_best = float(lambda2_grid[best_j - 1])

    full_rows = _ap_pruning_run_one(
        ports_train_raw=ports_train_valid,
        ports_valid_raw=None,
        ports_test_raw=ports_test,
        adj_w=adj_w,
        lambda0=lambda0_best,
        lambda2=lambda2_best,
        kmin=kmin,
        kmax=kmax,
    )

    exact_full_rows = [r for r in full_rows if r["portsN"] == port_n]

    if not exact_full_rows:
        raise RuntimeError(
            f"Full refit has no lasso-path row with portsN == {port_n}. "
            f"Exact portN is required."
        )

    full_first = exact_full_rows[0]

    w_sdf = np.asarray(full_first["weights_sdf"], dtype=float)
    w_trade = np.asarray(full_first["weights_trade"], dtype=float)

    # Backtest/wealth plots use investable gross-normalized weights.
    gross_trade = ports_test @ w_trade
    gross_sdf = ports_test @ w_sdf

    result = df.iloc[n_train_valid:][meta_cols].copy()
    result["method"] = method_name
    result["gross_ret"] = gross_trade
    result["sdf_ret"] = gross_sdf

    sw = _prep_stock_weights(stock_weights)
    candidate_w_trade = pd.Series(w_trade, index=cols)

    turnovers = []
    costs = []
    prev_stock_w = None

    for _, row in result.iterrows():
        meta = {c: row[c] for c in meta_cols}

        if cost_per_turnover == 0.0:
            turnover = 0.0
        elif use_stock_level_turnover and sw is not None:
            curr_stock_w = _month_stock_weights(
                stock_weights=sw,
                meta=meta,
                candidate_weights=candidate_w_trade,
                stock_weight_col="tilt_stock_w",
            )
            turnover = _stock_turnover(curr_stock_w, prev_stock_w)
            prev_stock_w = curr_stock_w
        else:
            turnover = STATIC_TURNOVER_PROXY

        turnovers.append(turnover)
        costs.append(turnover * cost_per_turnover)

    result["turnover_raw"] = turnovers
    result["turnover"] = turnovers
    result["cost"] = costs
    result["net_ret"] = result["gross_ret"] - result["cost"]

    weights = pd.DataFrame(
        {
            "candidate": cols,
            "weight_sdf": w_sdf,
            "weight_trade": w_trade,
        }
    )
    weights = weights[
        (weights["weight_sdf"].abs() > 1e-12)
        | (weights["weight_trade"].abs() > 1e-12)
    ].reset_index(drop=True)

    diag = pd.DataFrame(
        [
            {
                "sample": "cv_train",
                "start_row": 0,
                "end_row_exclusive": n_train,
                "sharpe_sdf_monthly": cv_table[(best_i, best_j)]["train_SR_sdf"],
                "sharpe_trade_monthly": cv_table[(best_i, best_j)]["train_SR_trade"],
            },
            {
                "sample": "cv_valid",
                "start_row": n_train,
                "end_row_exclusive": n_train_valid,
                "sharpe_sdf_monthly": cv_table[(best_i, best_j)]["valid_SR_sdf"],
                "sharpe_trade_monthly": cv_table[(best_i, best_j)]["valid_SR_trade"],
            },
            {
                "sample": "full_train_valid",
                "start_row": 0,
                "end_row_exclusive": n_train_valid,
                "sharpe_sdf_monthly": full_first["train_SR_sdf"],
                "sharpe_trade_monthly": full_first["train_SR_trade"],
            },
            {
                "sample": "test",
                "start_row": n_train_valid,
                "end_row_exclusive": len(df),
                "sharpe_sdf_monthly": _safe_sr(gross_sdf),
                "sharpe_trade_monthly": _safe_sr(gross_trade),
            },
        ]
    )

    selection = APPruningSelection(
        lambda0_index=best_i,
        lambda2_index=best_j,
        lambda0=lambda0_best,
        lambda2=lambda2_best,
        portsN=port_n,
        cv_valid_sr_sdf=float(cv_table[(best_i, best_j)]["valid_SR_sdf"]),
        full_train_sr_sdf=float(full_first["train_SR_sdf"]),
        full_test_sr_sdf=float(_safe_sr(gross_sdf)),
        full_train_sr_trade=float(full_first["train_SR_trade"]),
        full_test_sr_trade=float(_safe_sr(gross_trade)),
    )

    return result.reset_index(drop=True), weights, diag, selection


# ============================================================
# TC-aware rolling ablation / extension
# ============================================================

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
    Ablation objective:
        min_w 0.5 w'Σw - eta μ'w + 0.5 λ2 ||w||_2^2 + λtc * turnover

    Returned weights are gross-exposure-normalized for stable investable backtests.
    """
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
            - eta * (w @ mu)
            + 0.5 * lambda_l2 * np.sum(w * w)
            + lambda_tc * tc_penalty(w)
        )

    # For the extension we constrain net budget. Gross exposure is normalized after solve.
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
        return w_prev.copy()

    w = np.asarray(res.x, dtype=float)
    w[np.abs(w) < 1e-12] = 0.0
    return _normalize_gross_exposure(w)


def rolling_tc_optimize(
    returns_df: pd.DataFrame,
    window: int,
    lambda_l2: float,
    lambda_tc: float,
    eta: float,
    cost_per_turnover: float,
    method_name: str,
    turnover_mode: str = "portfolio",
    stock_weights=None,
    selected_candidates: list[str] | None = None,
    long_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling transaction-cost-aware ablation.

    Uses gross-exposure-normalized weights so the cumulative wealth plots are
    interpretable as investable backtests.
    """
    if turnover_mode not in {"portfolio", "stock"}:
        raise ValueError("turnover_mode must be 'portfolio' or 'stock'.")

    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)

    if "date_dt" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date_dt"])

    all_cols = _candidate_cols(df)

    if selected_candidates is None:
        cols = all_cols
    else:
        cols = [c for c in selected_candidates if c in all_cols]
        if not cols:
            raise ValueError("No selected candidates exist in returns_df.")

    x = df[cols].astype(float).fillna(0.0).reset_index(drop=True)
    meta_cols = _meta_cols(df)

    k = len(cols)
    w_prev = np.ones(k) / k
    prev_stock_w_series: pd.Series | None = None

    sw = _prep_stock_weights(stock_weights)

    rows = []
    w_rows = []

    for t in range(window, len(df)):
        hist = x.iloc[t - window:t]
        mu = hist.mean(axis=0).to_numpy()
        sigma = np.cov(hist.to_numpy(), rowvar=False, ddof=1)

        meta = df.iloc[t][meta_cols].to_dict()

        stock_matrix = None
        prev_stock_vec = None
        permnos = pd.Index([])

        if turnover_mode == "stock" and sw is not None:
            stock_matrix, permnos = _stock_weight_matrix_for_month(
                stock_weights=sw,
                meta=meta,
                cols=cols,
                stock_weight_col="tilt_stock_w",
            )

            if len(permnos) > 0:
                if prev_stock_w_series is None:
                    prev_stock_vec = np.zeros(len(permnos))
                else:
                    prev_stock_vec = prev_stock_w_series.reindex(permnos, fill_value=0.0).to_numpy()

        w = solve_tc_mean_variance_qp(
            mu=mu,
            sigma=sigma,
            w_prev=w_prev,
            eta=eta,
            lambda_l2=lambda_l2,
            lambda_tc=lambda_tc,
            turnover_mode=turnover_mode,
            long_only=long_only,
            stock_matrix=stock_matrix,
            prev_stock_vec=prev_stock_vec,
        )

        gross = float(x.iloc[t].to_numpy() @ w)
        raw_turnover = float(np.sum(np.abs(w - w_prev)))
        candidate_w = pd.Series(w, index=cols)

        if turnover_mode == "stock" and sw is not None:
            curr_stock_w = _month_stock_weights(
                stock_weights=sw,
                meta=meta,
                candidate_weights=candidate_w,
                stock_weight_col="tilt_stock_w",
            )
            turnover = _stock_turnover(curr_stock_w, prev_stock_w_series)
            prev_stock_w_series = curr_stock_w
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
                w_rows.append({**meta, "candidate": c, "weight_trade": wi})

        w_prev = w

    return pd.DataFrame(rows), pd.DataFrame(w_rows)
