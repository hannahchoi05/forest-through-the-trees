from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import ntile_r


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _safe_sr(xr: np.ndarray) -> float:
    xr = np.asarray(xr, dtype=float)
    xr = xr[np.isfinite(xr)]
    if len(xr) < 2:
        return np.nan
    sd = xr.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    return float(xr.mean() / sd)


# ---------------------------------------------------------------------------
# LARS (Least Angle Regression) — paper Appendix A.4
# ---------------------------------------------------------------------------
# Implements the augmented LARS path used in AP-Pruning:
#   augmented design:  X_aug = [X; sqrt(lambda_l2) * I]
#                      y_aug = [y; 0]
# This folds the ridge penalty into the LARS solve so a single pass gives
# the full elastic-net-like path at every sparsity level K simultaneously.
# ---------------------------------------------------------------------------

def _lars_path(
    X: np.ndarray,
    y: np.ndarray,
    lambda_l2: float = 0.0,
    max_iter: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Compute the LARS solution path with optional ridge augmentation.

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,)
    lambda_l2 : ridge penalty; augments X before LARS
    max_iter : cap on LARS steps (defaults to n_features)

    Returns
    -------
    alphas : (n_steps,) decreasing regularisation values at each step
    coef_path : list of (n_features,) coefficient vectors, one per step
                coef_path[k] has exactly k+1 nonzero entries (approximately)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    # Ridge augmentation: stack sqrt(lambda_l2)*I rows and zero response rows.
    if lambda_l2 > 0.0:
        aug = np.sqrt(lambda_l2) * np.eye(p)
        X = np.vstack([X, aug])
        y = np.concatenate([y, np.zeros(p)])

    if max_iter is None:
        max_iter = p

    active: list[int] = []
    betas = np.zeros(p)
    residual = y.copy()

    alphas: list[float] = []
    coef_path: list[np.ndarray] = []

    for _ in range(max_iter):
        # Correlations of all predictors with current residual.
        corr = X.T @ residual  # (p,)

        # Most correlated inactive predictor.
        inactive = [j for j in range(p) if j not in active]
        if not inactive:
            break

        abs_corr_inactive = np.abs(corr[inactive])
        best_j = inactive[int(np.argmax(abs_corr_inactive))]
        C = float(np.abs(corr[best_j]))

        if C < 1e-14:
            break  # numerical zero — stop

        active.append(best_j)
        alphas.append(C)

        # Solve OLS on active set to get equiangular direction.
        X_act = X[:, active]
        try:
            # Use pseudo-inverse for numerical safety when active set is near singular.
            betas_act, _, _, _ = np.linalg.lstsq(X_act, y, rcond=None)
        except np.linalg.LinAlgError:
            break

        betas_full = np.zeros(p)
        for idx, j in enumerate(active):
            betas_full[j] = betas_act[idx]

        coef_path.append(betas_full.copy())
        betas = betas_full
        residual = y - X @ betas

    return np.array(alphas), coef_path


# ---------------------------------------------------------------------------
# Eigendecomposition helper — paper Appendix A.4
# Matches R eigen() which returns eigenvalues in DESCENDING order.
# ---------------------------------------------------------------------------

def _eig_decomp(sigma: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetric eigendecomposition with truncation of near-zero eigenvalues.

    Returns V, D (kept eigenvalues) in descending order, matching R's eigen().
    """
    eigvals, eigvecs = np.linalg.eigh(sigma)          # ascending order from eigh
    order = np.argsort(eigvals)[::-1]                  # FIX: descending to match R
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    keep = eigvals > tol
    return eigvecs[:, keep], eigvals[keep]


# ---------------------------------------------------------------------------
# AP-Pruning static optimizer — paper Definition 1 / Proposition 1
# ---------------------------------------------------------------------------
# Objective (Definition 1):
#   min_w  0.5 w'Σ̂w + λ1||w||1 + 0.5 λ2||w||2^2
#   s.t.   w'μ̂ ≥ μ0,   w'1 = 1
#
# Proposition 1 (mean shrinkage):
#   Tracing frontier by varying μ0 ≡ shrinking μ̂ toward cross-sectional mean:
#   μ̃ = μ̂ + λ0 * 1   (one-to-one mapping μ0 ↔ λ0)
#
# Implementation via LARS on augmented system (Appendix A.4):
#   X_aug = [Σ̃^{1/2}; sqrt(λ2)*I],  y_aug = [μ̃_tilde; 0]
#   where Σ̃ = V D^{1/2} V',  μ̃_tilde = V D^{-1/2} V' (μ̂ + λ0 * μ̄ * 1)
# ---------------------------------------------------------------------------

def _ap_prune_lars(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_l2: float,
    lambda0: float,
    k_target: int,
    k_min: int = 10,
    k_max: int = 40,
) -> np.ndarray:
    """
    Run LARS-based AP-Pruning for one (lambda_l2, lambda0) configuration.

    Returns beta (LARS coefficients) at the step where sparsity == k_target,
    or the closest step within [k_min, k_max].

    Notes
    -----
    - lambda0 controls mean shrinkage toward cross-sectional average (Prop. 1).
    - lambda_l2 folds into ridge augmentation of the LARS design matrix.
    - lambda_l1 (LASSO) is implicit: determined by which LARS step we pick.
    - Triple Sort has no depth adjustment, so adj_w = 1 throughout.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    p = len(mu)

    # --- Mean shrinkage (Proposition 1) ---
    mu_bar = float(mu.mean())
    mu_shrunk = mu + lambda0 * mu_bar * np.ones(p)

    # --- Eigendecomposition (Appendix A.4) ---
    V, D = _eig_decomp(sigma)           # V: (p, r), D: (r,) descending
    D_sqrt = np.sqrt(D)                  # D^{1/2}
    D_inv_sqrt = 1.0 / D_sqrt           # D^{-1/2}

    # Σ̃ = V D^{1/2} V'  — used as the design matrix X in LARS
    Sigma_tilde = V * D_sqrt[np.newaxis, :]   # (p, r)  = V @ diag(D^{1/2})

    # μ̃_tilde = V D^{-1/2} V' μ_shrunk
    mu_tilde = V @ (D_inv_sqrt * (V.T @ mu_shrunk))   # (p,)

    # --- LARS with ridge augmentation ---
    # X_aug = [Sigma_tilde; sqrt(lambda_l2)*I_r],  y_aug = [mu_tilde; 0]
    # (ridge augmentation is handled inside _lars_path)
    _, coef_path = _lars_path(
        X=Sigma_tilde.T,   # LARS convention: (n_samples=r, n_features=p)
        y=mu_tilde,
        lambda_l2=lambda_l2,
        max_iter=min(p, k_max + 5),
    )

    if not coef_path:
        return np.zeros(p)

    # --- Walk path to find step closest to k_target within [k_min, k_max] ---
    best_beta = coef_path[-1]
    best_dist = float("inf")

    for beta in coef_path:
        k_nonzero = int(np.sum(np.abs(beta) > 1e-10))
        if k_nonzero < k_min or k_nonzero > k_max:
            continue
        dist = abs(k_nonzero - k_target)
        if dist < best_dist:
            best_dist = dist
            best_beta = beta.copy()
        if dist == 0:
            break  # exact match found

    return best_beta


def _normalize_b(b: np.ndarray) -> np.ndarray:
    """
    Normalize LARS beta to unit signed sum — matches R's b = b / abs(sum(b)).

    This is NOT the same as unit gross exposure (sum(|b|) = 1).
    R normalizes by abs(sum(b)), which preserves the sign structure of the
    long-short portfolio while fixing the scale.
    """
    s = float(np.sum(b))
    denom = abs(s)
    if denom < 1e-12:
        return np.zeros_like(b)
    return b / denom


def ap_pruning_static_optimize(
    returns_df: pd.DataFrame,
    n_train_valid: int,
    cv_n: int = 3,
    mu0_grid: list[float] | None = None,
    lambda_l2_grid: list[float] | None = None,
    k_target: int = 40,
    k_min: int = 10,
    k_max: int = 40,
    panel: pd.DataFrame | None = None,
    n_bins: tuple[int, int, int] = (2, 4, 4),
    method_name: str = "Triple Sort static (no TC)",
    cost_per_turnover: float = 0.0,
    use_stock_level_turnover: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Paper-faithful AP-Pruning optimizer matching R's lasso_valid_par_full.R
    and Pick_Best_Lambda.R exactly.

    R implementation details matched
    ---------------------------------
    1. LARS run on training window (240 months) for CV/validation Sharpe.
    2. Normalization: b = b / abs(sum(b))  — NOT sum(|b|). Mixed-sign
       portfolios need signed-sum normalization to preserve direction.
    3. Validation SR computed on normalized b (after normalization step).
    4. Best (lambda0, lambda2) selected by maximizing valid SR at fixed k_target.
    5. Final weights come from a SECOND LARS fit on the full train+valid window
       (360 months) using the best hyperparameters — matches R's 'full' fit.
    6. Triple Sort: adj_w = 1 throughout, so b * adj_w = b and b / adj_w = b.

    Two-pass structure (matching R)
    --------------------------------
    Pass 1 (CV pass): LARS on x_train (240m) → find best (lam0, lam2) by
                      validation SR on x_valid (120m).
    Pass 2 (full pass): LARS on x_train_valid (360m) with best params →
                        final weights used for test backtest.
    """
    if mu0_grid is None:
        mu0_grid = list(np.round(np.arange(0.0, 0.95, 0.05), 10))   # 19 values
    if lambda_l2_grid is None:
        lambda_l2_grid = list(np.logspace(-5.0, -8.0, 13))            # 13 values

    df = returns_df.copy().sort_values(["yy", "mm"]).reset_index(drop=True)
    cols = _candidate_cols(df)
    p = len(cols)

    if n_train_valid >= len(df):
        raise ValueError("n_train_valid must be smaller than total number of months.")

    x = df[cols].astype(float).fillna(0.0).to_numpy()   # (T, p)

    # --- Data splits ---
    # Matching R's cv fold i=cvN=3 (last fold only):
    #   ports_train = rows NOT in [(i-1)*n_valid+1 : i*n_valid] and NOT test
    #   ports_valid = rows [(i-1)*n_valid+1 : i*n_valid]
    # For cvN=3, n_valid=120: train = rows 0..239, valid = rows 240..359
    n_valid = n_train_valid // cv_n          # 120 months
    n_train = n_train_valid - n_valid        # 240 months

    x_train       = x[:n_train]             # 1964–1983 (240m) — CV training
    x_valid       = x[n_train:n_train_valid] # 1984–1993 (120m) — CV validation
    x_train_valid = x[:n_train_valid]       # 1964–1993 (360m) — full fit
    x_test        = x[n_train_valid:]       # 1994–2016 (276m) — test

    # --- Pass 1: fit on training window, select best hyperparameters ---
    mu_train  = x_train.mean(axis=0)
    sig_train = np.cov(x_train, rowvar=False, ddof=1)

    best_sr     = -np.inf
    best_lam0   = mu0_grid[0]
    best_lam2   = lambda_l2_grid[0]

    for lam2 in lambda_l2_grid:
        for lam0 in mu0_grid:
            beta = _ap_prune_lars(
                mu=mu_train,
                sigma=sig_train,
                lambda_l2=lam2,
                lambda0=lam0,
                k_target=k_target,
                k_min=k_min,
                k_max=k_max,
            )
            if np.all(beta == 0.0):
                continue

            # Normalize — matches R: b = b / abs(sum(b))
            b = _normalize_b(beta)
            if np.all(b == 0.0):
                continue

            # Validation SR on normalized b — matches R: sdf_valid = ports_valid %*% b
            # (adj_w = 1 for Triple Sort so b / adj_w = b)
            val_ret = x_valid @ b
            sr = _safe_sr(val_ret)
            if np.isfinite(sr) and sr > best_sr:
                best_sr   = sr
                best_lam0 = lam0
                best_lam2 = lam2

    best_params = {"lambda_l2": best_lam2, "lambda0": best_lam0}

    # --- Pass 2: refit on full train+valid window with best hyperparameters ---
    # Matches R's lasso_cv_helper called with ports_train = ports[1:n_train_valid,]
    mu_full  = x_train_valid.mean(axis=0)
    sig_full = np.cov(x_train_valid, rowvar=False, ddof=1)

    beta_full = _ap_prune_lars(
        mu=mu_full,
        sigma=sig_full,
        lambda_l2=best_lam2,
        lambda0=best_lam0,
        k_target=k_target,
        k_min=k_min,
        k_max=k_max,
    )

    if np.all(beta_full == 0.0):
        # Fallback: equal weights
        beta_full = np.ones(p) / p
        best_params = {"lambda_l2": np.nan, "lambda0": np.nan}

    # Normalize final LARS coefficients in TWO ways:
    #   1. w_sdf   = paper/SDF convention: b / abs(sum(b))
    #   2. w_trade = tradable convention: b / sum(abs(b))
    #
    # For Triple Sort, adj_w = 1, so the clean conversion is simply
    #     b_tilde = beta_full
    #     w_trade = b_tilde / sum(abs(b_tilde))
    #
    # Use w_trade for backtest returns, stock-level turnover, saved trading
    # weights, and summary metrics. Keep w_sdf only for diagnostics against
    # the paper's signed-sum/SDF convention.
    w_sdf = _normalize_b(beta_full)
    w_trade = _normalize_gross_exposure(beta_full)

    if np.all(w_trade == 0.0):
        w_trade = np.ones(p) / p
    if np.all(w_sdf == 0.0):
        w_sdf = w_trade.copy()

    # --- Backtest on test period ---
    meta_cols = _meta_cols(df)
    result = df.iloc[n_train_valid:][meta_cols].copy().reset_index(drop=True)
    result["method"] = method_name

    gross = x_test @ w_trade
    result["gross_ret"] = gross

    # --- Turnover and costs ---
    turnovers: list[float] = []
    costs:     list[float] = []
    prev_stock_w: pd.Series | None = None

    if use_stock_level_turnover:
        if panel is None:
            raise ValueError("panel is required when use_stock_level_turnover=True")
        panel_g = panel.groupby(["yy", "mm"], sort=False)

    for _, row in result.iterrows():
        if use_stock_level_turnover:
            m = panel_g.get_group((int(row["yy"]), int(row["mm"])))
            curr_stock_w = _final_stock_weights_for_month(
                m, candidate_weights=w_trade, n_bins=n_bins,
            )
            turnover = _stock_turnover(curr_stock_w, prev_stock_w)
            prev_stock_w = curr_stock_w
        else:
            turnover = 0.0
        turnovers.append(turnover)
        costs.append(float(cost_per_turnover) * turnover)

    result["turnover_raw"] = turnovers
    result["turnover"]     = turnovers
    result["cost"]         = costs
    result["net_ret"]      = result["gross_ret"] - result["cost"]

    # --- Weights output (nonzero only) ---
    # Store both conventions so it is explicit in the CSV. The main "weight"
    # column is the tradable gross-normalized weight used by the backtest.
    weights = pd.DataFrame({
        "candidate": cols,
        "weight": w_trade,
        "weight_trade": w_trade,
        "weight_sdf": w_sdf,
    })
    weights = (
        weights[weights["weight_trade"].abs() > 1e-12]
        .sort_values("weight_trade", ascending=False)
        .reset_index(drop=True)
    )

    # --- Diagnostics ---
    cv_train_ret_sdf       = x_train       @ w_sdf
    cv_valid_ret_sdf       = x_valid       @ w_sdf
    full_train_val_ret_sdf = x_train_valid @ w_sdf
    test_ret_sdf           = x_test        @ w_sdf

    cv_train_ret_trade       = x_train       @ w_trade
    cv_valid_ret_trade       = x_valid       @ w_trade
    full_train_val_ret_trade = x_train_valid @ w_trade
    test_ret_trade           = x_test        @ w_trade

    diag = pd.DataFrame({
        "sample":            ["cv_train", "cv_valid", "full_train_valid", "test"],
        "start_row":         [0, n_train, 0, n_train_valid],
        "end_row_exclusive": [n_train, n_train_valid, n_train_valid, len(df)],
        "sharpe_sdf_monthly": [
            _safe_sr(cv_train_ret_sdf),
            _safe_sr(cv_valid_ret_sdf),
            _safe_sr(full_train_val_ret_sdf),
            _safe_sr(test_ret_sdf),
        ],
        "sharpe_trade_monthly": [
            _safe_sr(cv_train_ret_trade),
            _safe_sr(cv_valid_ret_trade),
            _safe_sr(full_train_val_ret_trade),
            _safe_sr(test_ret_trade),
        ],
        "best_lambda_l2": [best_params.get("lambda_l2", np.nan)] * 4,
        "best_lambda0":   [best_params.get("lambda0",   np.nan)] * 4,
        "k_nonzero":      [int(np.sum(np.abs(w_trade) > 1e-10))] * 4,
        "gross_exposure_trade": [float(np.sum(np.abs(w_trade)))] * 4,
        "signed_sum_trade": [float(np.sum(w_trade))] * 4,
        "signed_sum_sdf": [float(np.sum(w_sdf))] * 4,
        "best_valid_sr":  [best_sr] * 4,
    })

    return result.reset_index(drop=True), weights, diag


# ---------------------------------------------------------------------------
# Legacy plain-QP solver (kept for rolling TC variants which don't use LARS)
# ---------------------------------------------------------------------------

def solve_portfolio_qp(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray | None = None,
    lambda_l1: float = 0.0,
    lambda_l2: float = 1e-6,
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

    if long_only:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(k)]
        x0 = np.clip(w_prev, 0.0, 1.0)

        if abs(x0.sum()) < 1e-12:
            x0 = np.ones(k) / k
        else:
            x0 = x0 / x0.sum()
    else:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]
        bounds = [(-1.0, 1.0) for _ in range(k)]
        x0 = _normalize_gross_exposure(w_prev)

        if np.sum(np.abs(x0)) < 1e-12:
            x0 = np.ones(k) / k

    if mu0 is not None:
        constraints.append({"type": "ineq", "fun": lambda w: float(w @ mu - mu0)})

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

    if long_only:
        s = np.sum(w)
        if abs(s) < 1e-12:
            return np.ones(k) / k
        return w / s

    return _normalize_gross_exposure(w)


# ---------------------------------------------------------------------------
# Rolling TC-aware optimizer (unchanged — uses QP, not LARS)
# ---------------------------------------------------------------------------

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

    If long_only=True:
        sum(w) = 1, w >= 0

    If long_only=False:
        sum(abs(w)) = 1, -1 <= w_i <= 1
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

    if long_only:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(k)]
        x0 = np.clip(w_prev, 0.0, 1.0)

        if abs(x0.sum()) <= 1e-12:
            x0 = np.ones(k) / k
        else:
            x0 = x0 / x0.sum()
    else:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]
        bounds = [(-1.0, 1.0) for _ in range(k)]
        x0 = _normalize_gross_exposure(w_prev)

        if np.sum(np.abs(x0)) <= 1e-12:
            x0 = np.ones(k) / k

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

    if long_only:
        s = np.sum(w)
        if abs(s) <= 1e-12:
            return np.ones(k) / k
        return w / s

    return _normalize_gross_exposure(w)

# ---------------------------------------------------------------------------
# Stock-level weight helpers (unchanged)
# ---------------------------------------------------------------------------

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
    df[size_col]   = df[size_col].astype(float)

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
    return out[out.abs() > 1e-14]


def _stock_weight_matrix_for_month(
    month_panel: pd.DataFrame,
    n_bins: tuple[int, int, int],
    feat_cols: tuple[str, str, str] = ("LME", "OP", "Investment"),
    size_col: str = "size",
    permno_col: str = "permno",
) -> tuple[np.ndarray, pd.Index]:
    df = month_panel[[permno_col, size_col, *feat_cols]].copy()
    df[permno_col] = df[permno_col].astype(str)
    df[size_col]   = df[size_col].astype(float)

    df = df[df[size_col].notna() & (df[size_col] > 0)].copy()
    n_ports = n_bins[0] * n_bins[1] * n_bins[2]
    if df.empty:
        return np.zeros((0, n_ports)), pd.Index([])

    b1 = ntile_r(df[feat_cols[0]].astype(float), n_bins[0])
    b2 = ntile_r(df[feat_cols[1]].astype(float), n_bins[1])
    b3 = ntile_r(df[feat_cols[2]].astype(float), n_bins[2])

    ok = b1.notna() & b2.notna() & b3.notna()
    df = df.loc[ok].copy()
    if df.empty:
        return np.zeros((0, n_ports)), pd.Index([])

    b1 = b1.loc[df.index].astype(int)
    b2 = b2.loc[df.index].astype(int)
    b3 = b3.loc[df.index].astype(int)

    n2, n3 = n_bins[1], n_bins[2]
    bucket_id = (b1 - 1) * (n2 * n3) + (b2 - 1) * n3 + (b3 - 1) + 1
    df["bucket_id"] = bucket_id.astype(int)

    sum_size = df.groupby("bucket_id")[size_col].transform("sum")
    df["base_w"] = df[size_col] / sum_size

    permnos = pd.Index(sorted(df[permno_col].astype(str).unique()))
    perm_to_row = {p: i for i, p in enumerate(permnos)}
    M = np.zeros((len(permnos), n_ports), dtype=float)

    cols_idx = (df["bucket_id"].to_numpy(dtype=int) - 1).clip(0, n_ports - 1)
    for permno, col, base_w in zip(
        df[permno_col].astype(str).to_numpy(),
        cols_idx,
        df["base_w"].to_numpy(dtype=float),
    ):
        M[perm_to_row[permno], int(col)] += float(base_w)

    return M, permnos


def _stock_turnover(curr: pd.Series, prev: pd.Series | None) -> float:
    if prev is None or prev.empty:
        return float(curr.abs().sum())
    idx = curr.index.union(prev.index)
    curr_aligned = curr.reindex(idx, fill_value=0.0)
    prev_aligned = prev.reindex(idx, fill_value=0.0)
    return float((curr_aligned - prev_aligned).abs().sum())


# ---------------------------------------------------------------------------
# Public entry points — wired to run_all.py
# ---------------------------------------------------------------------------

def static_paper_style_optimize(
    returns_df: pd.DataFrame,
    n_train_valid: int,
    panel: pd.DataFrame | None = None,
    n_bins: tuple[int, int, int] = (2, 4, 4),
    cv_n: int = 3,
    lambda_l2: float = 1e-6,           # ignored — grid searched internally
    mu0: float | None = None,          # ignored — grid searched internally
    long_only: bool = True,            # unused for LARS (no sign constraint in LARS)
    method_name: str = "Triple Sort static (no TC)",
    cost_per_turnover: float = 0.0,
    use_stock_level_turnover: bool = False,
    # New grid-search params (passed through from run_all.py or use defaults)
    mu0_grid: list[float] | None = None,
    lambda_l2_grid: list[float] | None = None,
    k_target: int = 40,
    k_min: int = 10,
    k_max: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Public wrapper — now delegates to LARS-based ap_pruning_static_optimize.

    The lambda_l2 and mu0 scalar arguments from the old interface are ignored;
    the grid is searched internally. Pass mu0_grid / lambda_l2_grid explicitly
    to override the paper's default grids.
    """
    return ap_pruning_static_optimize(
        returns_df=returns_df,
        n_train_valid=n_train_valid,
        cv_n=cv_n,
        mu0_grid=mu0_grid,
        lambda_l2_grid=lambda_l2_grid,
        k_target=k_target,
        k_min=k_min,
        k_max=k_max,
        panel=panel,
        n_bins=n_bins,
        method_name=method_name,
        cost_per_turnover=cost_per_turnover,
        use_stock_level_turnover=use_stock_level_turnover,
    )


def rolling_tc_optimize(
    returns_df: pd.DataFrame,
    window: int,
    panel: pd.DataFrame | None = None,
    n_bins: tuple[int, int, int] = (2, 4, 4),
    lambda_l2: float = 1e-6,
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

    rows:   list[dict] = []
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

        mu    = hist.mean(axis=0).to_numpy()
        sigma = np.cov(hist.to_numpy(), rowvar=False, ddof=1)

        meta = df.iloc[t][meta_cols].to_dict()

        stock_matrix   = None
        prev_stock_vec = None
        permnos        = pd.Index([])

        if turnover_mode == "stock":
            m = panel_g.get_group((int(meta["yy"]), int(meta["mm"])))
            stock_matrix, permnos = _stock_weight_matrix_for_month(m, n_bins=n_bins)
            if len(permnos) > 0:
                prev_stock_vec = (
                    np.zeros(len(permnos)) if prev_stock_w is None
                    else prev_stock_w.reindex(permnos, fill_value=0.0).to_numpy()
                )

        w = solve_tc_mean_variance_qp(
            mu=mu, sigma=sigma, w_prev=w_prev,
            lambda_tc=lambda_tc, lambda_l2=lambda_l2, eta=eta,
            long_only=long_only, turnover_mode=turnover_mode,
            stock_matrix=stock_matrix, prev_stock_vec=prev_stock_vec,
        )

        gross       = float(x.iloc[t].to_numpy() @ w)
        raw_turnover = float(np.sum(np.abs(w - w_prev)))

        if turnover_mode == "stock":
            curr_stock_w = _final_stock_weights_for_month(
            panel_g.get_group((int(meta["yy"]), int(meta["mm"]))),
            candidate_weights=w, n_bins=n_bins,
            )
            turnover     = _stock_turnover(curr_stock_w, prev_stock_w)
            prev_stock_w = curr_stock_w
        else:
            turnover = raw_turnover

        cost = float(cost_per_turnover) * turnover
        net  = gross - cost

        rows.append({
            **meta,
            "method":       method_name,
            "gross_ret":    gross,
            "turnover_raw": raw_turnover,
            "turnover":     turnover,
            "cost":         cost,
            "net_ret":      net,
        })

        for c, wi in zip(cols, w):
            if abs(wi) > 1e-12:
                w_rows.append({**meta, "candidate": c, "weight_trade": float(wi)})

        w_prev = w

    return pd.DataFrame(rows), pd.DataFrame(w_rows)