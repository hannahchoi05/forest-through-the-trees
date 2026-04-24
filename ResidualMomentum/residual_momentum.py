from __future__ import annotations
import numpy as np
import pandas as pd


def add_raw_momentum_signal(
    panel: pd.DataFrame,
    lookback: int = 12,
    skip_recent: int = 1,
    ret_col: str = "ret",
    permno_col: str = "permno",
    signal_col: str = "residual_mom",
) -> pd.DataFrame:
    """Fallback signal: 12-to-2 cumulative raw momentum."""
    df = panel.copy().sort_values([permno_col, "yy", "mm"]).reset_index(drop=True)
    gross = 1.0 + df[ret_col].astype(float).fillna(0.0)
    shifted = gross.groupby(df[permno_col]).shift(skip_recent)
    mom = (
        shifted.groupby(df[permno_col])
        .rolling(lookback, min_periods=lookback)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )
    df[signal_col] = mom
    return df


def add_market_residual_momentum_signal(
    panel: pd.DataFrame,
    market_returns: pd.Series,
    lookback: int = 12,
    skip_recent: int = 1,
    beta_window: int = 36,
    ret_col: str = "ret",
    permno_col: str = "permno",
    signal_col: str = "residual_mom",
) -> pd.DataFrame:
    """
    True residual momentum: rolling market-model residuals accumulated from t-12 to t-2.

    For each stock, estimate alpha/beta using prior beta_window observations only, compute the
    current residual, then compound historical residuals with a one-month skip.
    """
    df = panel.copy().sort_values([permno_col, "yy", "mm"]).reset_index(drop=True)
    months = df[["yy", "mm"]].drop_duplicates().sort_values(["yy", "mm"]).reset_index(drop=True)
    months["month_id"] = np.arange(len(months))
    df = df.merge(months, on=["yy", "mm"], how="left")

    mkt = pd.Series(market_returns).astype(float).reset_index(drop=True)
    df["mkt_ret"] = df["month_id"].map(dict(enumerate(mkt.iloc[: len(months)])))
    eps_all = pd.Series(np.nan, index=df.index, dtype=float)

    for _, g in df.groupby(permno_col, sort=False):
        idx = g.index.to_numpy()
        r = g[ret_col].astype(float).to_numpy()
        m = g["mkt_ret"].astype(float).to_numpy()
        eps = np.full(len(g), np.nan)
        for j in range(len(g)):
            start = max(0, j - beta_window)
            rr = r[start:j]
            mm = m[start:j]
            ok = np.isfinite(rr) & np.isfinite(mm)
            if ok.sum() < max(12, beta_window // 2):
                continue
            X = np.column_stack([np.ones(ok.sum()), mm[ok]])
            try:
                a, b = np.linalg.lstsq(X, rr[ok], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(r[j]) and np.isfinite(m[j]):
                eps[j] = r[j] - (a + b * m[j])
        eps_all.loc[idx] = eps

    df["residual_ret"] = eps_all
    gross_resid = 1.0 + df["residual_ret"].fillna(0.0)
    shifted = gross_resid.groupby(df[permno_col]).shift(skip_recent)
    mom = (
        shifted.groupby(df[permno_col])
        .rolling(lookback, min_periods=lookback)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )
    df[signal_col] = mom
    return df.drop(columns=["month_id", "mkt_ret"])
