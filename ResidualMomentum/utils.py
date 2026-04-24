from __future__ import annotations
import numpy as np
import pandas as pd


def ntile(x: pd.Series, q: int) -> pd.Series:
    """Approximate dplyr::ntile: integer buckets 1,...,q with ties broken by order."""
    x = pd.Series(x)
    out = pd.Series(np.nan, index=x.index)
    valid = x.notna()
    n = int(valid.sum())
    if n == 0:
        return out.astype("Int64")
    ranks = x.loc[valid].rank(method="first")
    buckets = np.ceil(ranks * q / n).astype(int).clip(1, q)
    out.loc[valid] = buckets
    return out.astype("Int64")


def zscore_cross_section(x: pd.Series, clip: float | None = 5.0) -> pd.Series:
    x = pd.Series(x, index=x.index).astype(float)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd < 1e-12:
        z = pd.Series(0.0, index=x.index)
    else:
        z = (x - x.mean(skipna=True)) / sd
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


def cumulative_wealth(r: pd.Series) -> pd.Series:
    r = pd.Series(r).astype(float).fillna(0.0)
    return (1.0 + r).cumprod()


def drawdown_from_returns(r: pd.Series) -> pd.Series:
    w = cumulative_wealth(r)
    peak = w.cummax()
    return w / peak - 1.0


def annualized_sharpe(monthly_returns: pd.Series) -> float:
    r = pd.Series(monthly_returns).astype(float).dropna()
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    return float(np.sqrt(12.0) * r.mean() / sd)


def month_end_from_yy_mm(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        df["yy"].astype(int).astype(str) + "-" + df["mm"].astype(int).astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)


def safe_method_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace("+", "plus")
