from __future__ import annotations

import numpy as np
import pandas as pd


def month_end_from_yy_mm(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        df["yy"].astype(int).astype(str)
        + "-"
        + df["mm"].astype(int).astype(str)
        + "-01"
    ) + pd.offsets.MonthEnd(0)


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


def safe_method_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace("+", "plus")


def ntile_r(x: pd.Series, n: int) -> pd.Series:
    """
    Mimic dplyr::ntile behavior used in the original triple-sort generator.
    Returns integer buckets 1..n (nullable Int64).
    """
    x = pd.Series(x)
    out = pd.Series(pd.NA, index=x.index, dtype="Int64")
    valid = x.notna()
    if int(valid.sum()) == 0:
        return out

    # Match the generator's rank(method="first") behavior on valid entries.
    ranks0 = x.loc[valid].rank(method="first").astype(int) - 1
    size = len(ranks0)
    base_size = size // n
    remainder = size % n

    buckets0 = np.where(
        ranks0 < remainder * (base_size + 1),
        ranks0 // (base_size + 1),
        remainder + (ranks0 - remainder * (base_size + 1)) // base_size,
    )

    out.loc[valid] = (buckets0 + 1).astype(int)
    return out

