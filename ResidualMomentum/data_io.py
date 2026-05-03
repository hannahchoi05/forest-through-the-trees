from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf

from utils import month_end_from_yy_mm


def load_yearly_chunks(
    chunk_dir: Path,
    chars: list[str],
    y_min: int,
    y_max: int,
) -> pd.DataFrame:
    """Load Data/data_chunk_files_quantile/<char1>_<char2>_<char3>/yYYYY.csv."""
    subdir = "_".join(chars)
    path = chunk_dir / subdir

    dfs = []

    for year in range(y_min, y_max + 1):
        f = path / f"y{year}.csv"

        if not f.exists():
            raise FileNotFoundError(f"Missing yearly chunk: {f}")

        dfs.append(pd.read_csv(f))

    out = pd.concat(dfs, ignore_index=True)

    out["permno"] = out["permno"].astype(str)
    out["yy"] = out["yy"].astype(int)
    out["mm"] = out["mm"].astype(int)
    out["date_dt"] = month_end_from_yy_mm(out)

    return out.sort_values(["yy", "mm", "permno"]).reset_index(drop=True)


def load_market_proxy(factor_dir: Path) -> pd.Series | None:
    """
    Try to load a monthly market return proxy from Data/factor/tradable_factors.csv.
    Returns decimal monthly returns if possible.
    """
    f = factor_dir / "tradable_factors.csv"

    if not f.exists():
        return None

    fac = pd.read_csv(f)

    possible = ["market", "Mkt.RF", "Mkt_RF", "mkt", "MKT"]
    col = next((c for c in possible if c in fac.columns), None)

    if col is None:
        numeric_cols = fac.select_dtypes("number").columns.tolist()

        if len(numeric_cols) == 0:
            return None

        col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

    s = fac[col].astype(float).reset_index(drop=True)

    if s.abs().median() > 0.2:
        s = s / 100.0

    return s


def load_yahoo_monthly_benchmark(
    ticker: str,
    start_date,
    end_date,
    method_name: str = "S&P 500 (SPY adjusted close)",
) -> pd.DataFrame:
    """
    Load a monthly benchmark from Yahoo Finance using adjusted close.

    For S&P 500 exposure, we use SPY adjusted close as a total-return proxy.
    Output matches the backtest schema:
        date, date_dt, yy, mm, method, gross_ret, turnover, cost, net_ret
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Pull a little earlier so first aligned monthly return can be computed.
    download_start = start_date - pd.DateOffset(months=2)
    download_end = end_date + pd.DateOffset(days=5)

    px = yf.download(
        ticker,
        start=download_start.strftime("%Y-%m-%d"),
        end=download_end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if px.empty:
        raise ValueError(f"Yahoo Finance returned no data for ticker={ticker}")

    # yfinance can return MultiIndex columns depending on version.
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = [c[0] for c in px.columns]

    price_col = "Adj Close" if "Adj Close" in px.columns else "Close"

    monthly_price = px[price_col].resample("ME").last()
    monthly_ret = monthly_price.pct_change().dropna()

    out = monthly_ret.rename("gross_ret").reset_index()
    out = out.rename(columns={"Date": "date_dt"})

    out["date_dt"] = pd.to_datetime(out["date_dt"]) + pd.offsets.MonthEnd(0)
    out = out[(out["date_dt"] >= start_date) & (out["date_dt"] <= end_date)].copy()

    out["date"] = out["date_dt"].dt.strftime("%Y%m%d").astype(int)
    out["yy"] = out["date_dt"].dt.year.astype(int)
    out["mm"] = out["date_dt"].dt.month.astype(int)

    out["method"] = method_name
    out["turnover_raw"] = 0.0
    out["turnover"] = 0.0
    out["cost"] = 0.0
    out["net_ret"] = out["gross_ret"]

    return out[
        [
            "date",
            "date_dt",
            "yy",
            "mm",
            "method",
            "gross_ret",
            "turnover_raw",
            "turnover",
            "cost",
            "net_ret",
        ]
    ].reset_index(drop=True)