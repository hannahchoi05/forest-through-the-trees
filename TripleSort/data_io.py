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
    dfs: list[pd.DataFrame] = []

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


def _month_index(start_yy: int, start_mm: int, n_months: int) -> pd.DataFrame:
    yy = start_yy
    mm = start_mm
    rows = []
    for _ in range(n_months):
        rows.append({"yy": yy, "mm": mm})
        mm += 1
        if mm == 13:
            mm = 1
            yy += 1
    out = pd.DataFrame(rows)
    out["date_dt"] = month_end_from_yy_mm(out)
    return out


def load_triplesort_excess_returns(
    portfolio_csv: Path,
    start_yy: int = 1964,
    start_mm: int = 1,
) -> pd.DataFrame:
    """
    Load the triple-sort excess portfolio return matrix and attach (yy, mm, date_dt).

    The CSV is expected to look like V1..VK with one row per month starting at 1964-01.
    Output columns: date_dt, yy, mm, port_V1..port_VK.
    """
    raw = pd.read_csv(portfolio_csv)
    idx = _month_index(start_yy, start_mm, n_months=len(raw))

    rename = {c: "port_" + c for c in raw.columns}
    ports = raw.rename(columns=rename)
    out = pd.concat([idx, ports], axis=1)
    out["date"] = out["date_dt"].dt.strftime("%Y%m%d").astype(int)

    # Treat missing candidate returns as 0.0 (consistent with upstream generation).
    for c in ports.columns:
        out[c] = out[c].astype(float).fillna(0.0)

    cols = ["date", "date_dt", "yy", "mm"] + list(ports.columns)
    return out[cols]


def load_yahoo_monthly_benchmark(
    ticker: str,
    start_date,
    end_date,
    method_name: str = "S&P 500 (adjusted close)",
) -> pd.DataFrame:
    """
    Load a monthly benchmark from Yahoo Finance using adjusted close.

    Output matches the backtest schema:
        date, date_dt, yy, mm, method, gross_ret, turnover_raw, turnover, cost, net_ret
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

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


def load_sp500_proxy(
    factor_dir: Path,
    method_name: str = "S&P 500 (SPY adjusted close)",
) -> pd.DataFrame:
    """
    Build a monthly S&P 500 benchmark series from Data/factor/tradable_factors.csv.

    Uses: gross_ret = (Mkt-RF) + rf
    and sets net_ret = gross_ret (no transaction costs).
    """
    f = factor_dir / "tradable_factors.csv"
    fac = pd.read_csv(f)

    if "Date" not in fac.columns:
        raise ValueError("tradable_factors.csv missing 'Date' column")
    if "Mkt-RF" not in fac.columns:
        raise ValueError("tradable_factors.csv missing 'Mkt-RF' column")

    if "rf" in fac.columns:
        rf = fac["rf"].astype(float)
    else:
        rf_file = factor_dir / "rf_factor.csv"
        rf = pd.read_csv(rf_file, header=None).squeeze().astype(float) / 100.0

    yymm = fac["Date"].astype(int)
    yy = (yymm // 100).astype(int)
    mm = (yymm % 100).astype(int)

    out = pd.DataFrame({"yy": yy, "mm": mm})
    out["date_dt"] = month_end_from_yy_mm(out)
    out["date"] = out["date_dt"].dt.strftime("%Y%m%d").astype(int)

    gross = fac["Mkt-RF"].astype(float) + rf.astype(float)

    out["method"] = method_name
    out["gross_ret"] = gross.to_numpy()
    out["turnover_raw"] = 0.0
    out["turnover"] = 0.0
    out["cost"] = 0.0
    out["net_ret"] = out["gross_ret"]

    return out.sort_values(["yy", "mm"]).reset_index(drop=True)
