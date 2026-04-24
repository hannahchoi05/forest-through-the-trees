from __future__ import annotations

from pathlib import Path
import pandas as pd

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

    # Treat missing candidate returns as 0.0 (consistent with upstream generation).
    for c in ports.columns:
        out[c] = out[c].astype(float).fillna(0.0)

    return out

