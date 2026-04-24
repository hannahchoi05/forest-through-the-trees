from __future__ import annotations
from pathlib import Path
import pandas as pd
from utils import month_end_from_yy_mm


def load_yearly_chunks(chunk_dir: Path, chars: list[str], y_min: int, y_max: int) -> pd.DataFrame:
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
