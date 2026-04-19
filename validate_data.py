"""
validate_data.py: load the provided yearly quantile
chunks, inspect structure, and check for data quality issues.

Usage: python validate_data.py data_chunk_files_quantile/LME_OP_Investment
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Union, List
import re
import pandas as pd
import numpy as np


def _resolve_chunks_path(chunks_path: Union[str, Path]) -> Path:
    """Resolve chunk directory from cwd, repo root, or the Data folder."""
    path = Path(chunks_path)
    candidates = [path]

    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parent
        candidates.extend([
            repo_root / path,
            repo_root / 'Data' / path,
        ])

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return path


def _extract_year(file_stem: str) -> Optional[int]:
    """Extract a 4-digit year from names like 1964 or y1964."""
    match = re.search(r'(19|20)\d{2}', file_stem)
    return int(match.group(0)) if match else None


def load_yearly_chunks(
    chunks_path: Union[str, Path],
    y_min: Optional[int] = None,
    y_max: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load all yearly chunk files into one long-format DataFrame.
    Supports .parquet, .csv, and .rds (the R binary format the paper uses).
    """
    chunks_path = _resolve_chunks_path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk directory not found: {chunks_path}. "
            "Try a path like Data/data_chunk_files_quantile/LME_OP_Investment"
        )

    dfs = []
    files_loaded = 0

    for f in sorted(chunks_path.iterdir()):
        year = _extract_year(f.stem)
        if year is None:
            continue  # not a year-keyed file

        if y_min is not None and year < y_min:
            continue
        if y_max is not None and year > y_max:
            continue

        if f.suffix == '.parquet':
            df = pd.read_parquet(f)
        elif f.suffix == '.csv':
            df = pd.read_csv(f)
        elif f.suffix == '.rds':
            try:
                import pyreadr
            except ImportError:
                raise ImportError("pip install pyreadr to read .rds files")
            df = list(pyreadr.read_r(str(f)).values())[0]
        else:
            continue

        dfs.append(df)
        files_loaded += 1
        if verbose:
            print(f"  loaded {f.name}: {len(df):,} rows")

    if not dfs:
        raise FileNotFoundError(f"No chunk files found in {chunks_path}")

    out = pd.concat(dfs, ignore_index=True)
    if 'date' in out.columns:
        out['date'] = pd.to_datetime(
            out['date'].astype(str), format='%Y%m%d', errors='coerce'
        )
        sort_cols = ['date', 'permno'] if 'permno' in out.columns else ['date']
        out = out.sort_values(sort_cols).reset_index(drop=True)

    if verbose:
        print(f"\nTotal: {files_loaded} files, {len(out):,} rows")
    return out


def summarize_panel(df: pd.DataFrame) -> Dict:
    """Basic shape and coverage summary."""
    summary = {
        'n_obs': len(df),
        'n_cols': df.shape[1],
        'columns': list(df.columns),
    }
    if 'date' in df.columns:
        summary['date_min'] = df['date'].min()
        summary['date_max'] = df['date'].max()
        summary['n_months'] = df['date'].nunique()
    if 'permno' in df.columns:
        summary['n_unique_stocks'] = df['permno'].nunique()
    return summary


def monthly_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Stocks per month — useful for spotting gaps and matching paper's Table 1."""
    if 'date' not in df.columns:
        raise ValueError("No 'date' column")
    out = (df.groupby('date')
             .agg(n_stocks=('permno', 'nunique') if 'permno' in df.columns else ('ret', 'size'),
                  mean_ret=('ret', 'mean'),
                  median_size=('size', 'median'))
             .reset_index())
    return out


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction missing per column — critical for confirming the paper's
    unbalanced-panel rule is respected (characteristics CAN be NaN)."""
    miss = df.isna().mean().sort_values(ascending=False)
    return miss.rename('frac_missing').to_frame()


def check_quantile_columns(df: pd.DataFrame, chars: List[str]) -> pd.DataFrame:
    """Verify each characteristic quantile column is in [0, 1]."""
    rows = []
    for char in chars:
        candidates = [f'{char}_q', char]
        col = next((name for name in candidates if name in df.columns), None)
        if col is None:
            rows.append({'requested_char': char, 'col': None, 'status': 'MISSING'})
            continue

        q = df[col].dropna()
        rows.append({
            'requested_char': char,
            'col': col,
            'min': q.min(),
            'max': q.max(),
            'mean': q.mean(),
            'median': q.median(),
            'status': 'ok' if (0 <= q.min() <= q.max() <= 1) else 'OUT_OF_RANGE',
        })
    return pd.DataFrame(rows)


def check_return_sanity(df: pd.DataFrame) -> Dict:
    """Sanity-check the return distribution against CRSP ballpark values."""
    r = df['ret'].dropna()
    return {
        'n': len(r),
        'mean_monthly': r.mean(),
        'median_monthly': r.median(),
        'std_monthly': r.std(),
        'min': r.min(),
        'max': r.max(),
        'pct_below_-99': (r < -0.99).mean(),  # the noise floor in the R code
        'pct_above_1': (r > 1.0).mean(),
    }


def run_full_report(
    chunks_path: Union[str, Path],
    chars: List[str] = ('LME', 'OP', 'Investment'),
) -> None:
    """One-shot diagnostic: run every check and print a readable report."""
    print("=" * 70)
    print(f"AP-Trees data validation report: {chunks_path}")
    print("=" * 70)

    print("\n[1] Loading chunks...")
    df = load_yearly_chunks(chunks_path, verbose=True)

    print("\n[2] Panel summary:")
    for k, v in summarize_panel(df).items():
        print(f"  {k}: {v}")

    print("\n[3] Monthly coverage (first and last 5 months):")
    cov = monthly_coverage(df)
    print(cov.head())
    print("...")
    print(cov.tail())

    print("\n[4] Missingness by column:")
    print(missingness_report(df))

    print(f"\n[5] Quantile column checks ({chars}):")
    print(check_quantile_columns(df, list(chars)))

    print("\n[6] Return sanity checks:")
    for k, v in check_return_sanity(df).items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate AP-Trees yearly quantile chunks.")
    parser.add_argument('chunks_dir', type=str,
                        help='e.g. data_chunk_files_quantile/LME_OP_Investment')
    parser.add_argument('--chars', nargs='+',
                        default=['LME', 'OP', 'Investment'])
    args = parser.parse_args()
    run_full_report(args.chunks_dir, args.chars)