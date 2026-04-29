"""
Slice the wide-format raw characteristic CSVs into 36 quantile panels matching
the layout your pipeline expects:

    data_chunk_files_quantile/<TRIPLE>/yYYYY.csv

with columns: yy, mm, date, permno, ret, <CHAR1>, <CHAR2>, <CHAR3>, size

where <CHAR1>, <CHAR2>, <CHAR3> hold the cross-sectional MONTHLY quantile of
each stock (values in [0, 1]), and `size` holds the raw market cap (LME) used
for value-weighting.

Inputs (wide-format, see Pelger Dropbox 'characteristics/' folder):
    <RAW_DIR>/date.csv         648 rows of YYYYMMDD ints, header 'x'
    <RAW_DIR>/LME.csv          header: Date, LME.<permno1>, LME.<permno2>, ...
    <RAW_DIR>/RET.csv          same shape, header: Date, RET.<permno>...
    <RAW_DIR>/<CHAR>.csv       same shape, one per characteristic
    <RAW_DIR>/rf_factor.csv    648 rows of monthly RF values, header position-only

This script does NOT produce a slice for cross-section 16 (LME_OP_Investment)
because you already have it. Set OVERWRITE_EXISTING = True to regenerate.

Usage
-----
    python slice_quantile_panels.py --raw-dir /path/to/characteristics \\
                                    --out-dir ./data_chunk_files_quantile

Memory: holds ~12 wide CSVs in memory simultaneously (~10 GB peak with naive
pandas reads). The script melts to long format eagerly to keep the working set
manageable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add repo root to path so cross_sections module resolves.
sys.path.insert(0, str(Path(__file__).parent))
from cross_sections import CROSS_SECTIONS, CHAR_NAME_MAP


# Paper-name -> raw CSV stem mapping. Adjust if filenames differ.
PAPER_TO_FILE_STEM = {
    "LME":        "LME",
    "BEME":       "BEME",
    "r12_2":      "r12_2",
    "OP":         "OP",
    "Investment": "Investment",
    "ST_Rev":     "ST_Rev",
    "LT_Rev":     "LT_Rev",
    "AC":         "AC",
    "IdioVol":    "IdioVol",
    "LTurnover":  "LTurnover",
}

REQUIRED_FILES = ["date.csv", "RET.csv"] + [f"{s}.csv" for s in PAPER_TO_FILE_STEM.values()]


def check_inputs(raw_dir: Path) -> None:
    """Fail fast if any expected file is missing."""
    missing = [f for f in REQUIRED_FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {raw_dir}:\n  " +
            "\n  ".join(missing) +
            "\n\nAll 12 files (date.csv, RET.csv, plus 10 characteristics) "
            "must be present. Download from Pelger's 'characteristics/' "
            "Dropbox folder."
        )


def load_dates(raw_dir: Path) -> pd.Series:
    """date.csv: 648 rows of YYYYMMDD ints (header 'x'). Return as int Series."""
    s = pd.read_csv(raw_dir / "date.csv")
    if "x" in s.columns:
        s = s["x"]
    else:
        s = s.iloc[:, 0]
    return s.astype(int)


def load_wide_to_long(
    path: Path,
    dates: pd.Series,
    char_name: str,
    column_prefix: str | None = None,
) -> pd.DataFrame:
    """
    Load a wide-format characteristic CSV and melt to long.

    Wide schema: column 0 is 'Date' (YYYYMMDD int), columns 1.. are named
    '<PREFIX>.<permno>'. We strip the prefix to recover permno.

    Parameters
    ----------
    path : Path
        Path to the wide CSV.
    dates : pd.Series
        Calendar dates for each row, length must match # rows in the CSV.
    char_name : str
        Name to give the value column in the output long-format DataFrame
        (e.g., "ret", "LME").
    column_prefix : str, optional
        The prefix used in column names like '<prefix>.<permno>'. Defaults
        to char_name. We also try a fallback where underscores in the prefix
        are replaced with dots — Pelger's r12_2.csv has columns named
        'r12.2.10000' rather than 'r12_2.10000'.

    Robust prefix-stripping: pandas auto-renames duplicate columns by
    appending '.1', '.2', etc., and R's `write.csv` sometimes adds a
    leading index column. We strip the known prefix and any trailing
    `.N` disambiguator, then keep only columns whose remainder parses
    as an integer permno. Other columns are silently dropped.

    Returns a long DataFrame: [month_idx, date, permno, <char_name>], drops
    NaN values. month_idx is 1-based row index in date.csv.
    """
    if column_prefix is None:
        column_prefix = char_name

    df = pd.read_csv(path)

    # Drop the date column from the wide df.
    date_col_candidates = [c for c in df.columns if c.lower() == "date"]
    if date_col_candidates:
        df = df.drop(columns=date_col_candidates)

    if len(df) != len(dates):
        raise ValueError(
            f"{path.name} has {len(df)} rows but date.csv has {len(dates)}. "
            f"Cannot align positionally."
        )

    # Try the canonical prefix first; if that matches no columns, fall back
    # to underscore→dot substitution. Pelger's r12_2.csv uses 'r12.2.<permno>'
    # column naming despite the file being named 'r12_2.csv'.
    candidate_prefixes = [f"{column_prefix}."]
    if "_" in column_prefix:
        candidate_prefixes.append(f"{column_prefix.replace('_', '.')}.")

    chosen_prefix = None
    for p in candidate_prefixes:
        if any(c.startswith(p) for c in df.columns):
            chosen_prefix = p
            break
    if chosen_prefix is None:
        raise ValueError(
            f"{path.name}: no columns match any of the candidate prefixes "
            f"{candidate_prefixes}. First few columns in file: "
            f"{list(df.columns)[:5]}"
        )

    # Build a (col, permno_int) mapping.
    keep_cols = {}
    skipped = []
    seen_permnos = set()
    for c in df.columns:
        if not c.startswith(chosen_prefix):
            skipped.append(c)
            continue
        rest = c[len(chosen_prefix):]
        # Strip pandas disambiguator '.N' from duplicate columns.
        rest = rest.split(".")[0]
        try:
            permno = int(rest)
        except ValueError:
            skipped.append(c)
            continue
        if permno in seen_permnos:
            skipped.append(c)
            continue
        keep_cols[c] = permno
        seen_permnos.add(permno)

    if skipped:
        print(f"    {path.name}: prefix={chosen_prefix!r}, "
              f"skipped {len(skipped)} columns. Examples: {skipped[:3]}")

    df = df[list(keep_cols.keys())].rename(columns=keep_cols)
    df.columns = df.columns.astype(int)

    # Build metadata columns separately, then concat (avoids fragmentation).
    n_rows = len(df)
    meta = pd.DataFrame({
        "month_idx": np.arange(1, n_rows + 1),
        "date": dates.values,
    })
    df = pd.concat([meta, df.reset_index(drop=True)], axis=1)

    long = df.melt(id_vars=["month_idx", "date"], var_name="permno",
                   value_name=char_name)
    long = long.dropna(subset=[char_name])
    long["permno"] = long["permno"].astype(int)
    return long


def build_master_long(
    raw_dir: Path, y_min: int = 1964, y_max: int = 2016
) -> pd.DataFrame:
    """
    Build one master long-format panel: every (month_idx, date, permno) with
    a valid return, plus all 10 characteristic raw values (NaN where missing).

    Critical timing quirk
    ---------------------
    `date.csv` and the wide CSVs (RET.csv, LME.csv, ...) are positionally
    aligned by row index, NOT by the calendar date in the `date` column. In
    Pelger's R script (Step1_Combine_Raw_Chars_Convert_Quantile_Split):

        i = 12 * (year - y_min) + month
        date = date[i]            # the calendar label at that row
        ret  = RET[i, ...]        # the actual return at that row

    Empirically, the data in row 1 of these wide CSVs corresponds to
    (year=1964, month=1), but `date[1] = 19630131`. So the calendar labels
    are offset by 12 months from what the data represents. We replicate the
    convention: derive yy/mm from row position, stamp the `date` field from
    date.csv as-is.

    Parameters
    ----------
    y_min : int
        Calendar year that row 1 of date.csv represents. Default 1964 to
        match Pelger's invocation in main_simplified.R.
    y_max : int
        Last year to include. Default 2016 (paper's data ends Dec 2016).
        date.csv has 648 rows = 54 calendar-labeled years, but only 53 are
        valid data years; rows 637-648 may contain stale/empty data despite
        having parseable date labels.
    """
    print(f"Loading inputs from {raw_dir}...")
    dates = load_dates(raw_dir)
    print(f"  date.csv: {len(dates)} months "
          f"(label range {dates.iloc[0]} .. {dates.iloc[-1]})")

    n_keep = (y_max - y_min + 1) * 12
    if n_keep > len(dates):
        raise ValueError(
            f"Requested {n_keep} months ({y_min}-{y_max}) but date.csv "
            f"only has {len(dates)} rows."
        )
    print(f"  Keeping rows 1..{n_keep} (yy {y_min}..{y_max})")

    # Returns: required for every output row. RET.csv columns use prefix "RET."
    # but we want the long-format value column named "ret" (lowercase, to
    # match the existing pipeline's input schema).
    ret_long = load_wide_to_long(
        raw_dir / "RET.csv", dates, char_name="ret", column_prefix="RET"
    )
    ret_long = ret_long[ret_long["month_idx"] <= n_keep].copy()
    print(f"  RET.csv -> {len(ret_long):,} (month_idx, date, permno, ret) rows "
          f"after y_max filter")

    master = ret_long

    # Characteristics: outer-join on (month_idx, date, permno).
    for paper_name, file_stem in PAPER_TO_FILE_STEM.items():
        long = load_wide_to_long(
            raw_dir / f"{file_stem}.csv", dates, paper_name
        )
        long = long[long["month_idx"] <= n_keep]
        master = master.merge(long, on=["month_idx", "date", "permno"],
                              how="left")
        print(f"  {file_stem}.csv merged ({len(long):,} non-NaN values)")

    # Derive yy and mm from row position. month_idx is 1-based:
    # row 1 -> (y_min, 1), row 13 -> (y_min+1, 1), etc.
    master["yy"] = y_min + (master["month_idx"] - 1) // 12
    master["mm"] = ((master["month_idx"] - 1) % 12) + 1

    print(f"Master panel: {len(master):,} rows, "
          f"{master['permno'].nunique():,} unique permnos, "
          f"yy range {master['yy'].min()}-{master['yy'].max()}")
    return master


def to_quantile(g: pd.Series) -> pd.Series:
    """Cross-sectional quantile within a single month, matching Pelger's R code:

        convert_quantile <- function(x) {
            x[!is.na(x)] = (rank(na.omit(x)) - 1) / (length(na.omit(x)) - 1)
            return(x)
        }

    This is (rank - 1) / (N - 1): smallest non-NA value gets 0.0, largest
    gets 1.0, others linearly spaced. Different from rank(pct=True) which
    uses rank / N (smallest gets 1/N, largest gets 1).

    Pandas `rank(method='average')` matches R's `rank()` for ties (average
    rank assigned to tied values).
    """
    n = g.notna().sum()
    if n <= 1:
        return pd.Series(np.nan, index=g.index)
    return (g.rank(method="average") - 1) / (n - 1)


def slice_one_triple(
    master: pd.DataFrame, char1: str, char2: str, char3: str
) -> pd.DataFrame:
    """
    Build the quantile panel for one cross-section. Output columns match what
    the existing pipeline consumes: yy, mm, date, permno, ret, <c1>, <c2>,
    <c3>, size.

    Universe rule: include only (month_idx, permno) with valid ret AND valid
    values for ALL THREE chosen characteristics. Quantiles are computed
    within this restricted universe per month, so each month's universe is
    balanced for the triple.

    Quantiles use Pelger's (rank - 1) / (N - 1) formula via to_quantile().
    """
    cols_needed = ["yy", "mm", "date", "month_idx", "permno", "ret",
                   char1, char2, char3]
    sub = master[cols_needed].dropna(subset=[char1, char2, char3, "ret"]).copy()

    # 'size' = the raw LME (market cap), used for value-weighting. char1 is
    # always LME by construction in this paper (every cross-section is
    # Size × X × Y).
    assert char1 == "LME", "Slicer assumes char1 is LME (matches the paper)."
    sub["size"] = sub[char1]

    # Replace each char with its monthly cross-sectional quantile. Group by
    # month_idx, not `date`, to match Pelger's R script (which iterates over
    # row indices in the wide CSV, not unique date values).
    for c in (char1, char2, char3):
        sub[c] = sub.groupby("month_idx")[c].transform(to_quantile)

    sub = sub.sort_values(["month_idx", "permno"]).reset_index(drop=True)
    return sub[["yy", "mm", "date", "permno", "ret", char1, char2, char3, "size"]]


def write_yearly_csvs(panel: pd.DataFrame, out_dir: Path) -> None:
    """Split a single triple's panel by year and write yYYYY.csv files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for yy, group in panel.groupby("yy"):
        group.to_csv(out_dir / f"y{yy}.csv", index=False, quoting=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True,
                   help="Directory with date.csv, LME.csv, RET.csv, etc.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output root for data_chunk_files_quantile/<TRIPLE>/")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-triple directories.")
    p.add_argument("--y-min", type=int, default=1964,
                   help="Calendar year that row 1 of date.csv represents. "
                        "Default 1964 to match Pelger's invocation.")
    p.add_argument("--y-max", type=int, default=2016,
                   help="Last year to include. Default 2016 (paper's data "
                        "ends Dec 2016).")
    p.add_argument("--only", type=int, nargs="*", default=None,
                   help="If given, only build cross-sections with these IDs "
                        "(1-36). Useful for testing one triple before the full run.")
    args = p.parse_args()

    check_inputs(args.raw_dir)
    master = build_master_long(args.raw_dir, y_min=args.y_min, y_max=args.y_max)

    triples_to_build = CROSS_SECTIONS
    if args.only:
        triples_to_build = [cs for cs in CROSS_SECTIONS if cs.id in args.only]
        if not triples_to_build:
            raise ValueError(f"--only IDs {args.only} matched no cross-sections")

    n_built = 0
    n_skipped = 0
    for cs in triples_to_build:
        # Output directory uses uppercase (LME_OP_Investment), matching your
        # existing example layout. Note .key uses lowercase, so we override.
        triple_dir = args.out_dir / "_".join(
            CHAR_NAME_MAP[c] for c in (cs.char1, cs.char2, cs.char3)
        )

        if triple_dir.exists() and not args.overwrite:
            print(f"[{cs.id:2d}] {triple_dir.name}  -> exists, skipping")
            n_skipped += 1
            continue

        char_panel_names = [CHAR_NAME_MAP[c] for c in (cs.char1, cs.char2, cs.char3)]
        panel = slice_one_triple(master, *char_panel_names)

        write_yearly_csvs(panel, triple_dir)
        print(f"[{cs.id:2d}] {triple_dir.name}  -> "
              f"{len(panel):,} rows, {panel['yy'].min()}-{panel['yy'].max()}")
        n_built += 1

    print(f"\nDone. Built {n_built}, skipped {n_skipped}.")


if __name__ == "__main__":
    main()