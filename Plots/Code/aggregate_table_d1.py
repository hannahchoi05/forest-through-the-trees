"""
Aggregator: walks a directory of per-cross-section backtest CSVs and produces
a Table-D.1-style dataframe with one row per (cross-section, method).

Expected directory layout:

    backtest_root/
        <cs_key_1>/
            backtest_comparison.csv
        <cs_key_2>/
            backtest_comparison.csv
        ...

where <cs_key> is the .key attribute of a CrossSection (e.g. 'lme_op_investment').

Each backtest_comparison.csv is expected to have at least these columns:
    date_dt   (parseable as datetime)
    method    (string, used to identify AP-Trees vs TS32 vs TS64)
    net_ret   (monthly net return of the SDF / mean-variance efficient portfolio)

For each cross-section, the aggregator:
  1. Filters to the testing period (1994-01 to 2016-12 by default — paper's window)
  2. For each method-of-interest, computes:
        - SR          : monthly Sharpe ratio = mean(net_ret) / std(net_ret, ddof=1)
        - alpha_FF3   : intercept (in pct/month) from regressing net_ret on FF3
        - tstat_FF3   : t-stat of that intercept
        - alpha_FF5, tstat_FF5    (analogous, FF5)
        - (XSF, FF11) : placeholders, computed only if those factor sets are passed
  3. Returns a long dataframe: columns = [cs_id, cs_key, method, metric, value]
     pivot to wide form for Table D.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from cross_sections import CROSS_SECTIONS, CrossSection


# Default testing window from the paper: 1994-01 through 2016-12.
DEFAULT_TEST_START = pd.Timestamp("1994-01-01")
DEFAULT_TEST_END   = pd.Timestamp("2016-12-31")


# Map paper-friendly method labels to substrings that match the `method` column
# in your backtest_comparison.csv. Adjust these to match what your pipeline
# actually emits. The aggregator will pick the row whose `method` column
# *contains* the substring.
DEFAULT_METHOD_PATTERNS = {
    "AP-Trees": "AP-Trees AP-pruning (static, no TC)",   # variant A1, K=40
    "TS":       "Triple Sort static (no TC)",       # extend pipeline to emit this
}


@dataclass
class AggregatorConfig:
    backtest_root: Path
    factors_path: Path                          # tradable_factors.csv
    test_start: pd.Timestamp = DEFAULT_TEST_START
    test_end:   pd.Timestamp = DEFAULT_TEST_END
    method_patterns: dict = None                # {method_label: substring_to_match}
    backtest_filename: str = "backtest_comparison.csv"

    def __post_init__(self):
        if self.method_patterns is None:
            self.method_patterns = dict(DEFAULT_METHOD_PATTERNS)
        self.backtest_root = Path(self.backtest_root)
        self.factors_path = Path(self.factors_path)


# ---------------------------------------------------------------------------
# Factor data loading
# ---------------------------------------------------------------------------

# In this project's tradable_factors.csv the long-short factors are named after
# their underlying characteristics (e.g. LME = size long-short = "SMB"). The
# mapping below turns the paper's factor model definitions into column names.
# Update if your factor file uses different naming.
FF3_COLS  = ["Mkt-RF", "LME", "BEME"]
FF5_COLS  = ["Mkt-RF", "LME", "BEME", "OP", "Investment"]
FF11_COLS = ["Mkt-RF", "LME", "BEME", "OP", "Investment",
             "r12_2", "ST_REV", "LT_REV", "AC", "IdioVol", "Lturnover"]

# For the XSF (cross-section-specific) model, we need market + the long-short
# factors corresponding to the three characteristics in that cross-section.
# Cross-section panel columns -> factor column in tradable_factors.csv.
# Handles the case differences (ST_Rev vs ST_REV, LTurnover vs Lturnover).
PANEL_TO_FACTOR_COL = {
    "LME":        "LME",
    "BEME":       "BEME",
    "r12_2":      "r12_2",
    "OP":         "OP",
    "Investment": "Investment",
    "ST_Rev":     "ST_REV",
    "LT_Rev":     "LT_REV",
    "AC":         "AC",
    "IdioVol":    "IdioVol",
    "LTurnover":  "Lturnover",
}


def _parse_dates(series: pd.Series) -> pd.Series:
    """Robust date parsing: handles YYYYMM int (e.g. 196401), 'YYYY-MM',
    pd.Timestamp, etc. Critical: checks YYYYMM BEFORE generic to_datetime,
    because pd.to_datetime(199401) silently treats integers as nanoseconds
    since epoch and returns 1970-01-01."""
    # YYYYMM-as-int detection (handles both int and string representations).
    s = series.astype(str).str.strip()
    if s.str.match(r"^\d{6}$").all():
        return pd.to_datetime(s, format="%Y%m") + pd.offsets.MonthEnd(0)
    # Otherwise let pandas figure it out.
    try:
        return pd.to_datetime(series)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not parse date column. Sample values: "
                         f"{series.head().tolist()}") from e


def load_factors(path: Path) -> pd.DataFrame:
    """
    Load tradable_factors.csv. Expected schema for this project:

        Date       (YYYYMM int, e.g. 196401)
        Mkt-RF     market excess return
        LME        size long-short
        BEME       value long-short
        OP         profitability long-short
        Investment investment long-short
        r12_2      momentum long-short
        ST_REV     short-term reversal long-short
        LT_REV     long-term reversal long-short
        AC         accruals long-short
        IdioVol    idiosyncratic vol long-short
        Lturnover  turnover long-short
        rf         monthly risk-free rate

    Returns indexed by date_dt (month-end). All factor returns must be in the
    same units as net_ret in the backtest CSV (decimal monthly returns).
    """
    df = pd.read_csv(path)

    date_col = None
    for cand in ("date_dt", "date", "Date", "DATE", "month"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"No recognized date column in {path}")

    df["date_dt"] = _parse_dates(df[date_col])
    df = df.set_index("date_dt").sort_index()

    # Standardize the risk-free column name (file has lowercase 'rf').
    if "rf" in df.columns and "RF" not in df.columns:
        df["RF"] = df["rf"]

    required = FF11_COLS  # the strictest set we use
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.warn(
            f"Factor file missing FF11 columns: {missing}. "
            f"Affected alphas will be NaN."
        )
    return df


# ---------------------------------------------------------------------------
# Per-method statistics
# ---------------------------------------------------------------------------

def _ols_with_tstats(y: np.ndarray, X: np.ndarray) -> tuple:
    """
    Plain OLS via the normal equations; returns (coef, tstats).
    X should already include a constant column (we add one here).
    Uses heteroskedasticity-naive standard errors. For Newey-West, swap in
    statsmodels.regression.linear_model.OLS with .get_robustcov_results().
    """
    X = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstats = coef / se
    return coef, tstats


def _alpha_and_tstat(y: np.ndarray, factor_block: pd.DataFrame, cols: list) -> tuple:
    """
    Helper: regress y on the given factor columns, return (alpha, t_stat).
    Returns (NaN, NaN) if any column is missing.
    """
    if not all(c in factor_block.columns for c in cols):
        return np.nan, np.nan
    X = factor_block[cols].to_numpy(dtype=float)
    coef, t = _ols_with_tstats(y, X)
    return coef[0], t[0]


def compute_method_stats(
    rets: pd.Series,
    factors: pd.DataFrame,
    xsf_cols: list | None = None,
) -> dict:
    """
    Given a series of monthly net returns (indexed by date_dt) and the factor
    panel, compute SR + alpha/t-stat against FF3, FF5, XSF, FF11.

    Parameters
    ----------
    rets : pd.Series
        Monthly net returns of the SDF (decimal, indexed by date_dt).
    factors : pd.DataFrame
        Factor panel from load_factors().
    xsf_cols : list of str, optional
        Column names for the XSF (cross-section-specific) model. Should be
        ['Mkt-RF', f1, f2, f3] where f1, f2, f3 are the three long-short
        factors corresponding to this cross-section's three characteristics.
        If None, XSF stats are NaN.

    Returns
    -------
    dict with keys: SR, alpha_FFx, tstat_FFx for x in {3, 5, XSF, 11}, n_obs.
    Alphas are in the same units as net_ret (decimal monthly).
    """
    out = {"n_obs": len(rets)}

    sr = rets.mean() / rets.std(ddof=1) if rets.std(ddof=1) > 0 else np.nan
    out["SR"] = sr

    aligned = pd.merge(
        rets.rename("ret").reset_index(),
        factors.reset_index(),
        on="date_dt",
        how="inner",
    ).dropna(subset=["ret"])

    if len(aligned) < 24:
        for k in ("alpha_FF3", "tstat_FF3", "alpha_FF5", "tstat_FF5",
                  "alpha_XSF", "tstat_XSF", "alpha_FF11", "tstat_FF11"):
            out[k] = np.nan
        return out

    y = aligned["ret"].to_numpy(dtype=float)

    # Run the four regressions.
    out["alpha_FF3"],  out["tstat_FF3"]  = _alpha_and_tstat(y, aligned, FF3_COLS)
    out["alpha_FF5"],  out["tstat_FF5"]  = _alpha_and_tstat(y, aligned, FF5_COLS)
    out["alpha_FF11"], out["tstat_FF11"] = _alpha_and_tstat(y, aligned, FF11_COLS)
    if xsf_cols is not None:
        out["alpha_XSF"], out["tstat_XSF"] = _alpha_and_tstat(y, aligned, xsf_cols)
    else:
        out["alpha_XSF"], out["tstat_XSF"] = np.nan, np.nan

    return out


def _xsf_cols_for(cs: CrossSection) -> list:
    """Build the XSF factor column list for a cross-section: market + the
    three long-short factors corresponding to its three characteristics."""
    cols = ["Mkt-RF"]
    for panel_col in cs.panel_chars:
        if panel_col not in PANEL_TO_FACTOR_COL:
            return None  # Can't build XSF for this cross-section
        cols.append(PANEL_TO_FACTOR_COL[panel_col])
    return cols


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

def aggregate(cfg: AggregatorConfig) -> pd.DataFrame:
    """
    Walk all 36 cross-sections; for each that has a backtest CSV, compute
    stats for every method in cfg.method_patterns. Returns a long dataframe.
    """
    factors = load_factors(cfg.factors_path)

    # All metric columns we expect in every row.
    metric_keys = ("SR",
                   "alpha_FF3", "tstat_FF3",
                   "alpha_FF5", "tstat_FF5",
                   "alpha_XSF", "tstat_XSF",
                   "alpha_FF11", "tstat_FF11")

    rows = []
    n_found = 0
    for cs in CROSS_SECTIONS:
        bt_path = cfg.backtest_root / cs.key / cfg.backtest_filename
        if not bt_path.exists():
            continue
        n_found += 1

        bt = pd.read_csv(bt_path)
        bt["date_dt"] = pd.to_datetime(bt["date_dt"])
        bt = bt[(bt["date_dt"] >= cfg.test_start) & (bt["date_dt"] <= cfg.test_end)]

        xsf_cols = _xsf_cols_for(cs)

        for method_label, pattern in cfg.method_patterns.items():
            sub = bt[bt["method"].str.contains(pattern, na=False, regex=False)]
            if sub.empty:
                row = {"cs_id": cs.id, "cs_key": cs.key, "method": method_label,
                       **{k: np.nan for k in metric_keys}, "n_obs": 0}
            else:
                rets = sub.set_index("date_dt")["net_ret"]
                stats = compute_method_stats(rets, factors, xsf_cols=xsf_cols)
                row = {"cs_id": cs.id, "cs_key": cs.key, "method": method_label, **stats}
            rows.append(row)

    print(f"Aggregated {n_found}/{len(CROSS_SECTIONS)} cross-sections "
          f"from {cfg.backtest_root}")
    return pd.DataFrame(rows)


def to_table_d1_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long aggregator output into a Table-D.1-shaped wide dataframe:
    one row per cross-section, columns like SR_AP-Trees-40, SR_TS-32,
    tstat_FF5_AP-Trees-40, etc.
    """
    metrics = [c for c in long_df.columns
               if c not in ("cs_id", "cs_key", "method", "n_obs")]
    wide = long_df.pivot(index=["cs_id", "cs_key"], columns="method", values=metrics)
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    return wide.reset_index().sort_values("cs_id").reset_index(drop=True)