"""
combined_plots_fixed.py — Cross-method comparison plots for the Forest Through the Trees project.

Reads backtest CSVs from:
    C:/Users/hongv/OneDrive/Tài liệu/forest-through-the-trees/backtest

Writes plots to:
    C:/Users/hongv/OneDrive/Tài liệu/forest-through-the-trees/backtest/plot

Expected files in BACKTEST_DIR:
    backtest_comparison_ts.csv       # Triple Sort
    backtest_comparison_ap.csv       # AP-Trees
    backtest_comparison_ap_rm.csv    # AP-Trees + RM
    NAVROR.csv                       # Optional: HedgeIndex Long/Short Equity

Plot sets:
  Set 1 — Within-method TC ablation, per method
  Set 2 — Static cross-method / momentum ablation
  Set 3 — Best-variant combined comparison
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================

BACKTEST_DIR = Path(r"C:\Users\hongv\OneDrive\Tài liệu\forest-through-the-trees\backtest")
PLOT_DIR = BACKTEST_DIR / "plot"

# =============================================================================
# Method configs
# =============================================================================

METHODS = {
    "ts": ("backtest_comparison_ts.csv", "Triple Sort"),
    "ap": ("backtest_comparison_ap.csv", "AP-Trees"),
    "ap_rm": ("backtest_comparison_ap_rm.csv", "AP-Trees + RM"),
}

BENCH_SPY = "S&P 500"
HF_CSV = BACKTEST_DIR / "NAVROR.csv"
HF_NAME = "CS L/S Equity HF Index"

# Colors
COLOR_STATIC_NO_TC = "#2166ac"
COLOR_STATIC_TC = "#4dac26"
COLOR_ROLLING_TC = "#d01c8b"
COLOR_SPY = "#969696"
COLOR_HF = "#f4a582"

COLOR_TS = "#2166ac"
COLOR_AP = "#d01c8b"
COLOR_AP_RM = "#e66101"


# =============================================================================
# Data loading helpers
# =============================================================================

def load_backtest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date_dt" not in df.columns:
        raise ValueError(f"{csv_path} must contain a date_dt column.")
    df["date_dt"] = pd.to_datetime(df["date_dt"])

    # Defensive: ensure these exist for benchmarks / plotting.
    if "gross_ret" not in df.columns and "net_ret" in df.columns:
        df["gross_ret"] = df["net_ret"]
    if "net_ret" not in df.columns and "gross_ret" in df.columns:
        df["net_ret"] = df["gross_ret"]
    if "turnover" not in df.columns:
        df["turnover"] = 0.0
    if "cost" not in df.columns:
        df["cost"] = 0.0

    return df.sort_values(["method", "date_dt"]).reset_index(drop=True)


def load_hf_index(csv_path: Path, start_date, end_date) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, skiprows=2)

    # Standardize columns
    raw = raw.rename(columns={"Date": "date_dt", "ROR": "ror_str"})

    # Keep only rows that actually look like dated monthly return rows
    raw["date_dt"] = pd.to_datetime(raw["date_dt"], errors="coerce")
    raw = raw[raw["date_dt"].notna()].copy()

    # Clean return column
    raw["ror_str"] = raw["ror_str"].astype(str)
    raw["gross_ret"] = (
        raw["ror_str"]
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    raw["gross_ret"] = pd.to_numeric(raw["gross_ret"], errors="coerce") / 100.0

    raw = raw[["date_dt", "gross_ret"]].dropna()
    raw["date_dt"] = raw["date_dt"] + pd.offsets.MonthEnd(0)

    raw = raw[(raw["date_dt"] >= start_date) & (raw["date_dt"] <= end_date)]
    raw = raw.sort_values("date_dt").reset_index(drop=True)

    raw["net_ret"] = raw["gross_ret"]
    raw["turnover"] = 0.0
    raw["cost"] = 0.0
    raw["method"] = HF_NAME
    raw["yy"] = raw["date_dt"].dt.year
    raw["mm"] = raw["date_dt"].dt.month

    return raw


def cumulative_wealth(r: pd.Series) -> pd.Series:
    return (1.0 + r.astype(float).fillna(0.0)).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    w = cumulative_wealth(r)
    return w / w.cummax() - 1.0


def annualized_sharpe(r: pd.Series) -> float:
    r = r.astype(float).dropna()
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    return float(np.sqrt(12) * r.mean() / sd)


def _date_indexed(df: pd.DataFrame, col: str) -> pd.Series:
    return df.sort_values("date_dt").set_index("date_dt")[col]


def _single_method(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep one method series; if duplicate labels exist, keep the first full label."""
    if df.empty:
        return df
    method_name = df["method"].iloc[0]
    out = df[df["method"] == method_name].copy()
    if out.empty:
        out = df.copy()
    return out.sort_values("date_dt")


def filter_variant(df: pd.DataFrame, *substrings: str, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return rows whose method contains all substrings and none of exclude."""
    if df.empty:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    for s in substrings:
        mask &= df["method"].str.contains(s, case=False, regex=False, na=False)
    for s in exclude:
        mask &= ~df["method"].str.contains(s, case=False, regex=False, na=False)

    return _single_method(df[mask].copy(), " / ".join(substrings))


def get_spy(df: pd.DataFrame) -> pd.DataFrame:
    return _single_method(
        df[df["method"].str.contains(BENCH_SPY, case=False, regex=False, na=False)].copy(),
        "SPY",
    )


def align_dates(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """Clip non-empty DataFrames to their common date range."""
    non_empty = [d for d in dfs if d is not None and not d.empty]
    if not non_empty:
        return []

    start = max(d["date_dt"].min() for d in non_empty)
    end = min(d["date_dt"].max() for d in non_empty)

    aligned = []
    for d in non_empty:
        clipped = d[(d["date_dt"] >= start) & (d["date_dt"] <= end)].copy()
        aligned.append(clipped)
    return aligned


# =============================================================================
# Variant selectors
# =============================================================================

def select_static_no_tc(bt: pd.DataFrame) -> pd.DataFrame:
    # Handles both "static (no TC)" and "static, no TC".
    return filter_variant(bt, "static", "no TC")


def select_static_stock_tc(bt: pd.DataFrame) -> pd.DataFrame:
    # Handles "static + stock-level TC" and similar.
    return filter_variant(bt, "static", "stock-level TC", exclude=("rolling", "portfolio"))


def select_rolling_stock_tc(bt: pd.DataFrame) -> pd.DataFrame:
    # Rolling TC-aware stock-level only; explicitly exclude portfolio-level line.
    return filter_variant(bt, "rolling TC-aware", "stock-level TC", exclude=("portfolio",))


def select_best_static(bt: pd.DataFrame) -> pd.DataFrame:
    v = select_static_stock_tc(bt)
    if v.empty:
        v = select_static_no_tc(bt)
    return v


def select_best_variant(bt: pd.DataFrame) -> pd.DataFrame:
    v = select_rolling_stock_tc(bt)
    if v.empty:
        v = select_static_stock_tc(bt)
    if v.empty:
        v = select_static_no_tc(bt)
    return v


# =============================================================================
# Plot primitives
# =============================================================================

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cumulative(series_list: list[tuple[str, pd.Series, str]], title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, r, color in series_list:
        if r.empty:
            continue
        ax.plot(r.index, cumulative_wealth(r), label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_drawdown_net(series_list: list[tuple[str, pd.Series, str]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, r, color in series_list:
        if r.empty:
            continue
        ax.plot(r.index, drawdown(r), label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_turnover(series_list: list[tuple[str, pd.Series, str]], title: str, out_path: Path, rolling: int = 12) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, to_series, color in series_list:
        if to_series.empty:
            continue
        ax.plot(to_series.index, to_series.rolling(rolling, min_periods=1).mean(), label=label, color=color)
    ax.set_title(f"{title} (12-month MA)")
    ax.set_ylabel("Turnover")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def _build_row(label: str, df: pd.DataFrame) -> dict:
    r_gross = df["gross_ret"].astype(float)
    r_net = df["net_ret"].astype(float)
    to = df["turnover"].astype(float).fillna(0.0)
    dd = drawdown(r_net)
    return {
        "label": label,
        "gross_sr": annualized_sharpe(r_gross),
        "net_sr": annualized_sharpe(r_net),
        "avg_turnover": float(to.mean()),
        "max_dd": float(dd.min()),
    }


def plot_summary_bar(rows: list[dict], title: str, out_path: Path) -> None:
    rows = [r for r in rows if r]
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.5), 6))
    ax.bar(x - 1.5 * width, [r["gross_sr"] for r in rows], width, label="Gross Sharpe", color="#2166ac")
    ax.bar(x - 0.5 * width, [r["net_sr"] for r in rows], width, label="Net Sharpe", color="#f4a582")
    ax.bar(x + 0.5 * width, [r["avg_turnover"] for r in rows], width, label="Avg Turnover", color="#4dac26")
    ax.bar(x + 1.5 * width, [r["max_dd"] for r in rows], width, label="Max DD (net)", color="#ca0020")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    _save(fig, out_path)


# =============================================================================
# Set 1 — Within-method TC ablation
# =============================================================================

def plot_tc_ablation(method_key: str, bt: pd.DataFrame, out_dir: Path) -> None:
    method_label = METHODS[method_key][1]

    a1 = select_static_no_tc(bt)
    a2 = select_static_stock_tc(bt)
    rc = select_rolling_stock_tc(bt)
    spy = get_spy(bt)

    if a1.empty or a2.empty:
        print(f"  WARNING: missing A1/A2 variants for {method_label}; skipping TC ablation.")
        print(f"    A1 rows: {len(a1)}, A2 rows: {len(a2)}")
        return

    pieces_raw = [a1, a2]
    labels_colors_raw = [
        (f"{method_label} — Static (no TC)", COLOR_STATIC_NO_TC),
        (f"{method_label} — Static + stock TC", COLOR_STATIC_TC),
    ]

    if not rc.empty:
        pieces_raw.append(rc)
        labels_colors_raw.append((f"{method_label} — Rolling TC-aware stock TC", COLOR_ROLLING_TC))

    if not spy.empty:
        pieces_raw.append(spy)
        labels_colors_raw.append(("S&P 500", COLOR_SPY))

    pieces = align_dates(*pieces_raw)
    labels_colors = labels_colors_raw[: len(pieces)]

    plot_cumulative(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        f"Cumulative Net Wealth — {method_label} TC Ablation",
        "Cumulative Net Wealth",
        out_dir / "plot_1_cumulative_net.png",
    )

    plot_drawdown_net(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        f"Drawdown — {method_label} TC Ablation",
        out_dir / "plot_2_drawdown.png",
    )

    # Turnover: exclude benchmarks.
    turn_pairs = [
        (lc[0], _date_indexed(p, "turnover"), lc[1])
        for p, lc in zip(pieces, labels_colors)
        if "S&P 500" not in lc[0] and HF_NAME not in lc[0]
    ]
    plot_turnover(turn_pairs, f"Turnover — {method_label} TC Ablation", out_dir / "plot_3_turnover.png")

    rows = [_build_row(lc[0], p) for p, lc in zip(pieces, labels_colors)]
    plot_summary_bar(rows, f"Summary Metrics — {method_label} TC Ablation", out_dir / "plot_4_summary.png")


# =============================================================================
# Set 2 — Cross-method momentum ablation
# =============================================================================

def plot_momentum_ablation(bt_ts: pd.DataFrame, bt_ap: pd.DataFrame, bt_ap_rm: pd.DataFrame, hf: pd.DataFrame, out_dir: Path) -> None:
    ts = select_best_static(bt_ts)
    ap = select_best_static(bt_ap)
    ap_rm = select_best_static(bt_ap_rm)
    spy = get_spy(bt_ap)

    pieces_raw = [ts, ap, ap_rm]
    labels_colors_raw = [
        ("Triple Sort", COLOR_TS),
        ("AP-Trees", COLOR_AP),
        ("AP-Trees + RM", COLOR_AP_RM),
    ]

    if not spy.empty:
        pieces_raw.append(spy)
        labels_colors_raw.append(("S&P 500", COLOR_SPY))
    if hf is not None and not hf.empty:
        pieces_raw.append(hf)
        labels_colors_raw.append((HF_NAME, COLOR_HF))

    if any(p.empty for p in pieces_raw[:3]):
        print("  WARNING: missing one of TS/AP/AP+RM static variants; skipping momentum ablation.")
        return

    pieces = align_dates(*pieces_raw)
    labels_colors = labels_colors_raw[: len(pieces)]

    plot_cumulative(
        [(lc[0], _date_indexed(p, "gross_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        "Cumulative Gross Wealth — Momentum Ablation",
        "Gross Wealth",
        out_dir / "plot_1_cumulative_gross.png",
    )

    plot_cumulative(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        "Cumulative Net Wealth — Momentum Ablation",
        "Net Wealth",
        out_dir / "plot_2_cumulative_net.png",
    )

    plot_drawdown_net(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        "Drawdown — Momentum Ablation",
        out_dir / "plot_3_drawdown.png",
    )

    rows = [_build_row(lc[0], p) for p, lc in zip(pieces, labels_colors)]
    plot_summary_bar(rows, "Summary Metrics — Momentum Ablation", out_dir / "plot_4_summary.png")


# =============================================================================
# Set 3 — Combined comparison
# =============================================================================

def plot_combined(bt_ts: pd.DataFrame, bt_ap: pd.DataFrame, bt_ap_rm: pd.DataFrame, hf: pd.DataFrame, out_dir: Path) -> None:
    ts = select_best_variant(bt_ts)
    ap = select_best_variant(bt_ap)
    ap_rm = select_best_variant(bt_ap_rm)
    spy = get_spy(bt_ap)

    pieces_raw = [ts, ap, ap_rm]
    labels_colors_raw = [
        ("Triple Sort (rolling TC-aware)", COLOR_TS),
        ("AP-Trees (rolling TC-aware)", COLOR_AP),
        ("AP-Trees + RM (rolling TC-aware)", COLOR_AP_RM),
    ]

    if not spy.empty:
        pieces_raw.append(spy)
        labels_colors_raw.append(("S&P 500", COLOR_SPY))
    if hf is not None and not hf.empty:
        pieces_raw.append(hf)
        labels_colors_raw.append((HF_NAME, COLOR_HF))

    if any(p.empty for p in pieces_raw[:3]):
        print("  WARNING: missing one of TS/AP/AP+RM best variants; skipping combined comparison.")
        return

    pieces = align_dates(*pieces_raw)
    labels_colors = labels_colors_raw[: len(pieces)]

    plot_cumulative(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        "Cumulative Net Wealth — Combined Comparison",
        "Net Wealth",
        out_dir / "plot_1_cumulative_net.png",
    )

    plot_drawdown_net(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        "Drawdown — Combined Comparison",
        out_dir / "plot_2_drawdown.png",
    )

    turn_pairs = [
        (lc[0], _date_indexed(p, "turnover"), lc[1])
        for p, lc in zip(pieces[:3], labels_colors[:3])
    ]
    plot_turnover(turn_pairs, "Turnover — Combined Comparison", out_dir / "plot_3_turnover.png")

    # Gross vs net Sharpe by method.
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [lc[0] for lc in labels_colors[:3]]
    x = np.arange(len(labels))
    width = 0.35
    gross_sr = [annualized_sharpe(p["gross_ret"]) for p in pieces[:3]]
    net_sr = [annualized_sharpe(p["net_ret"]) for p in pieces[:3]]
    ax.bar(x - width / 2, gross_sr, width, label="Gross Sharpe")
    ax.bar(x + width / 2, net_sr, width, label="Net Sharpe")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Gross vs Net Sharpe — Combined Comparison")
    ax.set_ylabel("Annualized Sharpe")
    ax.legend(fontsize=8)
    _save(fig, out_dir / "plot_4_sharpe_decay.png")

    rows = [_build_row(lc[0], p) for p, lc in zip(pieces, labels_colors)]
    plot_summary_bar(rows, "Summary Metrics — Combined Comparison", out_dir / "plot_5_summary.png")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print(f"Backtest directory: {BACKTEST_DIR}")
    print(f"Plot directory:     {PLOT_DIR}")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading backtest files...")
    bt: dict[str, pd.DataFrame] = {}
    for key, (fname, _) in METHODS.items():
        path = BACKTEST_DIR / fname
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {key}")
            bt[key] = pd.DataFrame()
        else:
            bt[key] = load_backtest(path)
            methods_found = bt[key]["method"].drop_duplicates().tolist()
            print(f"  {fname}: {len(bt[key])} rows")
            for m in methods_found:
                print(f"    - {m}")

    loaded = [df for df in bt.values() if not df.empty]
    if not loaded:
        raise RuntimeError("No backtest files loaded. Check BACKTEST_DIR and filenames.")

    ref = bt["ap"] if not bt.get("ap", pd.DataFrame()).empty else loaded[0]
    start = ref["date_dt"].min()
    end = ref["date_dt"].max()

    print(f"\nLoading HF index: {HF_CSV}")
    hf = pd.DataFrame()
    if HF_CSV.exists():
        try:
            hf = load_hf_index(HF_CSV, start, end)
            ref_dates = set(ref["date_dt"].unique())
            hf = hf[hf["date_dt"].isin(ref_dates)].copy()
            if not hf.empty:
                print(f"  HF index: {len(hf)} months, {hf['date_dt'].min().date()} to {hf['date_dt'].max().date()}")
            else:
                print("  WARNING: HF index loaded but has no dates overlapping the backtest window.")
        except Exception as e:
            print(f"  WARNING: could not load HF index: {e}")
            hf = pd.DataFrame()
    else:
        print("  WARNING: NAVROR.csv not found — HF benchmark omitted.")

    print("\n=== Set 1: TC ablation per method ===")
    for key, (_, display) in METHODS.items():
        if bt.get(key, pd.DataFrame()).empty:
            continue
        print(f"\n  {display}:")
        plot_tc_ablation(key, bt[key], PLOT_DIR / f"tc_ablation_{key}")

    print("\n=== Set 2: Momentum ablation / static cross-method comparison ===")
    if not any(bt.get(k, pd.DataFrame()).empty for k in ["ts", "ap", "ap_rm"]):
        plot_momentum_ablation(bt["ts"], bt["ap"], bt["ap_rm"], hf, PLOT_DIR / "momentum_ablation")
    else:
        print("  Skipping — not all three method files present.")

    print("\n=== Set 3: Combined comparison ===")
    if not any(bt.get(k, pd.DataFrame()).empty for k in ["ts", "ap", "ap_rm"]):
        plot_combined(bt["ts"], bt["ap"], bt["ap_rm"], hf, PLOT_DIR / "combined")
    else:
        print("  Skipping — not all three method files present.")

    print(f"\nDone. All plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
