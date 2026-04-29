"""
combined_plots_ts32_ts64_fixed.py

Cross-method comparison plots for Forest Through the Trees.

Reads backtest CSVs from:
    C:/Users/hongv/OneDrive/Tài liệu/forest-through-the-trees/backtest

Expected files:
    backtest_comparison_ts.csv       # contains BOTH TS32 and TS64, method names must include TS32 / TS64
    backtest_comparison_ap.csv       # AP-Trees
    backtest_comparison_ap_rm.csv    # AP-Trees + RM
    NAVROR.csv                       # optional hedge fund benchmark

Writes plots to:
    backtest/plot

Plot sets:
  Set 1 — TC ablation per method / variant group
  Set 2 — Static cross-method comparison
  Set 3 — Rolling TC-aware combined comparison
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
# Expected files
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
COLOR_TS32 = "#2166ac"
COLOR_TS64 = "#67a9cf"
COLOR_AP = "#d01c8b"
COLOR_AP_RM = "#e66101"
COLOR_SPY = "#969696"
COLOR_HF = "#f4a582"

COLOR_STATIC_NO_TC = "#2166ac"
COLOR_STATIC_TC = "#4dac26"
COLOR_ROLLING_PORT_TC = "#762a83"
COLOR_ROLLING_STOCK_TC = "#d01c8b"

# =============================================================================
# Helpers
# =============================================================================

def load_backtest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date_dt" not in df.columns:
        raise ValueError(f"{csv_path} must contain date_dt")

    df["date_dt"] = pd.to_datetime(df["date_dt"])

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
    raw = raw.rename(columns={"Date": "date_dt", "ROR": "ror_str"})

    raw["date_dt"] = pd.to_datetime(raw["date_dt"], errors="coerce")
    raw = raw[raw["date_dt"].notna()].copy()

    raw["ror_str"] = raw["ror_str"].astype(str)
    raw["gross_ret"] = (
        raw["ror_str"].str.replace("%", "", regex=False).str.strip()
    )
    raw["gross_ret"] = pd.to_numeric(raw["gross_ret"], errors="coerce") / 100.0

    raw = raw[["date_dt", "gross_ret"]].dropna()
    raw["date_dt"] = raw["date_dt"] + pd.offsets.MonthEnd(0)
    raw = raw[(raw["date_dt"] >= start_date) & (raw["date_dt"] <= end_date)].copy()
    raw = raw.sort_values("date_dt").reset_index(drop=True)

    raw["net_ret"] = raw["gross_ret"]
    raw["turnover"] = 0.0
    raw["cost"] = 0.0
    raw["method"] = HF_NAME
    raw["yy"] = raw["date_dt"].dt.year
    raw["mm"] = raw["date_dt"].dt.month
    raw["date"] = raw["date_dt"].dt.strftime("%Y%m%d").astype(int)

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
    return float(np.sqrt(12.0) * r.mean() / sd)


def _date_indexed(df: pd.DataFrame, col: str) -> pd.Series:
    return df.sort_values("date_dt").set_index("date_dt")[col]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def align_dates(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    non_empty = [d for d in dfs if d is not None and not d.empty]
    if not non_empty:
        return []
    start = max(d["date_dt"].min() for d in non_empty)
    end = min(d["date_dt"].max() for d in non_empty)
    return [d[(d["date_dt"] >= start) & (d["date_dt"] <= end)].copy() for d in non_empty]


def _single_method(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    method = df["method"].iloc[0]
    return df[df["method"] == method].copy().sort_values("date_dt")


def filter_variant(df: pd.DataFrame, *substrings: str, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = pd.Series(True, index=df.index)
    for s in substrings:
        mask &= df["method"].str.contains(s, case=False, regex=False, na=False)
    for s in exclude:
        mask &= ~df["method"].str.contains(s, case=False, regex=False, na=False)
    return _single_method(df[mask].copy())


def filter_method_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Group can be TS32, TS64, AP, AP_RM."""
    if df.empty:
        return df.copy()

    s = df["method"].astype(str)
    if group == "TS32":
        return df[s.str.contains("TS32", case=False, regex=False, na=False)].copy()
    if group == "TS64":
        return df[s.str.contains("TS64", case=False, regex=False, na=False)].copy()
    if group == "AP_RM":
        return df[s.str.contains("RM", case=False, regex=False, na=False)].copy()
    if group == "AP":
        # Plain AP-Trees file should not include RM, but be defensive.
        return df[~s.str.contains("RM", case=False, regex=False, na=False)].copy()
    raise ValueError(f"Unknown group: {group}")


def get_spy(df: pd.DataFrame) -> pd.DataFrame:
    return _single_method(
        df[df["method"].str.contains(BENCH_SPY, case=False, regex=False, na=False)].copy()
    )


def select_static_no_tc(bt: pd.DataFrame) -> pd.DataFrame:
    return filter_variant(bt, "static", "no TC")


def select_static_stock_tc(bt: pd.DataFrame) -> pd.DataFrame:
    # Robust to stock-level / stock level / stock_tc naming.
    if bt.empty:
        return bt.copy()
    s = bt["method"].astype(str)
    mask = (
        s.str.contains("static", case=False, regex=False, na=False)
        & s.str.contains("stock", case=False, regex=False, na=False)
        & s.str.contains("TC", case=False, regex=False, na=False)
        & ~s.str.contains("rolling", case=False, regex=False, na=False)
        & ~s.str.contains("portfolio", case=False, regex=False, na=False)
    )
    return _single_method(bt[mask].copy())


def select_rolling_portfolio_tc(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return bt.copy()
    s = bt["method"].astype(str)
    mask = (
        s.str.contains("rolling", case=False, regex=False, na=False)
        & s.str.contains("portfolio", case=False, regex=False, na=False)
        & s.str.contains("TC", case=False, regex=False, na=False)
    )
    return _single_method(bt[mask].copy())


def select_rolling_stock_tc(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return bt.copy()
    s = bt["method"].astype(str)
    mask = (
        s.str.contains("rolling", case=False, regex=False, na=False)
        & s.str.contains("stock", case=False, regex=False, na=False)
        & s.str.contains("TC", case=False, regex=False, na=False)
        & ~s.str.contains("portfolio", case=False, regex=False, na=False)
    )
    return _single_method(bt[mask].copy())


def select_preferred_variant(bt: pd.DataFrame) -> pd.DataFrame:
    """Preferred realistic variant: rolling stock TC, then static stock TC, then static no TC."""
    v = select_rolling_stock_tc(bt)
    if v.empty:
        v = select_static_stock_tc(bt)
    if v.empty:
        v = select_static_no_tc(bt)
    return v


def select_static_comparison_variant(bt: pd.DataFrame) -> pd.DataFrame:
    """For static cross-method comparison: static stock TC if available, else static no TC."""
    v = select_static_stock_tc(bt)
    if v.empty:
        v = select_static_no_tc(bt)
    return v


def _build_row(label: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    gross = df["gross_ret"].astype(float)
    net = df["net_ret"].astype(float)
    turnover = df["turnover"].astype(float).fillna(0.0)
    dd = drawdown(net)
    return {
        "label": label,
        "gross_sr": annualized_sharpe(gross),
        "net_sr": annualized_sharpe(net),
        "avg_turnover": float(turnover.mean()),
        "max_dd": float(dd.min()),
    }

# =============================================================================
# Plot primitives
# =============================================================================

def plot_cumulative(series_list: list[tuple[str, pd.Series, str]], title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, r, color in series_list:
        if r.empty:
            continue
        ax.plot(r.index, cumulative_wealth(r), label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_drawdown_net(series_list: list[tuple[str, pd.Series, str]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, r, color in series_list:
        if r.empty:
            continue
        ax.plot(r.index, drawdown(r), label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_turnover(series_list: list[tuple[str, pd.Series, str]], title: str, out_path: Path, rolling: int = 12) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, to_series, color in series_list:
        if to_series.empty:
            continue
        ax.plot(to_series.index, to_series.rolling(rolling, min_periods=1).mean(), label=label, color=color)
    ax.set_title(f"{title} (12-month MA)")
    ax.set_ylabel("Turnover")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_summary_bar(rows: list[dict], title: str, out_path: Path) -> None:
    rows = [r for r in rows if r]
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 2.0), 7))
    ax.bar(x - 1.5 * width, [r["gross_sr"] for r in rows], width, label="Gross Sharpe")
    ax.bar(x - 0.5 * width, [r["net_sr"] for r in rows], width, label="Net Sharpe")
    ax.bar(x + 0.5 * width, [r["avg_turnover"] for r in rows], width, label="Avg Turnover")
    ax.bar(x + 1.5 * width, [r["max_dd"] for r in rows], width, label="Max DD (net)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    _save(fig, out_path)

# =============================================================================
# Set 1: TC ablation for each method group
# =============================================================================

def plot_tc_ablation(label: str, bt: pd.DataFrame, color_base: str, out_dir: Path) -> None:
    a1 = select_static_no_tc(bt)
    a2 = select_static_stock_tc(bt)
    b = select_rolling_portfolio_tc(bt)
    c = select_rolling_stock_tc(bt)
    spy = get_spy(bt)

    pieces_raw = []
    labels_colors_raw = []

    if not a1.empty:
        pieces_raw.append(a1)
        labels_colors_raw.append((f"{label} static no TC", COLOR_STATIC_NO_TC))
    if not a2.empty:
        pieces_raw.append(a2)
        labels_colors_raw.append((f"{label} static stock TC", COLOR_STATIC_TC))
    if not b.empty:
        pieces_raw.append(b)
        labels_colors_raw.append((f"{label} rolling portfolio TC", COLOR_ROLLING_PORT_TC))
    if not c.empty:
        pieces_raw.append(c)
        labels_colors_raw.append((f"{label} rolling stock TC", COLOR_ROLLING_STOCK_TC))
    if not spy.empty:
        pieces_raw.append(spy)
        labels_colors_raw.append(("S&P 500", COLOR_SPY))

    if len(pieces_raw) < 2:
        print(f"  WARNING: not enough variants for {label}; skipping TC ablation.")
        return

    pieces = align_dates(*pieces_raw)
    labels_colors = labels_colors_raw[: len(pieces)]

    plot_cumulative(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        f"Cumulative Net Wealth — {label} TC Ablation",
        "Cumulative Net Wealth",
        out_dir / "plot_1_cumulative_net.png",
    )

    plot_drawdown_net(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        f"Drawdown — {label} TC Ablation",
        out_dir / "plot_2_drawdown.png",
    )

    turn_pairs = [
        (lc[0], _date_indexed(p, "turnover"), lc[1])
        for p, lc in zip(pieces, labels_colors)
        if "S&P" not in lc[0] and HF_NAME not in lc[0]
    ]
    plot_turnover(turn_pairs, f"Turnover — {label} TC Ablation", out_dir / "plot_3_turnover.png")

    rows = [_build_row(lc[0], p) for p, lc in zip(pieces, labels_colors)]
    plot_summary_bar(rows, f"Summary Metrics — {label} TC Ablation", out_dir / "plot_4_summary.png")

# =============================================================================
# Set 2: static comparison
# =============================================================================

def plot_static_cross_method(bt_ts: pd.DataFrame, bt_ap: pd.DataFrame, bt_ap_rm: pd.DataFrame, hf: pd.DataFrame, out_dir: Path) -> None:
    ts32 = select_static_comparison_variant(filter_method_group(bt_ts, "TS32"))
    ts64 = select_static_comparison_variant(filter_method_group(bt_ts, "TS64"))
    ap = select_static_comparison_variant(filter_method_group(bt_ap, "AP"))
    ap_rm = select_static_comparison_variant(filter_method_group(bt_ap_rm, "AP_RM"))
    spy = get_spy(bt_ap)

    pieces_raw = [ts32, ts64, ap, ap_rm]
    labels_colors_raw = [
        ("TS32 static", COLOR_TS32),
        ("TS64 static", COLOR_TS64),
        ("AP-Trees static", COLOR_AP),
        ("AP-Trees + RM static", COLOR_AP_RM),
    ]

    if not spy.empty:
        pieces_raw.append(spy)
        labels_colors_raw.append(("S&P 500", COLOR_SPY))
    if hf is not None and not hf.empty:
        pieces_raw.append(hf)
        labels_colors_raw.append((HF_NAME, COLOR_HF))

    if any(p.empty for p in pieces_raw[:4]):
        print("  WARNING: missing TS32/TS64/AP/AP+RM static variants; skipping static comparison.")
        for label, p in labels_colors_raw[:4]:
            print(f"    {label}: {len(p)} rows")
        return

    pieces = align_dates(*pieces_raw)
    labels_colors = labels_colors_raw[: len(pieces)]

    plot_cumulative(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        "Cumulative Net Wealth — Static Cross-Method Comparison",
        "Net Wealth",
        out_dir / "plot_1_cumulative_net.png",
    )

    plot_drawdown_net(
        [(lc[0], _date_indexed(p, "net_ret"), lc[1]) for p, lc in zip(pieces, labels_colors)],
        "Drawdown — Static Cross-Method Comparison",
        out_dir / "plot_2_drawdown.png",
    )

    rows = [_build_row(lc[0], p) for p, lc in zip(pieces, labels_colors)]
    plot_summary_bar(rows, "Summary Metrics — Static Cross-Method Comparison", out_dir / "plot_3_summary.png")

# =============================================================================
# Set 3: rolling/preferred combined comparison
# =============================================================================

def plot_combined(bt_ts: pd.DataFrame, bt_ap: pd.DataFrame, bt_ap_rm: pd.DataFrame, hf: pd.DataFrame, out_dir: Path) -> None:
    ts32 = select_preferred_variant(filter_method_group(bt_ts, "TS32"))
    ts64 = select_preferred_variant(filter_method_group(bt_ts, "TS64"))
    ap = select_preferred_variant(filter_method_group(bt_ap, "AP"))
    ap_rm = select_preferred_variant(filter_method_group(bt_ap_rm, "AP_RM"))
    spy = get_spy(bt_ap)

    pieces_raw = [ts32, ts64, ap, ap_rm]
    labels_colors_raw = [
        ("TS32 rolling TC-aware", COLOR_TS32),
        ("TS64 rolling TC-aware", COLOR_TS64),
        ("AP-Trees rolling TC-aware", COLOR_AP),
        ("AP-Trees + RM rolling TC-aware", COLOR_AP_RM),
    ]

    if not spy.empty:
        pieces_raw.append(spy)
        labels_colors_raw.append(("S&P 500", COLOR_SPY))
    if hf is not None and not hf.empty:
        pieces_raw.append(hf)
        labels_colors_raw.append((HF_NAME, COLOR_HF))

    if any(p.empty for p in pieces_raw[:4]):
        print("  WARNING: missing TS32/TS64/AP/AP+RM preferred variants; skipping combined comparison.")
        for label, p in labels_colors_raw[:4]:
            print(f"    {label}: {len(p)} rows")
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
        for p, lc in zip(pieces[:4], labels_colors[:4])
    ]
    plot_turnover(turn_pairs, "Turnover — Combined Comparison", out_dir / "plot_3_turnover.png")

    rows = [_build_row(lc[0], p) for p, lc in zip(pieces, labels_colors)]
    plot_summary_bar(rows, "Summary Metrics — Combined Comparison", out_dir / "plot_4_summary.png")

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
            continue
        bt[key] = load_backtest(path)
        print(f"  {fname}: {len(bt[key])} rows")
        for m in bt[key]["method"].drop_duplicates().tolist():
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
                print("  WARNING: HF index has no overlapping dates.")
        except Exception as e:
            print(f"  WARNING: could not load HF index: {e}")
            hf = pd.DataFrame()
    else:
        print("  WARNING: NAVROR.csv not found — HF benchmark omitted.")

    print("\n=== Set 1: TC ablation per method group ===")
    if not bt.get("ts", pd.DataFrame()).empty:
        plot_tc_ablation("TS32", filter_method_group(bt["ts"], "TS32"), COLOR_TS32, PLOT_DIR / "tc_ablation_ts32")
        plot_tc_ablation("TS64", filter_method_group(bt["ts"], "TS64"), COLOR_TS64, PLOT_DIR / "tc_ablation_ts64")
    if not bt.get("ap", pd.DataFrame()).empty:
        plot_tc_ablation("AP-Trees", filter_method_group(bt["ap"], "AP"), COLOR_AP, PLOT_DIR / "tc_ablation_ap")
    if not bt.get("ap_rm", pd.DataFrame()).empty:
        plot_tc_ablation("AP-Trees + RM", filter_method_group(bt["ap_rm"], "AP_RM"), COLOR_AP_RM, PLOT_DIR / "tc_ablation_ap_rm")

    print("\n=== Set 2: Static cross-method comparison ===")
    if not any(bt.get(k, pd.DataFrame()).empty for k in ["ts", "ap", "ap_rm"]):
        plot_static_cross_method(bt["ts"], bt["ap"], bt["ap_rm"], hf, PLOT_DIR / "static_cross_method")
    else:
        print("  Skipping — not all three method files present.")

    print("\n=== Set 3: Rolling/preferred combined comparison ===")
    if not any(bt.get(k, pd.DataFrame()).empty for k in ["ts", "ap", "ap_rm"]):
        plot_combined(bt["ts"], bt["ap"], bt["ap_rm"], hf, PLOT_DIR / "combined")
    else:
        print("  Skipping — not all three method files present.")

    print(f"\nDone. All plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
