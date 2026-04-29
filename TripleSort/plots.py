from __future__ import annotations

from pathlib import Path
import os
import tempfile
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from metrics import add_wealth_drawdown, performance_metrics
from utils import safe_method_name


def _x(g: pd.DataFrame):
    if "date_dt" in g.columns:
        return pd.to_datetime(g["date_dt"])
    return pd.to_datetime(
        g["yy"].astype(int).astype(str) + "-" + g["mm"].astype(int).astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)


def _short_label(method: str) -> str:
    s = str(method)

    s = s.replace("Triple Sort ", "TS ")
    s = s.replace("rolling TC-aware + portfolio-level TC", "rolling portfolio TC")
    s = s.replace("rolling TC-aware + stock-level TC", "rolling stock TC")
    s = s.replace("static + stock-level TC", "static stock TC")
    s = s.replace("static (no TC)", "static no TC")
    s = s.replace("S&P 500 (SPY adjusted close)", "S&P 500")

    return s


def _method_order(method: str) -> tuple:
    s = str(method).lower()

    if "ts32" in s:
        base = 0
    elif "ts64" in s:
        base = 1
    elif "ap-tree + rm" in s or "ap-trees + rm" in s:
        base = 2
    elif "ap-tree" in s or "ap-trees" in s:
        base = 3
    elif "s&p" in s or "spy" in s:
        base = 9
    else:
        base = 8

    if "static" in s and "no tc" in s:
        variant = 0
    elif "static" in s and "stock" in s:
        variant = 1
    elif "rolling" in s and "portfolio" in s:
        variant = 2
    elif "rolling" in s and "stock" in s:
        variant = 3
    else:
        variant = 9

    return (base, variant, s)


def _ordered_groups(df: pd.DataFrame):
    methods = sorted(df["method"].dropna().unique(), key=_method_order)
    for method in methods:
        yield method, df[df["method"] == method].copy().sort_values(["yy", "mm"])


def plot_cumulative_gross(d: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(12, 7))

    for method, g in _ordered_groups(d):
        plt.plot(_x(g), g["wealth_gross"], label=_short_label(method))

    plt.title("Plot A: Cumulative Gross Returns")
    plt.ylabel("Gross wealth")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_A_cumulative_gross.png", dpi=200)
    plt.close()


def plot_cumulative_net(d: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(12, 7))

    for method, g in _ordered_groups(d):
        plt.plot(_x(g), g["wealth_net"], label=_short_label(method))

    plt.title("Plot B: Cumulative Net Returns")
    plt.ylabel("Net wealth")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_B_cumulative_net.png", dpi=200)
    plt.close()


def plot_gross_vs_net(d: pd.DataFrame, out_dir: Path):
    for method, g in _ordered_groups(d):
        plt.figure(figsize=(10, 6))

        plt.plot(_x(g), g["wealth_gross"], label="Gross")
        plt.plot(_x(g), g["wealth_net"], label="Net")

        plt.title(f"Plot C: Gross vs Net - {_short_label(method)}")
        plt.ylabel("Wealth")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            out_dir / f"plot_C_gross_vs_net_{safe_method_name(_short_label(method))}.png",
            dpi=200,
        )
        plt.close()


def plot_drawdown(d: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(12, 7))

    for method, g in _ordered_groups(d):
        plt.plot(_x(g), g["drawdown_net"], label=_short_label(method))

    plt.title("Plot D: Drawdown (Net)")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_D_drawdown_net.png", dpi=200)
    plt.close()


def plot_turnover(backtest: pd.DataFrame, out_dir: Path, rolling: int = 12):
    plt.figure(figsize=(12, 7))

    for method, g in _ordered_groups(backtest):
        if "s&p" in str(method).lower() or "spy" in str(method).lower():
            continue

        h = g.copy().sort_values(["yy", "mm"])
        plt.plot(
            _x(h),
            h["turnover"].astype(float).rolling(rolling, min_periods=1).mean(),
            label=_short_label(method),
        )

    plt.title(f"Plot E: Turnover ({rolling}-month MA)")
    plt.ylabel("Turnover")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_E_turnover.png", dpi=200)
    plt.close()


def plot_summary_metrics(metrics: pd.DataFrame, out_dir: Path):
    metrics = metrics.copy()
    metrics["plot_label"] = metrics["method"].map(_short_label)
    metrics["_order"] = metrics["method"].map(_method_order)
    metrics = metrics.sort_values("_order")

    m = metrics.set_index("plot_label")

    cols = [
        "sharpe_gross_ann",
        "sharpe_net_ann",
        "avg_turnover",
        "max_drawdown_net",
    ]

    ax = m[cols].plot(kind="bar", figsize=(max(12, len(m) * 1.8), 6))

    ax.set_title("Plot F: Summary Metrics")
    ax.set_ylabel("Value")
    ax.legend(
        ["Gross Sharpe", "Net Sharpe", "Avg Turnover", "Max DD (net)"],
        fontsize=8,
    )

    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_F_summary_metrics.png", dpi=200)
    plt.close()


def make_all_plots(backtest: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    backtest = backtest.copy()
    if "date_dt" in backtest.columns:
        backtest["date_dt"] = pd.to_datetime(backtest["date_dt"])

    # Align all methods to common date range
    common_start = backtest.groupby("method")["date_dt"].min().max()
    common_end = backtest.groupby("method")["date_dt"].max().min()

    backtest = backtest[
        (backtest["date_dt"] >= common_start)
        & (backtest["date_dt"] <= common_end)
    ].copy()

    d = add_wealth_drawdown(backtest)
    metrics = performance_metrics(backtest)

    plot_cumulative_gross(d, out_dir)
    plot_cumulative_net(d, out_dir)
    plot_gross_vs_net(d, out_dir)
    plot_drawdown(d, out_dir)
    plot_turnover(backtest, out_dir)
    plot_summary_metrics(metrics, out_dir)