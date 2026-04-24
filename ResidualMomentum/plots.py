
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from metrics import add_wealth_drawdown, performance_metrics
from utils import safe_method_name


def _x(g: pd.DataFrame):
    if "date_dt" in g.columns:
        return pd.to_datetime(g["date_dt"])
    return pd.to_datetime(
        g["yy"].astype(int).astype(str) + "-" + g["mm"].astype(int).astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)


def plot_cumulative_gross(backtest: pd.DataFrame, out_dir: Path):
    d = add_wealth_drawdown(backtest)

    plt.figure(figsize=(10, 6))

    for method, g in d.groupby("method", sort=False):
        plt.plot(_x(g), g["wealth_gross"], label=method)

    plt.title("Plot A: Cumulative Gross Returns")
    plt.ylabel("Gross wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_A_cumulative_gross.png", dpi=200)
    plt.close()


def plot_cumulative_net(backtest: pd.DataFrame, out_dir: Path):
    d = add_wealth_drawdown(backtest)

    plt.figure(figsize=(10, 6))

    for method, g in d.groupby("method", sort=False):
        plt.plot(_x(g), g["wealth_net"], label=method)

    plt.title("Plot B: Cumulative Net Returns")
    plt.ylabel("Net wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_B_cumulative_net.png", dpi=200)
    plt.close()


def plot_gross_vs_net(backtest: pd.DataFrame, out_dir: Path):
    d = add_wealth_drawdown(backtest)

    for method, g in d.groupby("method", sort=False):
        plt.figure(figsize=(10, 6))

        plt.plot(_x(g), g["wealth_gross"], label="Gross")
        plt.plot(_x(g), g["wealth_net"], label="Net")

        plt.title(f"Plot C: Gross vs Net - {method}")
        plt.ylabel("Wealth")

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            out_dir / f"plot_C_gross_vs_net_{safe_method_name(method)}.png",
            dpi=200,
        )
        plt.close()


def plot_drawdown(backtest: pd.DataFrame, out_dir: Path):
    d = add_wealth_drawdown(backtest)

    plt.figure(figsize=(10, 6))

    for method, g in d.groupby("method", sort=False):
        plt.plot(_x(g), g["drawdown_net"], label=method)

    plt.title("Plot D: Drawdown (Net)")
    plt.ylabel("Drawdown")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_D_drawdown_net.png", dpi=200)
    plt.close()


def plot_turnover(backtest: pd.DataFrame, out_dir: Path, rolling: int = 12):
    plt.figure(figsize=(10, 6))

    for method, g in backtest.groupby("method", sort=False):
        h = g.copy().sort_values(["yy", "mm"])
        plt.plot(
            _x(h),
            h["turnover"].rolling(rolling, min_periods=1).mean(),
            label=method,
        )

    plt.title(f"Plot E: Turnover ({rolling}-month MA)")
    plt.ylabel("Turnover")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_E_turnover.png", dpi=200)
    plt.close()


def plot_summary_metrics(backtest: pd.DataFrame, out_dir: Path):
    m = performance_metrics(backtest).set_index("method")

    cols = [
        "sharpe_gross_ann",
        "sharpe_net_ann",
        "avg_turnover",
        "max_drawdown_net",
    ]

    ax = m[cols].plot(kind="bar", figsize=(11, 6))

    ax.set_title("Plot F: Summary Metrics")
    ax.set_ylabel("Value")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_F_summary_metrics.png", dpi=200)
    plt.close()


def make_all_plots(backtest: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cumulative_gross(backtest, out_dir)
    plot_cumulative_net(backtest, out_dir)
    plot_gross_vs_net(backtest, out_dir)
    plot_drawdown(backtest, out_dir)
    plot_turnover(backtest, out_dir)
    plot_summary_metrics(backtest, out_dir)
