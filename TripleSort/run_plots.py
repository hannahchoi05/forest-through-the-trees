from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


def _bootstrap_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_bootstrap_imports()

from config import OUTPUT_DIR, DEFAULT_CHARS  # noqa: E402
from plots import make_all_plots  # noqa: E402


def main() -> None:
    subdir = "_".join(DEFAULT_CHARS)
    out_dir = OUTPUT_DIR / subdir
    plot_dir = out_dir / "plots_lagged_trade"

    backtest_path = out_dir / "backtest_comparison_lagged_trade.csv"
    if not backtest_path.exists():
        raise FileNotFoundError(f"Missing lagged-trade backtest file: {backtest_path}")

    backtest = pd.read_csv(backtest_path)

    methods = set(backtest["method"].astype(str).unique()) if "method" in backtest else set()

    want_options = {
        "S&P 500 (SPY adjusted close)",
        "S&P 500 (adjusted close)",
    }
    proxy_markers = ("proxy", "Mkt-RF")

    if not any(w in methods for w in want_options):
        proxy_present = any(any(m in str(method) for m in proxy_markers) for method in methods)
        if proxy_present:
            raise RuntimeError(
                f"Found only proxy benchmark in {backtest_path}. "
                "Re-run the lagged triple-sort optimizer to regenerate the backtest with SPY."
            )
        raise RuntimeError(
            f"Missing SPY benchmark in {backtest_path}. "
            "Re-run the lagged triple-sort optimizer."
        )

    backtest = backtest[
        ~backtest["method"].astype(str).str.contains("proxy|Mkt-RF", regex=True, na=False)
    ].copy()

    plot_dir.mkdir(parents=True, exist_ok=True)
    for png in plot_dir.glob("*.png"):
        png.unlink()

    make_all_plots(backtest, plot_dir)
    print(f"Plots saved to: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()