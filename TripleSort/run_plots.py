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
    plot_dir = out_dir / "plots"

    backtest_path = out_dir / "backtest_comparison.csv"
    if not backtest_path.exists():
        raise FileNotFoundError(f"Missing backtest file: {backtest_path}")

    backtest = pd.read_csv(backtest_path)

    # Enforce the new benchmark naming and avoid plotting stale proxy-series outputs.
    methods = set(backtest["method"].astype(str).unique()) if "method" in backtest else set()
    want = "S&P 500 (SPY adjusted close)"
    proxy_markers = ("proxy", "Mkt-RF")

    if want not in methods:
        proxy_present = any(any(m in str(method) for m in proxy_markers) for method in methods)
        if proxy_present:
            raise RuntimeError(
                f"Found only proxy benchmark in {backtest_path}. "
                f"Re-run TripleSort/run_all.py to regenerate the backtest with '{want}'."
            )
        raise RuntimeError(
            f"Missing benchmark '{want}' in {backtest_path}. "
            "Re-run TripleSort/run_all.py."
        )

    # If both exist (e.g., mixed old/new), drop proxy rows for clean labeling.
    backtest = backtest[
        ~backtest["method"].astype(str).str.contains("proxy|Mkt-RF", regex=True)
    ].copy()

    # Clear old plot images so filenames/labels don't look stale.
    plot_dir.mkdir(parents=True, exist_ok=True)
    for png in plot_dir.glob("*.png"):
        png.unlink()

    make_all_plots(backtest, plot_dir)
    print(f"Plots saved to: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
