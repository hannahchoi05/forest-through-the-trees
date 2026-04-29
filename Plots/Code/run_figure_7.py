"""
Runner: builds Table D.1 + Figure 7 from your real backtest outputs.

Usage:
    python run_figure_7.py
        --backtest-root  ./backtest_results
        --factors        ./tradable_factors.csv
        --out-dir        ./figures

Edit the METHOD_PATTERNS dict below to match the exact `method` strings your
pipeline emits in backtest_comparison.csv. The aggregator picks rows by
substring match.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aggregate_table_d1 import (
    AggregatorConfig, aggregate, to_table_d1_wide,
)
from plot_figure_7 import plot_sr


# === EDIT ME =================================================================
# Map paper-friendly method labels to substrings in your pipeline's `method`
# column. The aggregator uses these substrings to identify which rows of the
# backtest CSV correspond to which method. If your pipeline emits something
# different — e.g. "AP-Trees baseline (static, no TC)" — update accordingly.
METHOD_PATTERNS = {
    "AP-Trees": "AP-Trees AP-pruning (static, no TC)",
    "TS":       "Triple Sort static (no TC)",
}
# === END EDIT ME =============================================================


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backtest-root", type=Path, required=True,
                   help="Directory containing one subfolder per cross-section, "
                        "each with backtest_comparison.csv.")
    p.add_argument("--factors", type=Path, required=True,
                   help="Path to tradable_factors.csv.")
    p.add_argument("--out-dir", type=Path, default=Path("./figures"),
                   help="Where to save the plot and Table D.1 CSVs.")
    p.add_argument("--test-start", default="1994-01-01")
    p.add_argument("--test-end",   default="2016-12-31")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AggregatorConfig(
        backtest_root=args.backtest_root,
        factors_path=args.factors,
        method_patterns=METHOD_PATTERNS,
        test_start=__import__("pandas").Timestamp(args.test_start),
        test_end=__import__("pandas").Timestamp(args.test_end),
    )

    long_df = aggregate(cfg)
    if long_df.empty:
        print("No backtest data found. Check --backtest-root path and "
              "ensure each cross-section subfolder has backtest_comparison.csv")
        return

    # Save both long and wide forms.
    long_path = args.out_dir / "table_d1_long.csv"
    wide_path = args.out_dir / "table_d1_wide.csv"
    long_df.to_csv(long_path, index=False)
    wide = to_table_d1_wide(long_df)
    wide.to_csv(wide_path, index=False)
    print(f"\nSaved Table D.1 long  -> {long_path}")
    print(f"Saved Table D.1 wide  -> {wide_path}")

    # Print a preview.
    print("\nTable D.1 (wide):")
    print(wide[["cs_id", "cs_key"] +
               [c for c in wide.columns if c.startswith("SR_")]].to_string(index=False))

    # Figure 7: sorted by AP-Trees SR.
    plot_sr(
        long_df,
        out_path=args.out_dir / "figure_7.png",
        sort_by_method="AP-Trees",
        methods_to_plot=("AP-Trees", "TS"),
    )


if __name__ == "__main__":
    main()