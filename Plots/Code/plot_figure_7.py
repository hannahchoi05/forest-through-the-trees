"""
Figure 7: out-of-sample monthly Sharpe ratios of SDFs spanned by AP-Trees,
TS(32), and TS(64), with cross-sections sorted on the x-axis by the
SR achieved with AP-Trees.

Same plotter works for Figure 9 — just point it at the AP-Trees-10 SR column
instead of AP-Trees-40, and re-sort.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ggplot2-style colors used in the paper, matched to method labels.
_COLORS = {
    "AP-Trees": "#F8766D",   # red
    "TS32":     "#00BA38",   # green
    "TS64":     "#619CFF",   # blue
}
_LINESTYLES = {
    "AP-Trees": "solid",
    "TS32":     (0, (5, 2, 1, 2)),
    "TS64":     (0, (3, 2)),
}
_MARKERS = {
    "AP-Trees": "o",
    "TS32":     "^",
    "TS64":     "s",
}
_PRETTY_LABELS = {
    "AP-Trees": "AP-Trees",
    "TS32":     "Triple Sort (32)",
    "TS64":     "Triple Sort (64)",
}


def plot_sr(
    long_df: pd.DataFrame,
    out_path: Path | str,
    sort_by_method: str = "AP-Trees",
    methods_to_plot: tuple = ("AP-Trees", "TS32", "TS64"),
    title: str | None = None,
):
    """
    Render Figure 7 (or Figure 9) from the aggregator's long-format dataframe.

    Parameters
    ----------
    long_df : pd.DataFrame
        Output of aggregate(). Expected columns: cs_id, cs_key, method, SR.
    out_path : Path
        Where to save the PNG.
    sort_by_method : str
        Method whose SR defines the x-axis ordering. Use 'AP-Trees-40' for
        Figure 7, 'AP-Trees-10' for Figure 9.
    methods_to_plot : tuple
        Methods to draw as separate lines.
    title : str, optional
        Plot title. None for no title (paper style).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pivot to wide so we can sort and grab series easily.
    wide = long_df.pivot(index=["cs_id", "cs_key"], columns="method", values="SR")

    # Sort the cross-sections by the SR achieved with sort_by_method.
    if sort_by_method not in wide.columns:
        raise ValueError(
            f"sort_by_method='{sort_by_method}' not in available methods "
            f"{list(wide.columns)}"
        )
    sort_series = wide[sort_by_method].copy()
    if sort_series.isna().all():
        raise ValueError(
            f"All SR values for {sort_by_method} are NaN — can't sort. "
            f"Have you produced backtest results yet?"
        )
    wide = wide.assign(_sort=sort_series).sort_values("_sort").drop(columns="_sort")

    # X-axis labels: use the cs_id (the paper's numbering).
    cs_ids = wide.index.get_level_values("cs_id").tolist()
    x = np.arange(len(cs_ids))

    fig, ax = plt.subplots(figsize=(14.7, 5.7))
    for method in methods_to_plot:
        if method not in wide.columns:
            print(f"WARNING: method '{method}' not in data — skipping line")
            continue
        y = wide[method].to_numpy(dtype=float)
        ax.plot(
            x, y,
            color=_COLORS.get(method, None),
            linestyle=_LINESTYLES.get(method, "solid"),
            marker=_MARKERS.get(method, "o"),
            markersize=8,
            linewidth=2.0,
            label=_PRETTY_LABELS.get(method, method),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in cs_ids], fontsize=12)
    ax.set_xlabel("Cross-sections", fontsize=14)
    ax.set_ylabel("Monthly Sharpe Ratio (SR)", fontsize=14)

    # Paper style: clean, no grid, frameless legend on the right.
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    ax.tick_params(labelsize=12)

    ax.legend(
        title="Basis portfolios:",
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=False, fontsize=12, title_fontsize=12,
    )
    if title:
        ax.set_title(title, fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path