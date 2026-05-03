from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cross_sections import CROSS_SECTIONS, CHAR_NAME_MAP, CrossSection
from aggregate_table_d1 import AggregatorConfig, aggregate


METHOD_PATTERNS = {
    "AP-Trees": "AP-Trees baseline (static, no TC)",
    "TS32": "Triple Sort (32) static (no TC)",
    "TS64": "Triple Sort (64) static (no TC)",
}


FF5_FALLBACK = ["Mkt-RF", "LME", "BEME", "OP", "Investment"]


def _triple_dir_name(cs: CrossSection) -> str:
    return "_".join(CHAR_NAME_MAP[c] for c in (cs.char1, cs.char2, cs.char3))


def _parse_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    if s.str.match(r"^\d{6}$").all():
        return pd.to_datetime(s, format="%Y%m") + pd.offsets.MonthEnd(0)
    return pd.to_datetime(series)


def _select_ff5_cols(factors: pd.DataFrame) -> list[str]:
    canonical = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    if all(c in factors.columns for c in canonical):
        return canonical
    if all(c in factors.columns for c in FF5_FALLBACK):
        return FF5_FALLBACK
    raise ValueError(
        "Could not identify FF5 columns. Expected either "
        f"{canonical} or {FF5_FALLBACK}."
    )


def _adj_r2(y: np.ndarray, X: np.ndarray) -> float:
    n = len(y)
    k = X.shape[1]
    if n <= k + 1:
        return np.nan

    X1 = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    sse = float(resid @ resid)
    y_centered = y - y.mean()
    sst = float(y_centered @ y_centered)
    if sst <= 0:
        return np.nan

    r2 = 1.0 - sse / sst
    return 1.0 - (1.0 - r2) * (n - 1.0) / (n - k - 1.0)


def _mean_adj_r2(asset_df: pd.DataFrame, factors_df: pd.DataFrame, ff5_cols: list[str]) -> tuple[float, int]:
    merged = pd.merge(
        asset_df.reset_index(),
        factors_df.reset_index(),
        on="date_dt",
        how="inner",
    )
    if merged.empty:
        return np.nan, 0

    X = merged[ff5_cols].to_numpy(dtype=float)
    vals: list[float] = []
    used = 0
    for col in asset_df.columns:
        y = merged[col].to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            continue
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < len(ff5_cols) + 2:
            continue
        r2a = _adj_r2(y[mask], X[mask, :])
        if np.isfinite(r2a):
            vals.append(r2a)
            used += 1
    if not vals:
        return np.nan, 0
    return float(np.mean(vals)), used


def _load_ap_basis(ap_dir: Path, max_assets: int) -> pd.DataFrame:
    df = pd.read_csv(ap_dir / "candidate_returns.csv")
    df["date_dt"] = pd.to_datetime(df["date_dt"])
    non_meta = [c for c in df.columns if c not in {"date", "date_dt", "yy", "mm"}]
    if not non_meta:
        return pd.DataFrame(index=df["date_dt"])

    # Stable subset so the AP panel is bounded to 40 basis assets.
    chosen = sorted(non_meta)[:max_assets]
    out = df[["date_dt"] + chosen].copy()
    out = out.set_index("date_dt")
    return out


def _load_ts_basis(path: Path, n_assets: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in df.columns][:n_assets]
    out = df[cols].copy()

    # Triple-sort portfolio files do not store explicit dates; use the project
    # monthly span (1964-01 .. 2016-12), then trim to the requested test window.
    idx = pd.date_range("1964-01-31", periods=len(out), freq="ME")
    out.insert(0, "date_dt", idx)
    return out.set_index("date_dt")


def build_figure8_panels(
    backtest_root: Path,
    factors_path: Path,
    ap_output_root: Path,
    ts32_root: Path,
    ts64_root: Path,
    test_start: str,
    test_end: str,
    ap_basis_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = AggregatorConfig(
        backtest_root=backtest_root,
        factors_path=factors_path,
        method_patterns=METHOD_PATTERNS,
        test_start=pd.Timestamp(test_start),
        test_end=pd.Timestamp(test_end),
    )
    long_df = aggregate(cfg)

    sr_order = (
        long_df[long_df["method"] == "AP-Trees"][["cs_id", "cs_key", "SR"]]
        .rename(columns={"SR": "SR_AP"})
        .sort_values("SR_AP", ascending=True)
        .reset_index(drop=True)
    )
    sr_order["rank"] = np.arange(1, len(sr_order) + 1)

    tstats = long_df[["cs_id", "cs_key", "method", "tstat_FF5"]].copy()
    tstats = tstats.merge(sr_order[["cs_id", "rank", "SR_AP"]], on="cs_id", how="left")
    tstats = tstats.sort_values(["rank", "method"]).reset_index(drop=True)

    factors = pd.read_csv(factors_path)
    date_col = "Date" if "Date" in factors.columns else "date"
    factors["date_dt"] = _parse_dates(factors[date_col])
    factors = factors.set_index("date_dt").sort_index()
    ff5_cols = _select_ff5_cols(factors)
    factors = factors.loc[(factors.index >= pd.Timestamp(test_start)) & (factors.index <= pd.Timestamp(test_end))]

    r2_rows = []
    for cs in CROSS_SECTIONS:
        triple_dir = _triple_dir_name(cs)

        ap_basis = _load_ap_basis(ap_output_root / triple_dir, max_assets=ap_basis_count)
        ap_basis = ap_basis.loc[(ap_basis.index >= pd.Timestamp(test_start)) & (ap_basis.index <= pd.Timestamp(test_end))]
        ap_r2, ap_n = _mean_adj_r2(ap_basis, factors, ff5_cols)

        ts32_basis = _load_ts_basis(ts32_root / triple_dir / "excess_ports.csv", n_assets=32)
        ts32_basis = ts32_basis.loc[(ts32_basis.index >= pd.Timestamp(test_start)) & (ts32_basis.index <= pd.Timestamp(test_end))]
        ts32_r2, ts32_n = _mean_adj_r2(ts32_basis, factors, ff5_cols)

        ts64_basis = _load_ts_basis(ts64_root / triple_dir / "excess_ports.csv", n_assets=64)
        ts64_basis = ts64_basis.loc[(ts64_basis.index >= pd.Timestamp(test_start)) & (ts64_basis.index <= pd.Timestamp(test_end))]
        ts64_r2, ts64_n = _mean_adj_r2(ts64_basis, factors, ff5_cols)

        r2_rows.extend([
            {"cs_id": cs.id, "cs_key": cs.key, "method": "AP-Trees", "adj_r2": ap_r2, "n_assets_used": ap_n},
            {"cs_id": cs.id, "cs_key": cs.key, "method": "TS32", "adj_r2": ts32_r2, "n_assets_used": ts32_n},
            {"cs_id": cs.id, "cs_key": cs.key, "method": "TS64", "adj_r2": ts64_r2, "n_assets_used": ts64_n},
        ])

    r2_df = pd.DataFrame(r2_rows)
    r2_df = r2_df.merge(sr_order[["cs_id", "rank", "SR_AP"]], on="cs_id", how="left")
    r2_df = r2_df.sort_values(["rank", "method"]).reset_index(drop=True)

    return tstats, r2_df


def plot_figure8(tstats: pd.DataFrame, r2_df: pd.DataFrame, out_path: Path) -> None:
    color = {
        "AP-Trees": "#1f77b4",
        "TS32": "#d62728",
        "TS64": "#2ca02c",
    }

    order = tstats[tstats["method"] == "AP-Trees"].sort_values("rank")
    x_ticks = order["rank"].to_numpy()
    x_labels = order["cs_id"].astype(str).to_list()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for m in ["AP-Trees", "TS32", "TS64"]:
        d = tstats[tstats["method"] == m].sort_values("rank")
        axes[0].plot(d["rank"], d["tstat_FF5"], marker="o", markersize=3, linewidth=1.5, label=m, color=color[m])
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[0].set_xlabel("Cross-sections")
    axes[0].set_ylabel("t-stat")
    axes[0].set_title("(a) t-stat of the robust SDF alpha (w.r.t the Fama-French 5 factor model)")
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(x_labels)
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")

    for m in ["AP-Trees", "TS32", "TS64"]:
        d = r2_df[r2_df["method"] == m].sort_values("rank")
        axes[1].plot(d["rank"], d["adj_r2"], marker="o", markersize=3, linewidth=1.5, label=m, color=color[m])
    axes[1].set_xlabel("Cross-sections")
    axes[1].set_ylabel("R2 Adjusted")
    axes[1].set_title("(b) R2 within cross-sections (w.r.t the Fama-French 5 factor model)")
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_labels)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_figure8a(tstats: pd.DataFrame, out_path: Path) -> None:
    color = {
        "AP-Trees": "#1f77b4",
        "TS32": "#d62728",
        "TS64": "#2ca02c",
    }
    order = tstats[tstats["method"] == "AP-Trees"].sort_values("rank")
    x_ticks = order["rank"].to_numpy()
    x_labels = order["cs_id"].astype(str).to_list()

    fig, ax = plt.subplots(1, 1, figsize=(14, 4.8))
    for m in ["AP-Trees", "TS32", "TS64"]:
        d = tstats[tstats["method"] == m].sort_values("rank")
        ax.plot(d["rank"], d["tstat_FF5"], marker="o", markersize=3, linewidth=1.5, label=m, color=color[m])
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Cross-sections")
    ax.set_ylabel("t-stat")
    ax.set_title("(a): t-stat of the robust SDF alpha (w.r.t the Fama-French 5 factor model)")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_figure8b(r2_df: pd.DataFrame, out_path: Path) -> None:
    color = {
        "AP-Trees": "#1f77b4",
        "TS32": "#d62728",
        "TS64": "#2ca02c",
    }
    order = r2_df[r2_df["method"] == "AP-Trees"].sort_values("rank")
    x_ticks = order["rank"].to_numpy()
    x_labels = order["cs_id"].astype(str).to_list()

    fig, ax = plt.subplots(1, 1, figsize=(14, 4.8))
    for m in ["AP-Trees", "TS32", "TS64"]:
        d = r2_df[r2_df["method"] == m].sort_values("rank")
        ax.plot(d["rank"], d["adj_r2"], marker="o", markersize=3, linewidth=1.5, label=m, color=color[m])
    ax.set_xlabel("Cross-sections")
    ax.set_ylabel("R2-Adjusted")
    ax.set_title("(b): R2 within cross-sections (w.r.t the Fama-French 5 factor model)")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backtest-root", type=Path, required=True)
    p.add_argument("--factors", type=Path, required=True)
    p.add_argument("--ap-output-root", type=Path, default=Path("ap-trees/outputs"))
    p.add_argument("--ts32-root", type=Path, default=Path("TripleSort/ts_portfolio_py"))
    p.add_argument("--ts64-root", type=Path, default=Path("TripleSort/ts64_portfolio_py"))
    p.add_argument("--out-dir", type=Path, default=Path("Plots/Code/figures"))
    p.add_argument("--test-start", default="1994-01-01")
    p.add_argument("--test-end", default="2016-12-31")
    p.add_argument("--ap-basis-count", type=int, default=40)
    args = p.parse_args()

    tstats, r2_df = build_figure8_panels(
        backtest_root=args.backtest_root,
        factors_path=args.factors,
        ap_output_root=args.ap_output_root,
        ts32_root=args.ts32_root,
        ts64_root=args.ts64_root,
        test_start=args.test_start,
        test_end=args.test_end,
        ap_basis_count=args.ap_basis_count,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tstats_path = out_dir / "figure_8a_tstats_ff5.csv"
    r2_path = out_dir / "figure_8b_adj_r2.csv"
    fig_path = out_dir / "figure_8.png"
    fig8a_path = out_dir / "figure_8a.png"
    fig8b_path = out_dir / "figure_8b.png"

    tstats.to_csv(tstats_path, index=False)
    r2_df.to_csv(r2_path, index=False)
    plot_figure8(tstats, r2_df, fig_path)
    plot_figure8a(tstats, fig8a_path)
    plot_figure8b(r2_df, fig8b_path)

    print(f"Saved top-panel data -> {tstats_path}")
    print(f"Saved bottom-panel data -> {r2_path}")
    print(f"Saved Figure 8 -> {fig_path}")
    print(f"Saved Figure 8A -> {fig8a_path}")
    print(f"Saved Figure 8B -> {fig8b_path}")


if __name__ == "__main__":
    main()
