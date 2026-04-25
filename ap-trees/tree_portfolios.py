from __future__ import annotations

from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

from utils import ntile, zscore_cross_section


# ══════════════════════════════════════════════════════════════════════════
# Streaming helpers (fixes OOM on the full 1964-2016 sample)
# ══════════════════════════════════════════════════════════════════════════

_STREAM_COLS = ("permno", "node_id", "tilt_stock_w", "base_stock_w")


def _flush_month_weights(rows: list[pd.DataFrame], out_dir: Path, yy: int, mm: int) -> None:
    """Write one month's stock-weight rows to a gzip-pickled file and free RAM."""
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    keep = [c for c in _STREAM_COLS if c in df.columns]
    path = out_dir / f"{int(yy):04d}_{int(mm):02d}.pkl"
    df[keep].to_pickle(path, compression="gzip")


def node_names(tree_depth: int, q_num: int = 2) -> list[str]:
    nodes = ["1"]
    frontier = ["1"]
    for _ in range(tree_depth):
        nxt = []
        for node in frontier:
            for b in range(1, q_num + 1):
                child = f"{node}{b}"
                nodes.append(child)
                nxt.append(child)
        frontier = nxt
    return nodes


def assign_tree_paths_for_sequence(
    df_month: pd.DataFrame,
    feat_sequence: tuple[str, ...],
    q_num: int = 2,
) -> pd.DataFrame:
    df = df_month.copy()
    df["path"] = "1"
    active_nodes = ["1"]

    for feat in feat_sequence:
        next_nodes = []
        for node in active_nodes:
            mask = df["path"].eq(node)
            if not mask.any():
                continue
            buckets = ntile(df.loc[mask, feat], q_num)
            for b in range(1, q_num + 1):
                child = f"{node}{b}"
                child_idx = buckets.index[buckets.eq(b).fillna(False)]
                df.loc[child_idx, "path"] = child
                next_nodes.append(child)
        active_nodes = next_nodes

    return df


def _node_return_and_weights(
    node_df: pd.DataFrame,
    node_id: str,
    tau: float,
    signal_col: str,
    ret_col: str = "ret",
    size_col: str = "size",
) -> tuple[dict, pd.DataFrame]:
    d = node_df.copy()
    d = d[d[ret_col].notna() & d[size_col].notna() & (d[size_col].astype(float) > 0)].copy()

    if d.empty:
        stats = {"node_id": node_id, "baseline_ret": np.nan, "tilt_ret": np.nan,
                 "n_stocks": 0, "within_node_tilt_l1": np.nan}
        return stats, pd.DataFrame()

    d["base_stock_w"] = d[size_col].astype(float) / d[size_col].astype(float).sum()
    z = zscore_cross_section(d[signal_col].astype(float).fillna(0.0))
    raw_tilt_w = d["base_stock_w"] * np.exp(tau * z)
    d["tilt_stock_w"] = raw_tilt_w / raw_tilt_w.sum()

    ret = d[ret_col].astype(float)
    stats = {
        "node_id": node_id,
        "baseline_ret": float((d["base_stock_w"] * ret).sum()),
        "tilt_ret":     float((d["tilt_stock_w"] * ret).sum()),
        "n_stocks": int(len(d)),
        "within_node_tilt_l1": float(np.abs(d["tilt_stock_w"] - d["base_stock_w"]).sum()),
    }

    keep = [c for c in ["date", "date_dt", "yy", "mm", "permno", "ret", "size", signal_col]
            if c in d.columns]
    wdf = d[keep + ["base_stock_w", "tilt_stock_w"]].copy()
    wdf["node_id"] = node_id
    wdf["permno"] = wdf["permno"].astype(str)
    return stats, wdf


def build_all_ap_tree_candidate_returns(
    panel: pd.DataFrame,
    chars: list[str],
    tree_depth: int,
    tau: float,
    q_num: int = 2,
    signal_col: str = "residual_mom",
    deduplicate: bool = True,
    run_full_tree_set: bool = True,
    stock_weights_dir: Path | None = None,   # ← stream weights here; None = in-memory
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Build AP-tree candidate portfolio returns.

    stock_weights_dir
        If given, per-month stock-weight DataFrames are written to gzip-pickled
        files in this directory instead of being accumulated in RAM.  Pass the
        path you intend to use as the streaming stock-weights directory.
        The function returns None for stock_weights in this mode.

    Returns
    -------
    candidate_returns : pd.DataFrame  (one row per month, one col per portfolio)
    stock_weights     : pd.DataFrame | None
    node_stats        : pd.DataFrame
    """
    if stock_weights_dir is not None:
        stock_weights_dir.mkdir(parents=True, exist_ok=True)
        stream_mode = True
    else:
        stream_mode = False

    if run_full_tree_set:
        sequences = list(product(chars, repeat=tree_depth))
    else:
        sequences = [tuple(chars[i % len(chars)] for i in range(tree_depth))]

    nodes = node_names(tree_depth, q_num=q_num)

    rows, stat_rows = [], []
    weight_rows_accum = []  # only used when NOT streaming

    grouped = list(panel.groupby(["yy", "mm"], sort=True))
    total_months = len(grouped)
    total_trees = len(sequences)

    for month_idx, ((yy, mm), df_m0) in enumerate(grouped, start=1):
        if month_idx == 1 or month_idx % 25 == 0:
            print(f"  AP-tree construction: month {month_idx}/{total_months} "
                  f"({int(yy)}-{int(mm):02d})", flush=True)

        date    = df_m0["date"].iloc[0] if "date" in df_m0.columns else np.nan
        date_dt = df_m0["date_dt"].iloc[0] if "date_dt" in df_m0.columns else pd.NaT

        row = {"date": date, "date_dt": date_dt, "yy": int(yy), "mm": int(mm)}
        month_weight_rows = []  # accumulate for this month only

        for s_idx, seq in enumerate(sequences, start=1):
            if month_idx == 1 or s_idx % 20 == 0:
                print(f"    Tree {s_idx}/{total_trees}: {seq}", flush=True)

            assigned = assign_tree_paths_for_sequence(df_m0, seq, q_num=q_num)
            paths = assigned["path"].astype(str)
            seq_code = "".join(str(chars.index(f) + 1) for f in seq)

            for node in nodes:
                node_len = len(node)
                mask = paths.str[:node_len].eq(node).to_numpy()
                if not mask.any():
                    continue

                members = assigned.loc[mask]
                node_id = f"T{seq_code}_N{node}"

                stats, wdf = _node_return_and_weights(
                    members, node_id=node_id, tau=tau, signal_col=signal_col)

                row[f"baseline_{node_id}"] = stats["baseline_ret"]
                row[f"tilt_{node_id}"]     = stats["tilt_ret"]
                stat_rows.append({"date": date, "date_dt": date_dt,
                                   "yy": int(yy), "mm": int(mm), **stats})

                if not wdf.empty:
                    month_weight_rows.append(wdf)

        rows.append(row)

        # Either flush to disk or accumulate in RAM
        if stream_mode:
            _flush_month_weights(month_weight_rows, stock_weights_dir, int(yy), int(mm))
        else:
            weight_rows_accum.extend(month_weight_rows)

    candidate_returns = (
        pd.DataFrame(rows).sort_values(["yy", "mm"]).reset_index(drop=True)
    )
    node_stats = pd.DataFrame(stat_rows)

    if stream_mode:
        stock_weights_out = None
    else:
        stock_weights_out = (pd.concat(weight_rows_accum, ignore_index=True)
                             if weight_rows_accum else pd.DataFrame())

    if deduplicate:
        candidate_returns = deduplicate_candidate_columns(candidate_returns)

    return candidate_returns, stock_weights_out, node_stats


def deduplicate_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    meta = [c for c in ["date", "date_dt", "yy", "mm"] if c in df.columns]
    out = df[meta].copy()
    for prefix in ["baseline_", "tilt_"]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        mat = df[cols].fillna(0.0).T
        keep_cols = list(mat.index[~mat.duplicated()])
        out = pd.concat([out, df[keep_cols]], axis=1)
    return out


def select_candidate_matrix(
    candidate_returns: pd.DataFrame,
    prefix: str = "baseline_",
) -> pd.DataFrame:
    meta = [c for c in ["date", "date_dt", "yy", "mm"] if c in candidate_returns.columns]
    cols = [c for c in candidate_returns.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix={prefix!r}")
    out = candidate_returns[meta + cols].copy()
    return out.rename(columns={c: "port_" + c[len(prefix):] for c in cols})