from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ntile, zscore_cross_section


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

            for bucket in range(1, q_num + 1):
                child = f"{node}{bucket}"
                child_idx = buckets.index[buckets.eq(bucket).fillna(False)]
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
    valid = (
        node_df[ret_col].notna()
        & node_df[size_col].notna()
        & (node_df[size_col].astype(float) > 0)
    )

    if not valid.any():
        stats = {
            "node_id": node_id,
            "baseline_ret": np.nan,
            "tilt_ret": np.nan,
            "n_stocks": 0,
            "within_node_tilt_l1": np.nan,
        }
        return stats, pd.DataFrame()

    d = node_df.loc[valid].copy()

    size = d[size_col].astype(float)
    ret = d[ret_col].astype(float)

    base_w = size / size.sum()

    signal = d[signal_col].astype(float).fillna(0.0)
    z = zscore_cross_section(signal)

    raw_tilt_w = base_w * np.exp(tau * z)
    tilt_w = raw_tilt_w / raw_tilt_w.sum()

    d["base_stock_w"] = base_w
    d["tilt_stock_w"] = tilt_w

    stats = {
        "node_id": node_id,
        "baseline_ret": float((base_w * ret).sum()),
        "tilt_ret": float((tilt_w * ret).sum()),
        "n_stocks": int(len(d)),
        "within_node_tilt_l1": float(np.abs(tilt_w - base_w).sum()),
    }

    keep = ["date", "date_dt", "yy", "mm", "permno", "ret", "size", signal_col]
    keep = [c for c in keep if c in d.columns]

    wdf = d[keep + ["base_stock_w", "tilt_stock_w"]].copy()
    wdf["node_id"] = node_id
    wdf["permno"] = wdf["permno"].astype(str)

    return stats, wdf


def _compact_stock_weight_df(df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    out = df.copy()

    if "yy" in out.columns:
        out["yy"] = out["yy"].astype("int16")

    if "mm" in out.columns:
        out["mm"] = out["mm"].astype("int8")

    for col in ["ret", "size", signal_col, "base_stock_w", "tilt_stock_w"]:
        if col in out.columns:
            out[col] = out[col].astype("float32")

    out["permno"] = out["permno"].astype(str)
    out["node_id"] = out["node_id"].astype(str)

    return out


def build_all_ap_tree_candidate_returns(
    panel: pd.DataFrame,
    chars: list[str],
    tree_depth: int,
    tau: float,
    q_num: int = 2,
    signal_col: str = "residual_mom",
    deduplicate: bool = True,
    run_full_tree_set: bool = True,
    stock_weights_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path | None, pd.DataFrame]:
    """
    Optimized AP-tree candidate construction.

    Same logic as before:
      - same recursive AP-tree paths
      - same baseline value-weighted node returns
      - same residual-momentum tilt
      - same node_id naming

    Faster because:
      - avoids looping through all possible nodes with repeated string masks
      - groups realized node prefixes directly
      - streams stock weights month-by-month to Parquet
    """
    if run_full_tree_set:
        sequences = list(product(chars, repeat=tree_depth))
    else:
        sequence = tuple(chars[i % len(chars)] for i in range(tree_depth))
        sequences = [sequence]

    if stock_weights_dir is not None:
        stock_weights_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    stat_rows = []

    grouped = list(panel.groupby(["yy", "mm"], sort=True))
    total_months = len(grouped)
    total_trees = len(sequences)

    for month_idx, ((yy, mm), df_m0) in enumerate(grouped, start=1):
        if month_idx == 1 or month_idx % 25 == 0:
            print(
                f"  AP-tree construction: month {month_idx}/{total_months} "
                f"({int(yy)}-{int(mm):02d})",
                flush=True,
            )

        date = df_m0["date"].iloc[0] if "date" in df_m0.columns else np.nan
        date_dt = df_m0["date_dt"].iloc[0] if "date_dt" in df_m0.columns else pd.NaT

        row = {
            "date": date,
            "date_dt": date_dt,
            "yy": int(yy),
            "mm": int(mm),
        }

        month_weight_rows = []

        for s_idx, seq in enumerate(sequences, start=1):
            if month_idx == 1 or s_idx % 20 == 0:
                print(f"    Tree {s_idx}/{total_trees}: {seq}", flush=True)

            assigned = assign_tree_paths_for_sequence(df_m0, seq, q_num=q_num)
            paths = assigned["path"].astype(str)

            seq_code = "".join(str(chars.index(f) + 1) for f in seq)

            # Level 0 node is prefix length 1, final depth is tree_depth + 1.
            for level in range(tree_depth + 1):
                prefix_len = level + 1
                prefixes = paths.str.slice(0, prefix_len)

                # Group only realized prefixes instead of scanning all possible nodes.
                for node, idx in prefixes.groupby(prefixes, sort=True).groups.items():
                    members = assigned.loc[idx]
                    node_id = f"T{seq_code}_N{node}"

                    stats, wdf = _node_return_and_weights(
                        members,
                        node_id=node_id,
                        tau=tau,
                        signal_col=signal_col,
                    )

                    row[f"baseline_{node_id}"] = stats["baseline_ret"]
                    row[f"tilt_{node_id}"] = stats["tilt_ret"]

                    stat_rows.append(
                        {
                            "date": date,
                            "date_dt": date_dt,
                            "yy": int(yy),
                            "mm": int(mm),
                            **stats,
                        }
                    )

                    if stock_weights_dir is not None and not wdf.empty:
                        month_weight_rows.append(wdf)

        if stock_weights_dir is not None and month_weight_rows:
            month_df = pd.concat(month_weight_rows, ignore_index=True)
            month_df = _compact_stock_weight_df(month_df, signal_col=signal_col)

            file_path = stock_weights_dir / f"{int(yy)}_{int(mm):02d}.parquet"
            month_df.to_parquet(file_path, index=False)

        rows.append(row)

    candidate_returns = (
        pd.DataFrame(rows)
        .sort_values(["yy", "mm"])
        .reset_index(drop=True)
    )

    node_stats = pd.DataFrame(stat_rows)

    if deduplicate:
        candidate_returns = deduplicate_candidate_columns(candidate_returns)

    return candidate_returns, stock_weights_dir, node_stats


def deduplicate_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    meta = [c for c in ["date", "date_dt", "yy", "mm"] if c in df.columns]
    out = df[meta].copy()

    for prefix in ["baseline_", "tilt_"]:
        cols = [c for c in df.columns if c.startswith(prefix)]

        if not cols:
            continue

        mat = df[cols].fillna(0.0).T
        keep_mask = ~mat.duplicated()
        keep_cols = list(mat.index[keep_mask])

        out = pd.concat([out, df[keep_cols]], axis=1)

    return out


def select_candidate_matrix(
    candidate_returns: pd.DataFrame,
    prefix: str = "tilt_",
) -> pd.DataFrame:
    meta = [c for c in ["date", "date_dt", "yy", "mm"] if c in candidate_returns.columns]
    cols = [c for c in candidate_returns.columns if c.startswith(prefix)]

    if not cols:
        raise ValueError(f"No candidate columns found with prefix={prefix!r}")

    out = candidate_returns[meta + cols].copy()
    rename = {c: "port_" + c[len(prefix):] for c in cols}

    return out.rename(columns=rename)