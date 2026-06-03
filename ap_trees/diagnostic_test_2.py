from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\hongv\OneDrive\Tài liệu\forest-through-the-trees\ap-trees\outputs\LME_OP_Investment")

WEIGHTS_PATH = BASE_DIR / "weights_A1_static_no_tc_lagged_trade.csv"
STOCK_WEIGHTS_DIR = BASE_DIR / "stock_weights_by_month_lagged_trade"
OUT_DIR = BASE_DIR / "diagnostics_return_contributions"
OUT_DIR.mkdir(exist_ok=True)

STOCK_WEIGHT_COL = "base_stock_w"

TOP_K = 10

weights_df = pd.read_csv(WEIGHTS_PATH)
weights_df["candidate"] = weights_df["candidate"].astype(str)

weight_col = "weight" if "weight" in weights_df.columns else "weight_trade"

w_map = weights_df.set_index("candidate")[weight_col].astype(float)
w_map.index = [c.replace("port_", "") for c in w_map.index]
w_map = w_map[w_map.abs() > 1e-12]

active_nodes = set(w_map.index)

print(f"Loaded {len(w_map)} active AP-tree candidate weights.")
print(f"Using stock weight column: {STOCK_WEIGHT_COL}")


records = []
top_contrib_rows = []
top_weight_rows = []

files = sorted(STOCK_WEIGHTS_DIR.glob("*.pkl"))
print(f"Found {len(files)} monthly stock-weight files.")

for idx, f in enumerate(files, start=1):
    if idx == 1 or idx % 25 == 0:
        print(f"Processing {idx}/{len(files)}: {f.name}", flush=True)

    yy, mm = map(int, f.stem.split("_")[:2])

    try:
        sw = pd.read_pickle(f, compression="gzip")
    except Exception:
        sw = pd.read_pickle(f)

    if sw.empty:
        continue

    needed = {"permno", "node_id", STOCK_WEIGHT_COL, "ret"}
    missing = needed.difference(sw.columns)
    if missing:
        raise ValueError(f"{f.name} missing columns: {missing}. Available: {list(sw.columns)}")

    sw = sw[["permno", "node_id", STOCK_WEIGHT_COL, "ret"]].copy()
    sw["permno"] = sw["permno"].astype(str)
    sw["node_id"] = sw["node_id"].astype(str)

    # Keep only selected AP-tree nodes.
    sw = sw[sw["node_id"].isin(active_nodes)]
    if sw.empty:
        continue

    sw["candidate_w"] = sw["node_id"].map(w_map).astype(float)
    sw["stock_component_w"] = sw["candidate_w"] * sw[STOCK_WEIGHT_COL].astype(float)

    # Aggregate overlapping AP-tree node exposure into final stock weight.
    stock = (
        sw.groupby("permno", sort=False)
        .agg(
            stock_w=("stock_component_w", "sum"),
            ret=("ret", "first"),
        )
        .reset_index()
    )

    stock = stock[stock["stock_w"].abs() > 1e-14].copy()
    if stock.empty:
        continue

    stock["contribution"] = stock["stock_w"] * stock["ret"].astype(float)
    stock["abs_contribution"] = stock["contribution"].abs()
    stock["abs_weight"] = stock["stock_w"].abs()

    gross_exposure = float(stock["abs_weight"].sum())
    net_exposure = float(stock["stock_w"].sum())
    long_exposure = float(stock.loc[stock["stock_w"] > 0, "stock_w"].sum())
    short_exposure = float(stock.loc[stock["stock_w"] < 0, "stock_w"].sum())
    max_abs_weight = float(stock["abs_weight"].max())

    month_ret = float(stock["contribution"].sum())
    abs_month_ret = abs(month_ret)

    top1_abs_contrib = float(stock["abs_contribution"].max())
    top5_abs_contrib = float(stock["abs_contribution"].nlargest(5).sum())
    top10_abs_contrib = float(stock["abs_contribution"].nlargest(10).sum())

    top1_pct_abs_ret = top1_abs_contrib / abs_month_ret if abs_month_ret > 1e-12 else np.nan
    top5_pct_abs_ret = top5_abs_contrib / abs_month_ret if abs_month_ret > 1e-12 else np.nan
    top10_pct_abs_ret = top10_abs_contrib / abs_month_ret if abs_month_ret > 1e-12 else np.nan

    records.append({
        "yy": yy,
        "mm": mm,
        "month_ret_from_stock_contrib": month_ret,
        "gross_stock_exposure": gross_exposure,
        "net_stock_exposure": net_exposure,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "long_minus_abs_short": long_exposure - abs(short_exposure),
        "max_abs_stock_weight": max_abs_weight,
        "top1_abs_contribution": top1_abs_contrib,
        "top5_abs_contribution": top5_abs_contrib,
        "top10_abs_contribution": top10_abs_contrib,
        "top1_pct_abs_month_ret": top1_pct_abs_ret,
        "top5_pct_abs_month_ret": top5_pct_abs_ret,
        "top10_pct_abs_month_ret": top10_pct_abs_ret,
        "n_stocks": int(len(stock)),
    })

    # Top return contributors by absolute contribution
    topc = stock.sort_values("abs_contribution", ascending=False).head(TOP_K)
    for rank, row in enumerate(topc.itertuples(index=False), start=1):
        top_contrib_rows.append({
            "yy": yy,
            "mm": mm,
            "rank": rank,
            "permno": row.permno,
            "stock_w": float(row.stock_w),
            "ret": float(row.ret),
            "contribution": float(row.contribution),
            "abs_contribution": float(row.abs_contribution),
        })

    topw = stock.sort_values("abs_weight", ascending=False).head(TOP_K)
    for rank, row in enumerate(topw.itertuples(index=False), start=1):
        top_weight_rows.append({
            "yy": yy,
            "mm": mm,
            "rank": rank,
            "permno": row.permno,
            "stock_w": float(row.stock_w),
            "ret": float(row.ret),
            "contribution": float(row.contribution),
            "abs_weight": float(row.abs_weight),
        })

diag = pd.DataFrame(records)
top_contrib = pd.DataFrame(top_contrib_rows)
top_weights = pd.DataFrame(top_weight_rows)

diag["wealth_from_contrib"] = (1.0 + diag["month_ret_from_stock_contrib"]).cumprod()

diag_path = OUT_DIR / f"return_contribution_diagnostics_{STOCK_WEIGHT_COL}.csv"
topc_path = OUT_DIR / f"top_return_contributors_{STOCK_WEIGHT_COL}.csv"
topw_path = OUT_DIR / f"top_weight_holdings_{STOCK_WEIGHT_COL}.csv"

diag.to_csv(diag_path, index=False)
top_contrib.to_csv(topc_path, index=False)
top_weights.to_csv(topw_path, index=False)

print("\n===== LONG/SHORT EXPOSURE SUMMARY =====")
print(
    diag[
        [
            "gross_stock_exposure",
            "net_stock_exposure",
            "long_exposure",
            "short_exposure",
            "long_minus_abs_short",
        ]
    ].describe()
)

print("\n===== RETURN CONTRIBUTION SUMMARY =====")
print(
    diag[
        [
            "month_ret_from_stock_contrib",
            "top1_pct_abs_month_ret",
            "top5_pct_abs_month_ret",
            "top10_pct_abs_month_ret",
            "max_abs_stock_weight",
        ]
    ].describe()
)

print("\n===== BIGGEST MONTHLY RETURNS =====")
print(
    diag.sort_values("month_ret_from_stock_contrib", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\n===== WORST MONTHLY RETURNS =====")
print(
    diag.sort_values("month_ret_from_stock_contrib", ascending=True)
    .head(15)
    .to_string(index=False)
)

print("\n===== MONTHS WHERE TOP 1 STOCK DOMINATES ABS MONTHLY RETURN =====")
print(
    diag.sort_values("top1_pct_abs_month_ret", ascending=False)
    .head(15)
    .to_string(index=False)
)

print(f"\nSaved:\n{diag_path}\n{topc_path}\n{topw_path}")