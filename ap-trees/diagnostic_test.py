from pathlib import Path
import numpy as np
import pandas as pd

WEIGHTS_PATH = Path(r"C:\Users\hongv\OneDrive\Tài liệu\forest-through-the-trees\ap-trees\outputs\LME_OP_Investment\weights_A1_static_no_tc_lagged_trade.csv")
STOCK_WEIGHTS_DIR = Path(r"C:\Users\hongv\OneDrive\Tài liệu\forest-through-the-trees\ap-trees\outputs\LME_OP_Investment\stock_weights_by_month_lagged_trade")
OUT_DIR = WEIGHTS_PATH.parent / "diagnostics_stock_concentration"
OUT_DIR.mkdir(exist_ok=True)

# Use "tilt_stock_w" to match current optimizer default.
# Use "base_stock_w" if you want pure value-weighted AP-tree node holdings.
STOCK_WEIGHT_COL = "tilt_stock_w"

TOP_K = 10
WARN_SINGLE_STOCK = 0.10
WARN_EXTREME = 0.25
WARN_LOW_NEFF = 20

weights_df = pd.read_csv(WEIGHTS_PATH)
weights_df["candidate"] = weights_df["candidate"].astype(str)

weight_col = "weight" if "weight" in weights_df.columns else "weight_trade"

w_map = weights_df.set_index("candidate")[weight_col].astype(float)
w_map.index = [c.replace("port_", "") for c in w_map.index]
w_map = w_map[w_map.abs() > 1e-12]

active_nodes = set(w_map.index)

print(f"Loaded {len(w_map)} active candidate weights.")
print(f"Using stock weight column: {STOCK_WEIGHT_COL}")
print(f"Stock weights dir: {STOCK_WEIGHTS_DIR}")

files = sorted(STOCK_WEIGHTS_DIR.glob("*.pkl"))
print(f"Found {len(files)} monthly stock-weight files.")

records = []
top_rows = []

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

    if STOCK_WEIGHT_COL not in sw.columns:
        raise ValueError(f"{f.name} missing {STOCK_WEIGHT_COL}. Available: {list(sw.columns)}")

    sw = sw[["permno", "node_id", STOCK_WEIGHT_COL]].copy()
    sw["node_id"] = sw["node_id"].astype(str)
    sw["permno"] = sw["permno"].astype(str)

    # CRITICAL SPEEDUP: keep only selected AP-tree nodes before doing anything else.
    sw = sw[sw["node_id"].isin(active_nodes)]
    if sw.empty:
        continue

    sw["candidate_w"] = sw["node_id"].map(w_map)
    sw["final_w"] = sw["candidate_w"].astype(float) * sw[STOCK_WEIGHT_COL].astype(float)

    stock_w = sw.groupby("permno", sort=False)["final_w"].sum()
    stock_w = stock_w[stock_w.abs() > 1e-14]

    if stock_w.empty:
        continue

    abs_w = stock_w.abs()
    gross = float(abs_w.sum())
    net = float(stock_w.sum())
    max_abs = float(abs_w.max())
    max_signed = float(stock_w.loc[abs_w.idxmax()])
    top5 = float(abs_w.nlargest(5).sum())
    top10 = float(abs_w.nlargest(10).sum())
    hhi = float((abs_w ** 2).sum())
    n_eff = float(1.0 / hhi) if hhi > 0 else np.nan

    records.append({
        "yy": yy,
        "mm": mm,
        "gross_stock_exposure": gross,
        "net_stock_exposure": net,
        "max_abs_stock_weight": max_abs,
        "max_signed_stock_weight": max_signed,
        "top5_abs_weight_sum": top5,
        "top10_abs_weight_sum": top10,
        "effective_n_stocks": n_eff,
        "n_stocks_nonzero": int(len(stock_w)),
        "flag_gt_10pct": max_abs > WARN_SINGLE_STOCK,
        "flag_gt_25pct": max_abs > WARN_EXTREME,
        "flag_low_neff": n_eff < WARN_LOW_NEFF,
    })

    for rank, (permno, val) in enumerate(abs_w.sort_values(ascending=False).head(TOP_K).items(), start=1):
        top_rows.append({
            "yy": yy,
            "mm": mm,
            "rank": rank,
            "permno": permno,
            "abs_weight": float(val),
            "signed_weight": float(stock_w.loc[permno]),
        })

diag = pd.DataFrame(records)
tops = pd.DataFrame(top_rows)

diag_path = OUT_DIR / f"stock_concentration_{STOCK_WEIGHT_COL}.csv"
tops_path = OUT_DIR / f"top_holdings_{STOCK_WEIGHT_COL}.csv"

diag.to_csv(diag_path, index=False)
tops.to_csv(tops_path, index=False)

print("\n===== SUMMARY =====")
print(diag.describe(include="all"))

print("\n===== WORST MONTHS BY MAX STOCK WEIGHT =====")
print(diag.sort_values("max_abs_stock_weight", ascending=False).head(20).to_string(index=False))

print("\n===== FLAGGED MONTHS =====")
flagged = diag[diag["flag_gt_10pct"] | diag["flag_gt_25pct"] | diag["flag_low_neff"]]
print(flagged.head(50).to_string(index=False))

print(f"\nSaved diagnostics to:\n{diag_path}\n{tops_path}")