import pandas as pd
from pathlib import Path

base = Path("outputs/LME_OP_Investment")
wfile = base / "weights_C_rolling_tc_stock_level_tc_lagged_trade.csv"
stock_dir = base / "stock_weights_by_month_tau_0.5_lagged_trade"

w = pd.read_csv(wfile)
w["date_dt"] = pd.to_datetime(w["date_dt"])

grosses = []

for (yy, mm), wg in w.groupby(["yy", "mm"]):
    f = stock_dir / f"{int(yy)}_{int(mm):02d}.parquet"
    if not f.exists():
        continue

    sw = pd.read_parquet(f)
    sw["node_id"] = sw["node_id"].astype(str)

    cw = wg.set_index("candidate")["weight_trade"].copy()
    cw.index = cw.index.str.replace("port_", "", regex=False)

    merged = sw.merge(cw.rename("candidate_w"), left_on="node_id", right_index=True, how="inner")
    merged["final_stock_w"] = merged["tilt_stock_w"] * merged["candidate_w"]

    grosses.append({
        "yy": yy,
        "mm": mm,
        "stock_gross": merged.groupby("permno")["final_stock_w"].sum().abs().sum(),
        "candidate_gross": wg["weight_trade"].abs().sum(),
    })

g = pd.DataFrame(grosses)
print(g.describe())
print(g.sort_values("stock_gross", ascending=False).head(10))