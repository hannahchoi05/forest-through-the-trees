from __future__ import annotations

import numpy as np
import pandas as pd

from utils import cumulative_wealth, drawdown_from_returns, annualized_sharpe


def add_wealth_drawdown(backtest: pd.DataFrame) -> pd.DataFrame:
    outs = []

    for method, g in backtest.groupby("method", sort=False):
        h = g.copy()

        if {"yy", "mm"}.issubset(h.columns):
            h = h.sort_values(["yy", "mm"])
        else:
            h = h.sort_values("date_dt")

        h["wealth_gross"] = cumulative_wealth(h["gross_ret"])
        h["wealth_net"] = cumulative_wealth(h["net_ret"])
        h["drawdown_net"] = drawdown_from_returns(h["net_ret"])

        outs.append(h)

    return pd.concat(outs, ignore_index=True)


def performance_metrics(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method, g in backtest.groupby("method", sort=False):
        gross = g["gross_ret"].astype(float).dropna()
        net = g["net_ret"].astype(float).dropna()
        turnover = g["turnover"].astype(float).fillna(0.0)
        cost = g["cost"].astype(float).fillna(0.0)

        dd = drawdown_from_returns(net)

        gross_sd = gross.std(ddof=1)
        net_sd = net.std(ddof=1)

        sr_gross = annualized_sharpe(gross)
        sr_net = annualized_sharpe(net)

        rows.append({
            "method": method,
            "n_months": len(g),

            "start_date": g["date_dt"].min() if "date_dt" in g else None,
            "end_date": g["date_dt"].max() if "date_dt" in g else None,

            "mean_gross_monthly": gross.mean(),
            "mean_net_monthly": net.mean(),

            "mean_gross_ann": 12.0 * gross.mean(),
            "mean_net_ann": 12.0 * net.mean(),

            "vol_gross_monthly": gross_sd,
            "vol_net_monthly": net_sd,

            "vol_gross_ann": np.sqrt(12.0) * gross_sd,
            "vol_net_ann": np.sqrt(12.0) * net_sd,

            "sharpe_gross_ann": sr_gross,
            "sharpe_net_ann": sr_net,
            "sharpe_decay_due_to_costs": sr_gross - sr_net,

            "hit_rate_gross": float((gross > 0).mean()),
            "hit_rate_net": float((net > 0).mean()),

            "avg_turnover": turnover.mean(),
            "avg_cost": cost.mean(),

            "max_drawdown_net": dd.min(),

            "terminal_wealth_gross": cumulative_wealth(gross).iloc[-1] if len(gross) else np.nan,
            "terminal_wealth_net": cumulative_wealth(net).iloc[-1] if len(net) else np.nan,
        })

    return pd.DataFrame(rows)