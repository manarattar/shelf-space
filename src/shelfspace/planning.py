"""Glue between forecast, response model and optimiser for one store and one category."""

from __future__ import annotations

import pandas as pd

from shelfspace.config import ALPHA, AVG_FACINGS_PER_ITEM, MAX_FACINGS, MIN_FACINGS
from shelfspace.optimise import equal_split, optimise, proportional
from shelfspace.response import revenue


def default_budget(n_items: int) -> int:
    return int(round(AVG_FACINGS_PER_ITEM * n_items))


def plan(
    demand: pd.DataFrame,
    store: str,
    category: str,
    budget: int | None = None,
    alpha: float | None = None,
) -> dict:
    """Optimise one store x category.

    demand needs: store_id, cat_id, item_id, forecast (units), actual (units), price.
    "current" is a stylised equal split at the default budget; budget may differ (what-if).
    Expected uplift uses forecast demand; backtest uplift re-scores the same allocation with the
    demand that actually happened in the holdout window.
    """
    d = demand[(demand["store_id"] == store) & (demand["cat_id"] == category)]
    d = d.sort_values("item_id").reset_index(drop=True)
    n = len(d)
    alpha = ALPHA[category] if alpha is None else alpha
    base_budget = default_budget(n)
    budget = base_budget if budget is None else budget

    current = equal_split(n, base_budget)
    heuristic = proportional(d["forecast"] * d["price"], base_budget)
    best = optimise(d["forecast"] * d["price"], current, alpha, budget, MIN_FACINGS, MAX_FACINGS)

    def rev(alloc, units):
        return revenue(alloc, current, units, d["price"], alpha)

    r_cur, r_heu, r_opt = (rev(a, d["forecast"]) for a in (current, heuristic, best))
    b_cur, b_heu, b_opt = (rev(a, d["actual"]) for a in (current, heuristic, best))
    out = d[["item_id", "forecast", "actual", "price"]].copy()
    out["current"], out["heuristic"], out["recommended"] = current, heuristic, best
    return {
        "items": out,
        "alpha": alpha,
        "budget": budget,
        "default_budget": base_budget,
        "revenue_current": r_cur,
        "revenue_recommended": r_opt,
        "uplift_vs_current": r_opt / r_cur - 1,
        "uplift_vs_heuristic": r_opt / r_heu - 1,
        "backtest_uplift_vs_current": b_opt / b_cur - 1,
        "backtest_uplift_vs_heuristic": b_opt / b_heu - 1,
    }


def summary(demand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (store, cat), _ in demand.groupby(["store_id", "cat_id"]):
        p = plan(demand, store, cat)
        rows.append(
            {
                "store": store,
                "category": cat,
                "items": len(p["items"]),
                "alpha": p["alpha"],
                "uplift_vs_equal_split": p["uplift_vs_current"],
                "uplift_vs_proportional": p["uplift_vs_heuristic"],
                "backtest_uplift_vs_equal_split": p["backtest_uplift_vs_current"],
            }
        )
    return pd.DataFrame(rows)
