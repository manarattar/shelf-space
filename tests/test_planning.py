import numpy as np
import pandas as pd
import pytest

from shelfspace.planning import default_budget, plan


@pytest.fixture
def demand():
    rng = np.random.default_rng(0)
    rows = []
    for store in ["A", "B"]:
        for cat in ["FOODS", "HOBBIES"]:
            for i in range(20):
                f = float(rng.lognormal(3, 1))
                rows.append(
                    {
                        "store_id": store,
                        "cat_id": cat,
                        "item_id": f"{cat}_{i:02d}",
                        "forecast": f,
                        "actual": f * rng.uniform(0.8, 1.2),
                        "price": float(rng.uniform(1, 9)),
                    }
                )
    return pd.DataFrame(rows)


def test_plan_allocations_use_the_budget(demand):
    p = plan(demand, "A", "FOODS")
    items = p["items"]
    assert items["current"].sum() == items["recommended"].sum() == default_budget(20)
    assert items["heuristic"].sum() == default_budget(20)


def test_plan_beats_current_and_heuristic(demand):
    p = plan(demand, "B", "HOBBIES")
    assert p["uplift_vs_current"] > 0
    assert p["uplift_vs_heuristic"] >= -1e-9


def test_smaller_budget_gives_less_revenue(demand):
    base = plan(demand, "A", "FOODS")
    cut = plan(demand, "A", "FOODS", budget=default_budget(20) - 10)
    assert cut["revenue_recommended"] < base["revenue_recommended"]


def test_higher_alpha_means_larger_gain_from_reallocating(demand):
    low = plan(demand, "A", "FOODS", alpha=0.1)["uplift_vs_current"]
    high = plan(demand, "A", "FOODS", alpha=0.6)["uplift_vs_current"]
    assert high > low
