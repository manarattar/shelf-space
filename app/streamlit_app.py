"""Shelf Space Allocator: one-page demo. Pick, see, result."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from shelfspace.config import ALPHA, CATEGORIES, MAX_FACINGS, MIN_FACINGS, RESULTS_DIR, STORES
from shelfspace.experiment import power, required_n_per_arm, simulate_experiment
from shelfspace.planning import default_budget, plan

REPO = "https://github.com/manarattar/shelf-space"
GREEN, GREY, BLUE = "#0b6e4f", "#9aa5a0", "#2b6cb0"

st.set_page_config(page_title="Shelf Space Allocator", page_icon="🛒", layout="wide")


@st.cache_data
def load():
    demand = pd.read_parquet(RESULTS_DIR / "demand_forecast.parquet")
    daily = pd.read_parquet(RESULTS_DIR / "holdout_daily.parquet")
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    return demand, daily, metrics


demand, daily, metrics = load()
pct = lambda x: f"{x * 100:+.1f}%"  # noqa: E731

st.title("Shelf Space Allocator")
st.markdown(
    "How should a fixed amount of shelf space be split between products? "
    "**Forecast demand, allocate space, then test whether it works.** "
    "Three steps below, about two minutes to go through."
)
st.info(
    f"**Demo on {metrics['data_source']} retail data.** The relation between shelf space and "
    "sales is an **assumption**, not a measurement, so the uplifts below show how the method "
    "works, not what it would earn in a real store. Details at the bottom.",
    icon="ℹ️",
)

# ---- Step 1: pick -----------------------------------------------------------------------
st.header("1. Pick a store and a category")
c1, c2, c3 = st.columns([1, 1, 2])
store = c1.selectbox("Store", STORES)
category = c2.selectbox("Category", CATEGORIES)
n_items = int(((demand["store_id"] == store) & (demand["cat_id"] == category)).sum())
base_budget = default_budget(n_items)
budget = c3.slider(
    f"Shelf budget (total facings, current planogram = {base_budget})",
    min_value=n_items * MIN_FACINGS,
    max_value=min(n_items * MAX_FACINGS, base_budget * 2),
    value=base_budget,
    help="Facings = the number of product units side by side on the shelf. Try cutting it.",
)
baseline_choice = st.radio(
    "Compare the recommendation with",
    [
        "Equal split (every item gets the same space)",
        "Proportional rule (space follows expected revenue)",
    ],
    horizontal=True,
    index=1,
    help="The proportional rule is what a sensible category manager might already do, "
    "so it is the fairer yardstick.",
)
vs_rule = baseline_choice.startswith("Proportional")

p = plan(demand, store, category, budget=budget)
items = p["items"]

# ---- Step 2: see ------------------------------------------------------------------------
st.header("2. See the forecast and the recommended shelf")
left, right = st.columns(2)

with left:
    st.subheader("Demand forecast, next 28 days")
    cat_metrics = metrics["item_28d_totals"]["by_category"][category]
    m1, m2 = st.columns(2)
    m1.metric("Baseline error (WAPE)", f"{cat_metrics['seasonal_naive']:.1%}")
    m2.metric(
        "Model error (WAPE)",
        f"{cat_metrics['lightgbm']:.1%}",
        f"{(cat_metrics['lightgbm'] / cat_metrics['seasonal_naive'] - 1):+.0%} vs baseline",
        delta_color="inverse",
    )
    st.caption(
        "Error of the 28-day total per item (lower is better). Baseline = average of the same "
        "weekday over the last 4 weeks. Model = LightGBM on lag, price and calendar features, "
        "scored on 28 days it never saw."
    )
    d = daily[(daily["store_id"] == store) & (daily["cat_id"] == category)].melt(
        id_vars="date",
        value_vars=["actual", "baseline", "lgbm"],
        var_name="series",
        value_name="units",
    )
    d["series"] = d["series"].map({"actual": "Actual", "baseline": "Baseline", "lgbm": "Model"})
    chart = (
        alt.Chart(d)
        .mark_line()
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("units:Q", title="Units per day"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Actual", "Baseline", "Model"],
                    range=["#1b2a25", GREY, GREEN],
                ),
            ),
            strokeDash=alt.condition(
                alt.datum.series == "Baseline", alt.value([4, 3]), alt.value([0])
            ),
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("Recommended facings")
    items["change"] = items["recommended"] - items["current"]
    moved = pd.concat([items.nlargest(8, "change"), items.nsmallest(8, "change")]).drop_duplicates(
        "item_id"
    )
    moved = moved[moved["change"] != 0].assign(
        direction=lambda x: x["change"].apply(lambda v: "More space" if v > 0 else "Less space")
    )
    bar = (
        alt.Chart(moved)
        .mark_bar()
        .encode(
            y=alt.Y(
                "item_id:N",
                sort=alt.EncodingSortField("change", order="descending"),
                title=None,
            ),
            x=alt.X("change:Q", title="Facings added or removed vs current"),
            color=alt.Color(
                "direction:N",
                legend=None,
                scale=alt.Scale(domain=["More space", "Less space"], range=[GREEN, "#c0564b"]),
            ),
            tooltip=[
                "item_id",
                "current",
                "recommended",
                alt.Tooltip("forecast:Q", format=".0f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(bar, use_container_width=True)
    st.caption(
        f"The {len(moved)} biggest changes out of {n_items} items. Fast sellers gain space, "
        "slow sellers give it up, within {lo} to {hi} facings per item.".format(
            lo=MIN_FACINGS, hi=MAX_FACINGS
        )
    )

with st.expander("Full table for this store and category"):
    st.dataframe(
        items.rename(
            columns={
                "item_id": "Item",
                "forecast": "Forecast units (28d)",
                "actual": "Actual units (28d)",
                "price": "Avg price",
                "current": "Current facings",
                "heuristic": "Proportional rule",
                "recommended": "Recommended",
                "change": "Change",
            }
        ).round(1),
        hide_index=True,
        use_container_width=True,
    )

# ---- Step 3: result ---------------------------------------------------------------------
st.header("3. Would it work? The expected gain, and how a real test would show it")
uplift = p["uplift_vs_heuristic"] if vs_rule else p["uplift_vs_current"]
backtest = p["backtest_uplift_vs_heuristic"] if vs_rule else p["backtest_uplift_vs_current"]
yardstick = "the proportional rule" if vs_rule else "the equal split"

r1, r2, r3 = st.columns(3)
r1.metric("Expected revenue change", pct(uplift), help="Using the forecast demand.")
r2.metric(
    "Same decision, scored on what actually sold",
    pct(backtest),
    help="The allocation is chosen from the forecast, then re-scored with the real demand of the "
    "held-out 28 days. If it differs much from the left number, the forecast is misleading.",
)
r3.metric("Response curve exponent (assumed)", f"{ALPHA[category]:.2f}")
st.caption(f"Versus {yardstick}, at a budget of {budget} facings.")

st.subheader("Simulated store experiment")
n_arm = st.slider("Stores per arm (treatment and control)", 10, 1000, 30, step=10)
if abs(uplift) < 1e-6:
    st.write("No difference between the allocations, so there is nothing to test.")
else:
    res = simulate_experiment(uplift, n_per_arm=n_arm, seed=42)
    need = required_n_per_arm(abs(uplift))
    pw = power(abs(uplift), n_arm)
    e1, e2, e3 = st.columns(3)
    e1.metric("Estimated effect", pct(res["estimate"]))
    e2.metric(
        "95% confidence interval",
        f"{res['ci_low'] * 100:+.1f}% to {res['ci_high'] * 100:+.1f}%",
    )
    e3.metric("p-value", f"{res['p_value']:.3f}")
    verdict = (
        "The test **detects** the effect"
        if res["significant"]
        else "The test **cannot tell** this apart from noise"
    )
    st.markdown(
        f"{verdict} with {n_arm} stores per arm (power {pw:.0%}). "
        f"To reliably detect a {abs(uplift) * 100:.1f}% change you need about **{need} stores per arm** "
        "for 80% power."
    )
    st.caption(
        "Simulated: store sizes vary, weekly revenue is noisy (12%), 8 weeks before and 8 weeks after. "
        "Each store is compared with its own past, then treatment is compared with control."
    )

# ---- Assumptions ------------------------------------------------------------------------
with st.expander("Assumptions and limits (read this)", expanded=False):
    st.markdown(
        f"""
- **Data:** {metrics["data_source"]}. 3 stores, 3 categories, {len(demand) // 3} items per store.
  The code reads the public M5 Walmart files unchanged if they are placed in `data/raw/`.
- **Shelf space is simulated.** Public data has no planogram, so sales are assumed to follow
  `sales = demand × (space ÷ current space)^α` with α = {ALPHA}. Real α would have to come from store
  experiments or planogram history.
- **"Current" is a stylised equal split.** That is why the gain against it is large. The proportional
  rule is the more honest comparison, and against it the optimiser adds little. That is a finding:
  most of the value is in not splitting space equally.
- **Optimiser:** greedy marginal allocation. For this concave objective it is provably optimal
  (tested against brute force).
- **Forecast:** direct 28-day-ahead, features older than 28 days only (no leakage), evaluated on the
  last 28 days. Item-level daily error is high (count noise); 28-day totals are what the shelf
  decision uses.
- **Not modelled:** substitution between items, store-specific space limits, margin, supplier deals.
        """
    )
    st.markdown(f"Source code, tests and write-up: [{REPO}]({REPO})")
