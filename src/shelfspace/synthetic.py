"""Synthetic retail data in the exact M5 file schema.

Used as a fallback so the whole pipeline runs without a Kaggle download. Swapping in the real
M5 files (calendar.csv, sales_train_evaluation.csv, sell_prices.csv) needs no code change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shelfspace.config import CATEGORIES, ITEMS_PER_CATEGORY, STORES

DEFAULT_DAYS = 800
START_DATE = "2022-01-01"

# weekday multipliers (Mon..Sun) per category
_WEEKDAY = {
    "FOODS": [0.9, 0.9, 0.95, 1.0, 1.15, 1.3, 1.0],
    "HOUSEHOLD": [0.95, 0.95, 1.0, 1.0, 1.1, 1.2, 0.95],
    "HOBBIES": [0.8, 0.85, 0.9, 1.0, 1.1, 1.4, 1.2],
}
_STORE_SIZE = {"CA_1": 1.3, "CA_2": 1.0, "TX_1": 0.8}
_STATE = {"CA_1": "CA", "CA_2": "CA", "TX_1": "TX"}


def _calendar(n_days: int) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, periods=n_days, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["wm_yr_wk"] = 11101 + (np.arange(n_days) // 7)
    cal["weekday"] = dates.day_name()
    cal["wday"] = ((dates.day_of_week + 2) % 7) + 1  # M5: Saturday = 1
    cal["month"] = dates.month
    cal["year"] = dates.year
    cal["d"] = [f"d_{i + 1}" for i in range(n_days)]
    cal["event_name_1"] = ""
    cal["event_type_1"] = ""
    events = {
        (12, 25): "Christmas",
        (1, 1): "NewYear",
        (12, 31): "NewYearEve",
        (11, 25): "BlackFriday",
        (5, 5): "KingsDay",
        (2, 14): "Valentines",
        (4, 20): "Easter",
    }
    for (m, d), name in events.items():
        mask = (dates.month == m) & (dates.day == d)
        cal.loc[mask, "event_name_1"] = name
        cal.loc[mask, "event_type_1"] = "Cultural"
    return cal


def generate(
    n_days: int = DEFAULT_DAYS, seed: int = 7, scale: float = 1.0
) -> dict[str, pd.DataFrame]:
    """Return calendar, sales (wide, M5 layout) and prices frames. scale shrinks the item counts."""
    rng = np.random.default_rng(seed)
    cal = _calendar(n_days)
    dates = cal["date"]
    t = np.arange(n_days)
    yearly = 1 + 0.12 * np.sin(2 * np.pi * (t - 60) / 365.25)
    event_boost = np.where(cal["event_name_1"] != "", 1.35, 1.0)
    weekday_idx = dates.dt.day_of_week.to_numpy()

    items = []
    for cat in CATEGORIES:
        n = max(3, int(round(ITEMS_PER_CATEGORY[cat] * scale)))
        for k in range(n):
            items.append((f"{cat}_{k // 20 + 1}_{k + 1:03d}", f"{cat}_{k // 20 + 1}", cat))

    price_rows, sales_rows = [], []
    weeks = cal["wm_yr_wk"].unique()
    for item_id, dept_id, cat in items:
        base_rate = float(
            np.exp(rng.normal(0.6 if cat == "FOODS" else 0.1 if cat == "HOUSEHOLD" else -0.5, 0.9))
        )
        base_price = float(np.round(np.exp(rng.normal(1.3, 0.55)), 2)) + 0.49
        elasticity = rng.uniform(0.6, 2.2)
        trend = rng.normal(0, 0.0002)
        for store in STORES:
            # weekly price with occasional promotions
            wk_price = np.full(len(weeks), base_price) * (1 + rng.normal(0, 0.01, len(weeks)))
            promo = rng.random(len(weeks)) < 0.08
            wk_price = np.where(promo, wk_price * rng.uniform(0.75, 0.9), wk_price).round(2)
            price_rows.append(
                pd.DataFrame(
                    {
                        "store_id": store,
                        "item_id": item_id,
                        "wm_yr_wk": weeks,
                        "sell_price": wk_price,
                    }
                )
            )
            price_d = np.repeat(wk_price, 7)[:n_days]
            price_effect = (price_d / base_price) ** (-elasticity)
            mu = (
                base_rate
                * _STORE_SIZE[store]
                * yearly
                * event_boost
                * np.array(_WEEKDAY[cat])[weekday_idx]
                * price_effect
                * np.exp(trend * t)
            )
            # negative binomial noise (overdispersed counts), r controls dispersion
            r = 4.0
            sales = rng.negative_binomial(r, r / (r + mu))
            row = {
                "id": f"{item_id}_{store}_evaluation",
                "item_id": item_id,
                "dept_id": dept_id,
                "cat_id": cat,
                "store_id": store,
                "state_id": _STATE[store],
            }
            row.update({f"d_{i + 1}": int(v) for i, v in enumerate(sales)})
            sales_rows.append(row)

    return {
        "calendar": cal.assign(date=cal["date"].dt.strftime("%Y-%m-%d")),
        "sales": pd.DataFrame(sales_rows),
        "prices": pd.concat(price_rows, ignore_index=True),
    }


def write(out_dir: Path, n_days: int = DEFAULT_DAYS, seed: int = 7, scale: float = 1.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = generate(n_days, seed, scale)
    frames["calendar"].to_csv(out_dir / "calendar.csv", index=False)
    frames["sales"].to_csv(out_dir / "sales_train_evaluation.csv", index=False)
    frames["prices"].to_csv(out_dir / "sell_prices.csv", index=False)
