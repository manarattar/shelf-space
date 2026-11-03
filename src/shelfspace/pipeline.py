"""Offline training run: Spark features -> LightGBM forecast -> small result files for the app.

    python -m shelfspace.pipeline

Reads M5-schema CSVs from data/raw (real M5 or synthetic), writes to results/.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from shelfspace import data, forecast, planning
from shelfspace.config import INTERIM_DIR, RAW_DIR, RESULTS_DIR


def holdout_to_demand(test: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the holdout window to one row per store x item."""
    g = test.groupby(["store_id", "cat_id", "item_id"], as_index=False).agg(
        forecast=("lgbm", "sum"),
        baseline=("baseline", "sum"),
        actual=("sales", "sum"),
        price=("price", "mean"),
    )
    return g.round({"forecast": 2, "baseline": 2, "price": 2})


def holdout_daily(test: pd.DataFrame) -> pd.DataFrame:
    g = test.groupby(["store_id", "cat_id", "date"], as_index=False).agg(
        actual=("sales", "sum"), baseline=("baseline", "sum"), lgbm=("lgbm", "sum")
    )
    return g.round(2)


def main(reuse: bool = False) -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    cache = INTERIM_DIR / "features.parquet"
    if reuse and cache.exists():
        features = pd.read_parquet(cache)
    else:
        spark = data.get_spark()
        try:
            long = data.load_long(spark, RAW_DIR)
            features = data.to_pandas(data.build_features(long))
        finally:
            spark.stop()
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache, index=False)
    print(f"features: {len(features):,} rows in {time.time() - t0:.0f}s")

    test, metrics = forecast.run(features)
    demand = holdout_to_demand(test)
    # the shelf decision uses 28-day item totals, where daily count noise averages out
    metrics["item_28d_totals"] = {
        "overall": {
            "seasonal_naive": forecast.wape(demand["actual"], demand["baseline"]),
            "lightgbm": forecast.wape(demand["actual"], demand["forecast"]),
        },
        "by_category": {
            cat: {
                "seasonal_naive": forecast.wape(g["actual"], g["baseline"]),
                "lightgbm": forecast.wape(g["actual"], g["forecast"]),
            }
            for cat, g in demand.groupby("cat_id")
        },
    }
    metrics["data_source"] = (
        (RAW_DIR / "DATA_SOURCE.txt").read_text().strip()
        if (RAW_DIR / "DATA_SOURCE.txt").exists()
        else "unknown"
    )

    demand.to_parquet(RESULTS_DIR / "demand_forecast.parquet", index=False)
    holdout_daily(test).to_parquet(RESULTS_DIR / "holdout_daily.parquet", index=False)
    planning.summary(demand).round(4).to_csv(RESULTS_DIR / "plan_summary.csv", index=False)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["overall"], indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(reuse="--reuse" in sys.argv)
