"""PySpark data pipeline: load M5-schema files, subset, melt to long format, build features.

Forecasts are direct 28-day-ahead: every feature only uses information at least HORIZON days
older than the target day, so no recursive prediction and no leakage into the holdout window.
"""

from __future__ import annotations

import os
import sys
from functools import reduce
from pathlib import Path

import pandas as pd

from shelfspace.config import CATEGORIES, HORIZON, ITEMS_PER_CATEGORY, RAW_DIR, STORES

LAGS = [HORIZON, HORIZON + 7, HORIZON + 14, HORIZON + 21]  # same weekday, 4 weeks back


def get_spark(memory: str = "1g"):
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.master("local[1]")
        .appName("shelfspace")
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def load_long(
    spark,
    raw_dir: Path = RAW_DIR,
    stores=STORES,
    categories=CATEGORIES,
    items_per_category=ITEMS_PER_CATEGORY,
    last_days: int | None = None,
):
    """Wide M5 sales -> long frame (id, item, store, cat, date, d_idx, sales, price, calendar)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    sales = spark.read.csv(
        str(raw_dir / "sales_train_evaluation.csv"), header=True, inferSchema=True
    )
    cal = spark.read.csv(str(raw_dir / "calendar.csv"), header=True, inferSchema=True)
    prices = spark.read.csv(str(raw_dir / "sell_prices.csv"), header=True, inferSchema=True)

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales = sales.filter(
        F.col("store_id").isin(list(stores)) & F.col("cat_id").isin(list(categories))
    )

    if last_days:
        day_cols = day_cols[-last_days:]

    # melt first: summing 800+ wide columns builds an expression tree deep enough to overflow
    # the JVM stack
    stack = ", ".join(f"'{c}', `{c}`" for c in day_cols)
    long = sales.select(
        "id",
        "item_id",
        "cat_id",
        "store_id",
        F.expr(f"stack({len(day_cols)}, {stack}) as (d, sales)"),
    )

    # keep the best-selling items per category (same item set in every store)
    item_rank = (
        long.groupBy("cat_id", "item_id")
        .agg(F.sum("sales").alias("t"))
        .withColumn("rk", F.row_number().over(Window.partitionBy("cat_id").orderBy(F.desc("t"))))
    )
    keep = [(cat, n) for cat, n in items_per_category.items() if cat in categories]
    cond = reduce(
        lambda a, b: a | b,
        [(F.col("cat_id") == c) & (F.col("rk") <= n) for c, n in keep],
    )
    items = item_rank.filter(cond).select("item_id")
    long = long.join(items, "item_id")
    cal = cal.select(
        "d",
        F.to_date("date").alias("date"),
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        F.when(F.col("event_name_1").isNotNull() & (F.col("event_name_1") != ""), 1)
        .otherwise(0)
        .alias("is_event"),
    )
    long = long.join(cal, "d").join(prices, ["store_id", "item_id", "wm_yr_wk"], "left")
    long = long.withColumn("d_idx", F.regexp_extract("d", r"d_(\d+)", 1).cast("int"))
    return long.withColumnRenamed("sell_price", "price")


def build_features(long):
    """Lag / rolling / price / calendar features, all at least HORIZON days old for sales."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("id").orderBy("d_idx")
    df = long.withColumn("price", F.last("price", ignorenulls=True).over(w.rowsBetween(-60, 0)))
    for lag in LAGS:
        df = df.withColumn(f"lag_{lag}", F.lag("sales", lag).over(w))
    df = df.withColumn("seasonal_naive", sum(F.col(f"lag_{lag}") for lag in LAGS) / len(LAGS))
    df = df.withColumn(
        "roll_mean_28", F.avg("sales").over(w.rowsBetween(-(HORIZON + 27), -HORIZON))
    )
    df = df.withColumn("roll_mean_7", F.avg("sales").over(w.rowsBetween(-(HORIZON + 6), -HORIZON)))
    df = df.withColumn(
        "roll_std_28", F.stddev("sales").over(w.rowsBetween(-(HORIZON + 27), -HORIZON))
    )
    wi = Window.partitionBy("id")
    df = df.withColumn("price_ratio", F.col("price") / F.avg("price").over(wi))
    df = df.withColumn("dow", F.dayofweek("date")).withColumn("week", F.weekofyear("date"))
    keep = [
        "id",
        "item_id",
        "cat_id",
        "store_id",
        "date",
        "d_idx",
        "sales",
        "price",
        *[f"lag_{lag}" for lag in LAGS],
        "seasonal_naive",
        "roll_mean_28",
        "roll_mean_7",
        "roll_std_28",
        "price_ratio",
        "dow",
        "week",
        "month",
        "is_event",
    ]
    return df.select(*keep).filter(F.col(f"lag_{LAGS[-1]}").isNotNull())


def to_pandas(df) -> pd.DataFrame:
    pdf = df.toPandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    return pdf
