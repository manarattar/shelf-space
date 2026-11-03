"""End-to-end check of the Spark feature pipeline and a forecast quality gate (needs Java)."""

import numpy as np
import pytest

from shelfspace import data, forecast
from shelfspace.synthetic import write

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    write(raw, n_days=330, scale=0.2, seed=11)
    spark = data.get_spark("1g")
    try:
        long = data.load_long(spark, raw)
        feats = data.to_pandas(data.build_features(long))
    finally:
        spark.stop()
    return feats


def test_features_are_not_leaky(features):
    """Every lag feature must be built from sales at least HORIZON days older than the target."""
    one = features[features["id"] == features["id"].iloc[0]].sort_values("d_idx")
    sales = one.set_index("d_idx")["sales"]
    row = one.iloc[100]
    assert row["lag_28"] == sales.get(row["d_idx"] - 28, np.nan)
    assert row["lag_49"] == sales.get(row["d_idx"] - 49, np.nan)
    window = sales.loc[row["d_idx"] - 55 : row["d_idx"] - 28]
    assert row["roll_mean_28"] == pytest.approx(window.mean())


def test_features_have_no_missing_lags(features):
    assert features[["lag_28", "lag_35", "lag_42", "lag_49", "price"]].notna().all().all()


def test_quality_gate_model_beats_seasonal_naive(features):
    test, metrics = forecast.run(features, train_days=250)
    overall = metrics["overall"]
    assert overall["lightgbm"]["wape"] < overall["seasonal_naive"]["wape"]
    assert metrics["n_series"] > 0
