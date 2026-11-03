import numpy as np
import pandas as pd
import pytest

from shelfspace.forecast import mase, split_holdout, wape


def test_wape_known_values():
    assert wape([10, 10], [8, 12]) == pytest.approx(0.2)
    assert wape([10, 10], [10, 10]) == 0.0


def test_wape_all_zero_actuals_does_not_divide_by_zero():
    assert np.isfinite(wape([0, 0], [1, 1]))


def test_split_holdout_uses_last_horizon_days():
    df = pd.DataFrame({"d_idx": range(1, 101)})
    train, test = split_holdout(df, horizon=28)
    assert train["d_idx"].max() == 72 and test["d_idx"].min() == 73 and len(test) == 28


def test_mase_scale_comes_from_training_data():
    pattern = np.tile([1, 2, 3, 4, 5, 6, 7], 10)
    test = pd.DataFrame(
        {
            "id": "a",
            "d_idx": range(70, 77),
            "sales": [1, 2, 3, 4, 5, 6, 7],
            "pred": [2, 3, 4, 5, 6, 7, 8],
        }
    )
    # a pure weekly pattern has zero in-sample naive error, so the series has no scale and is dropped
    pure = pd.DataFrame({"id": "a", "d_idx": range(70), "sales": pattern})
    assert np.isnan(mase(pure, test, "pred"))
    # with weekly-naive MAE of 1 in training and a forecast MAE of 1, MASE is exactly 1
    noisy = pure.assign(sales=pattern + np.tile([0, 1], 35) * (np.arange(70) // 7 % 2))
    scale = np.abs(noisy["sales"].to_numpy()[7:] - noisy["sales"].to_numpy()[:-7]).mean()
    assert mase(noisy, test, "pred") == pytest.approx(1 / scale)
