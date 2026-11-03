"""Demand forecast: seasonal-naive baseline vs LightGBM, evaluated on the last HORIZON days."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shelfspace.config import HORIZON

FEATURES = [
    "lag_28",
    "lag_35",
    "lag_42",
    "lag_49",
    "seasonal_naive",
    "roll_mean_28",
    "roll_mean_7",
    "roll_std_28",
    "price",
    "price_ratio",
    "dow",
    "week",
    "month",
    "is_event",
    "cat_code",
    "store_code",
]


def add_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cat_code"] = df["cat_id"].astype("category").cat.codes
    df["store_code"] = df["store_id"].astype("category").cat.codes
    return df


def split_holdout(df: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["d_idx"].max() - horizon
    return df[df["d_idx"] <= cutoff], df[df["d_idx"] > cutoff]


def wape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9))


def mase(train: pd.DataFrame, test: pd.DataFrame, pred_col: str) -> float:
    """Mean over series of MAE / in-sample weekly-naive MAE (scale from training data only)."""
    scale = (
        train.sort_values("d_idx")
        .groupby("id")["sales"]
        .apply(lambda s: np.abs(s.to_numpy()[7:] - s.to_numpy()[:-7]).mean())
    )
    mae = test.assign(_e=(test["sales"] - test[pred_col]).abs()).groupby("id")["_e"].mean()
    ratio = (mae / scale.reindex(mae.index)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.mean())


def fit_lgbm(train: pd.DataFrame, n_estimators: int = 400, seed: int = 0):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.2,
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=2,
    )
    model.fit(train[FEATURES], train["sales"])
    return model


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Metrics overall and per category for the baseline and the model (columns baseline, lgbm)."""
    out: dict = {"overall": {}, "by_category": {}}
    for name, col in [("seasonal_naive", "baseline"), ("lightgbm", "lgbm")]:
        out["overall"][name] = {
            "wape": wape(test["sales"], test[col]),
            "mase": mase(train, test, col),
        }
    for cat, g in test.groupby("cat_id"):
        gt = train[train["cat_id"] == cat]
        out["by_category"][cat] = {
            name: {"wape": wape(g["sales"], g[col]), "mase": mase(gt, g, col)}
            for name, col in [("seasonal_naive", "baseline"), ("lightgbm", "lgbm")]
        }
    return out


def run(features: pd.DataFrame, train_days: int = 450) -> tuple[pd.DataFrame, dict]:
    """Fit on the trailing train_days before the holdout, predict the holdout, return metrics."""
    df = add_codes(features)
    train, test = split_holdout(df)
    train = train[train["d_idx"] > train["d_idx"].max() - train_days].dropna(subset=FEATURES)
    test = test.copy()
    model = fit_lgbm(train)
    test["lgbm"] = np.clip(model.predict(test[FEATURES]), 0, None)
    test["baseline"] = test["seasonal_naive"]
    metrics = evaluate(train, test)
    metrics["horizon_days"] = HORIZON
    metrics["n_series"] = int(test["id"].nunique())
    metrics["holdout_start"] = str(test["date"].min().date())
    metrics["holdout_end"] = str(test["date"].max().date())
    return test, metrics
