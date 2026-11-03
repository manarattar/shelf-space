"""Simulated store-level A/B test for a shelf-space change (difference of pre/post changes).

Metric per store: mean weekly revenue after / mean before - 1. Comparing that change between
treatment and control stores removes the (large) differences in store size. Everything here is a
simulation of how a real test would be run and powered, not a result about real stores.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _store_sd(weekly_cv: float, pre_weeks: int, post_weeks: int) -> float:
    return weekly_cv * math.sqrt(1 / pre_weeks + 1 / post_weeks)


def simulate_experiment(
    true_uplift: float,
    n_per_arm: int = 30,
    pre_weeks: int = 8,
    post_weeks: int = 8,
    store_cv: float = 0.3,
    weekly_cv: float = 0.12,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    rng = np.random.default_rng(seed)
    n = 2 * n_per_arm
    level = np.exp(rng.normal(0, store_cv, n))
    treated = np.arange(n) >= n_per_arm
    noise = np.exp(rng.normal(-(weekly_cv**2) / 2, weekly_cv, (n, pre_weeks + post_weeks)))
    rev = level[:, None] * noise
    rev[treated, pre_weeks:] *= 1 + true_uplift
    change = rev[:, pre_weeks:].mean(axis=1) / rev[:, :pre_weeks].mean(axis=1) - 1
    a, b = change[treated], change[~treated]
    diff = float(a.mean() - b.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    crit = stats.t.ppf(1 - alpha / 2, df)
    p = float(2 * (1 - stats.t.cdf(abs(diff) / se, df)))
    return {
        "estimate": diff,
        "ci_low": diff - crit * se,
        "ci_high": diff + crit * se,
        "p_value": p,
        "significant": p < alpha,
        "n_per_arm": n_per_arm,
    }


def power(
    true_uplift: float,
    n_per_arm: int,
    pre_weeks: int = 8,
    post_weeks: int = 8,
    weekly_cv: float = 0.12,
    alpha: float = 0.05,
) -> float:
    """Analytic power of the two-sided test (normal approximation)."""
    se = _store_sd(weekly_cv, pre_weeks, post_weeks) * math.sqrt(2 / n_per_arm)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(abs(true_uplift) / se - z))


def required_n_per_arm(
    true_uplift: float,
    target_power: float = 0.8,
    pre_weeks: int = 8,
    post_weeks: int = 8,
    weekly_cv: float = 0.12,
    alpha: float = 0.05,
) -> int:
    """Stores needed in each arm to detect true_uplift with the target power."""
    if true_uplift == 0:
        raise ValueError("cannot power a test for a zero effect")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(target_power)
    sd = _store_sd(weekly_cv, pre_weeks, post_weeks)
    return max(2, math.ceil(2 * (z * sd / true_uplift) ** 2))


def minimum_detectable_effect(
    n_per_arm: int,
    target_power: float = 0.8,
    pre_weeks: int = 8,
    post_weeks: int = 8,
    weekly_cv: float = 0.12,
    alpha: float = 0.05,
) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(target_power)
    return float(z * _store_sd(weekly_cv, pre_weeks, post_weeks) * math.sqrt(2 / n_per_arm))
