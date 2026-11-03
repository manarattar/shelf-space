import itertools

import numpy as np
import pytest

from shelfspace.optimise import equal_split, optimise, proportional
from shelfspace.response import revenue


def test_equal_split_sums_to_budget_and_is_balanced():
    f = equal_split(7, 20)
    assert f.sum() == 20
    assert f.max() - f.min() <= 1


@pytest.mark.parametrize("seed", range(5))
def test_optimiser_respects_constraints(seed):
    rng = np.random.default_rng(seed)
    n, budget = 30, 75
    w = rng.lognormal(0, 1, n)
    cur = equal_split(n, budget)
    f = optimise(w, cur, 0.3, budget, lo=1, hi=8)
    assert f.sum() == budget
    assert f.min() >= 1 and f.max() <= 8


@pytest.mark.parametrize("seed", range(5))
def test_optimiser_never_worse_than_baselines(seed):
    rng = np.random.default_rng(seed)
    n, budget = 40, 100
    demand, price = rng.lognormal(2, 1, n), rng.uniform(1, 10, n)
    w = demand * price
    cur = equal_split(n, budget)
    best = revenue(optimise(w, cur, 0.3, budget), cur, demand, price, 0.3)
    assert best >= revenue(cur, cur, demand, price, 0.3) - 1e-9
    assert best >= revenue(proportional(w, budget), cur, demand, price, 0.3) - 1e-9


def test_optimiser_matches_brute_force_on_small_case():
    rng = np.random.default_rng(3)
    n, budget, lo, hi = 4, 9, 1, 5
    demand, price = rng.lognormal(1, 1, n), rng.uniform(1, 5, n)
    cur = equal_split(n, budget, lo, hi)
    best = max(
        (c for c in itertools.product(range(lo, hi + 1), repeat=n) if sum(c) == budget),
        key=lambda c: revenue(c, cur, demand, price, 0.4),
    )
    got = optimise(demand * price, cur, 0.4, budget, lo, hi)
    assert revenue(got, cur, demand, price, 0.4) == pytest.approx(
        revenue(best, cur, demand, price, 0.4)
    )


def test_infeasible_budget_raises():
    with pytest.raises(ValueError):
        optimise([1, 2, 3], [1, 1, 1], 0.3, budget=2, lo=1, hi=5)
    with pytest.raises(ValueError):
        equal_split(3, 100, lo=1, hi=5)


def test_proportional_respects_bounds_and_budget():
    f = proportional([100, 1, 1, 1], 12, lo=1, hi=6)
    assert f.sum() == 12 and f.min() >= 1 and f.max() <= 6
    assert f[0] == f.max()


def test_revenue_scales_with_alpha():
    cur = np.array([2, 2])
    assert revenue([4, 4], cur, [10, 10], [1, 1], 1.0) == pytest.approx(40)
    assert revenue([4, 4], cur, [10, 10], [1, 1], 0.0) == pytest.approx(20)
