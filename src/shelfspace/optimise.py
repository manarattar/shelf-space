"""Allocate a fixed number of shelf facings across items to maximise expected revenue."""

from __future__ import annotations

import heapq

import numpy as np

from shelfspace.config import MAX_FACINGS, MIN_FACINGS


def _check(n: int, budget: int, lo: int, hi: int) -> None:
    if budget < n * lo or budget > n * hi:
        raise ValueError(f"budget {budget} infeasible for {n} items with facings in [{lo}, {hi}]")


def equal_split(n: int, budget: int, lo: int = MIN_FACINGS, hi: int = MAX_FACINGS) -> np.ndarray:
    """The stylised current planogram: the same number of facings for every item."""
    _check(n, budget, lo, hi)
    base, extra = divmod(budget, n)
    f = np.full(n, base)
    f[:extra] += 1
    return f


def proportional(weights, budget: int, lo: int = MIN_FACINGS, hi: int = MAX_FACINGS) -> np.ndarray:
    """Heuristic a category manager might use: facings proportional to expected revenue."""
    w = np.asarray(weights, float)
    _check(len(w), budget, lo, hi)
    raw = budget * w / w.sum() if w.sum() > 0 else np.full(len(w), budget / len(w))
    f = np.clip(np.floor(raw), lo, hi).astype(int)
    while f.sum() < budget:
        gap = np.where(f < hi, raw - f, -np.inf)
        f[int(np.argmax(gap))] += 1
    while f.sum() > budget:
        gap = np.where(f > lo, raw - f, np.inf)
        f[int(np.argmin(gap))] -= 1
    return f


def optimise(
    weights,
    current,
    alpha: float,
    budget: int,
    lo: int = MIN_FACINGS,
    hi: int = MAX_FACINGS,
) -> np.ndarray:
    """Maximise sum_i w_i * (f_i / current_i) ** alpha over integer facings, sum f = budget.

    The objective is separable and concave in each f_i, so adding facings one at a time to the item
    with the largest marginal gain is optimal (not just a heuristic).
    """
    w = np.asarray(weights, float)
    cur = np.asarray(current, float)
    n = len(w)
    _check(n, budget, lo, hi)
    f = np.full(n, lo, dtype=int)
    scale = w / cur**alpha

    def gain(i: int) -> float:
        return scale[i] * ((f[i] + 1) ** alpha - f[i] ** alpha)

    heap = [(-gain(i), i) for i in range(n)]
    heapq.heapify(heap)
    for _ in range(budget - n * lo):
        _, i = heapq.heappop(heap)
        f[i] += 1
        if f[i] < hi:
            heapq.heappush(heap, (-gain(i), i))
    return f
