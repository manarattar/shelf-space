"""Shelf-space response model. ASSUMED, not measured: diminishing returns per category.

sales_i(s) = demand_i * (s / s_current_i) ** alpha. With alpha < 1 the response is concave, which
is what makes a non-trivial allocation (and a greedy optimiser that is provably optimal) possible.
"""

from __future__ import annotations

import numpy as np


def revenue(facings, current, demand, price, alpha: float) -> float:
    """Expected revenue over the demand window for a given allocation."""
    facings, current = np.asarray(facings, float), np.asarray(current, float)
    return float(np.sum(np.asarray(demand) * np.asarray(price) * (facings / current) ** alpha))
