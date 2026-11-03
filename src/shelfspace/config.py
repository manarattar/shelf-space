"""Shared constants. The response exponents are ASSUMPTIONS, see docs and README."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim"
RESULTS_DIR = Path(os.environ.get("SHELFSPACE_RESULTS", ROOT / "results"))

HORIZON = 28  # days ahead the forecast is made (direct multi-horizon, no recursion)

# Demo subset (same ids as M5 Walmart): 3 stores, 3 categories
STORES = ["CA_1", "CA_2", "TX_1"]
CATEGORIES = ["FOODS", "HOUSEHOLD", "HOBBIES"]
ITEMS_PER_CATEGORY = {"FOODS": 120, "HOUSEHOLD": 100, "HOBBIES": 80}

# ASSUMPTION: sales respond to shelf facings with diminishing returns,
# sales_i(s) = demand_i * (s / s_current_i) ** alpha_category.
# alpha in (0, 1): 1 would be "twice the space, twice the sales".
# Impulse-like categories react more to space than staples. Not measured, set by hand.
ALPHA = {"FOODS": 0.35, "HOUSEHOLD": 0.25, "HOBBIES": 0.15}

AVG_FACINGS_PER_ITEM = 2.5  # shelf budget per category per store = this * item count
MIN_FACINGS = 1
MAX_FACINGS = 8
