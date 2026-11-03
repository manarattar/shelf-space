"""Write synthetic M5-schema data to data/raw/ (fallback when the Kaggle files are not present)."""

import argparse

from shelfspace.config import RAW_DIR
from shelfspace.synthetic import DEFAULT_DAYS, write

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    write(RAW_DIR, args.days, args.seed)
    print(f"wrote synthetic M5-schema files to {RAW_DIR}")
