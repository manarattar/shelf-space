from shelfspace.synthetic import generate

M5_SALES_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def test_synthetic_matches_m5_schema():
    frames = generate(n_days=60, scale=0.05)
    sales = frames["sales"]
    assert list(sales.columns[:6]) == M5_SALES_ID_COLS
    assert [c for c in sales.columns if c.startswith("d_")] == [f"d_{i}" for i in range(1, 61)]
    assert {"date", "wm_yr_wk", "d", "event_name_1", "wday", "month", "year"} <= set(
        frames["calendar"].columns
    )
    assert list(frames["prices"].columns) == [
        "store_id",
        "item_id",
        "wm_yr_wk",
        "sell_price",
    ]


def test_synthetic_is_deterministic_and_non_negative():
    a, b = generate(n_days=40, scale=0.05, seed=1), generate(n_days=40, scale=0.05, seed=1)
    assert a["sales"].equals(b["sales"])
    assert (a["sales"].filter(like="d_") >= 0).all().all()
    assert (a["prices"]["sell_price"] > 0).all()
