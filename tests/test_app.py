"""Smoke test: the Streamlit page renders and responds to its controls without exceptions."""

from pathlib import Path

import pytest

from shelfspace.config import RESULTS_DIR

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

pytestmark = pytest.mark.skipif(
    not (RESULTS_DIR / "demand_forecast.parquet").exists(),
    reason="results/ not generated (run python -m shelfspace.pipeline)",
)


def _run():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=60).run()


def test_page_renders_without_exceptions():
    at = _run()
    assert not at.exception
    assert at.title[0].value == "Shelf Space Allocator"
    assert len(at.selectbox) == 2 and len(at.slider) == 2


def test_changing_category_and_budget_still_renders():
    at = _run()
    at.selectbox[1].select("HOBBIES").run()
    assert not at.exception
    at.slider[0].set_value(at.slider[0].min).run()
    assert not at.exception


def test_proportional_comparison_renders():
    at = _run()
    at.radio[0].set_value(at.radio[0].options[1]).run()
    assert not at.exception
