"""Quarterly review of the forward ledgers: percentile, kill rules, report."""
import json

import numpy as np
import pandas as pd
import pytest

from quantlab.forward import evaluate_all, run_all
from quantlab.forward.evaluate import kill_reasons, percentile_of
from test_forward import SPEC, at, fake_fetch


def test_percentile_places_the_live_return_in_the_backtest_distribution():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert percentile_of(-0.30, reference, 20) == 0
    assert percentile_of(0.50, reference, 20) == 100
    assert 30 < percentile_of(0.02, reference, 20) < 70  # about the median 20-day run


def test_two_quarter_ends_below_the_fifth_percentile_kill():
    assert kill_reasons([3.0, 4.0], -0.05, -0.20)
    assert not kill_reasons([3.0, 12.0], -0.05, -0.20)  # not in a row
    assert not kill_reasons([3.0], -0.05, -0.20)        # one quarter is not enough


def test_a_drawdown_past_one_and_a_half_times_the_backtest_kills():
    assert kill_reasons([50.0], -0.31, -0.20)
    assert not kill_reasons([50.0], -0.29, -0.20)


def test_a_recorded_review_covers_every_started_ledger(tmp_path):
    specs = tmp_path / "forward" / "specs"
    specs.mkdir(parents=True)
    (specs / "book.json").write_text(json.dumps({**SPEC, "name": "book"}))
    run_all(now=at(3), fetch=fake_fetch, root=tmp_path)
    run_all(now=at(6), fetch=fake_fetch, root=tmp_path)

    text = evaluate_all(root=tmp_path, fetch=fake_fetch, record=True, today="2026-03-31")
    assert "| book |" in text and "Recorded quarter-end review" in text
    history = json.loads((tmp_path / "forward" / "reviews" / "history.json").read_text())
    assert history["book"][0]["date"] == "2026-03-31"
    assert 0 <= history["book"][0]["percentile"] <= 100


def test_an_informal_review_records_nothing(tmp_path):
    specs = tmp_path / "forward" / "specs"
    specs.mkdir(parents=True)
    (specs / "book.json").write_text(json.dumps({**SPEC, "name": "book"}))
    run_all(now=at(3), fetch=fake_fetch, root=tmp_path)
    text = evaluate_all(root=tmp_path, fetch=fake_fetch, record=False)
    assert "informational only" in text
    assert not (tmp_path / "forward" / "reviews" / "history.json").exists()
