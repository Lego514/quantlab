"""The forward ledger: frozen rules, recorded one completed bar at a time.

Everything here runs on synthetic bars with an injected clock and fetcher,
so "what day is it" and "what did the data say" are both under test control.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quantlab.forward import ledger, load_spec, reconcile, run
from quantlab.forward.runner import completed_bars, ledger_dir

SPEC = {
    "name": "test_book",
    "description": "two synthetic legs",
    "fee": 0.001, "slippage": 0.0005, "fill": "next_open",
    "legs": [
        {"symbol": "AAA", "market": "stock", "weight": 0.5, "strategy": "vol_momentum",
         "params": {"lookback": 20, "target_vol": 0.15}, "history": {"period": "2y"}},
        {"symbol": "BBB", "market": "crypto", "weight": 0.5, "strategy": "vol_momentum",
         "params": {"lookback": 20, "target_vol": 0.15}, "history": {"days": 730}},
    ],
    "expected": {"sharpe": 1.0, "sharpe_ci": [0.2, 1.8]},
}


def _bars(index, seed):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.002, 0.01, len(index)))
    opens = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, len(index)))
    df = pd.DataFrame({"open": opens, "close": close}, index=index)
    df.attrs["source"] = "synthetic"
    return df


DATA = {
    "AAA": _bars(pd.bdate_range(end="2026-03-06", periods=300), 1),   # weekdays
    "BBB": _bars(pd.date_range(end="2026-03-07", periods=420), 2),    # every day
}


def fake_fetch(leg):
    return DATA[leg.symbol].copy()


def at(day, hour=23):  # 23:00 UTC = 18:00 New York in early March
    return datetime(2026, 3, day, hour, tzinfo=timezone.utc)


@pytest.fixture
def book(tmp_path):
    spec_path = tmp_path / "test_book.json"
    spec_path.write_text(json.dumps(SPEC))
    return spec_path, tmp_path


def _rows(book, symbol):
    spec_path, root = book
    return ledger.read_leg(ledger_dir(load_spec(spec_path), root), symbol)


def test_first_run_writes_a_flat_anchor_and_locks_the_rules(book):
    spec_path, root = book
    assert run(spec_path, now=at(3), fetch=fake_fetch, root=root) == {"AAA": 1, "BBB": 1}
    a = _rows(book, "AAA").iloc[0]
    assert a["date"] == pd.Timestamp("2026-03-03")          # stock closed at 16:00 NY
    assert _rows(book, "BBB")["date"].iloc[0] == pd.Timestamp("2026-03-02")  # 3/3 still forming
    assert (a["equity"], a["held"], a["carried"]) == (1.0, 0.0, 0.0)
    lock = ledger.read_lock(ledger_dir(load_spec(spec_path), root))
    assert lock["sha256"] == load_spec(spec_path).sha256


def test_a_second_run_on_the_same_day_records_nothing(book):
    spec_path, root = book
    run(spec_path, now=at(3), fetch=fake_fetch, root=root)
    assert run(spec_path, now=at(3, 23), fetch=fake_fetch, root=root) == {"AAA": 0, "BBB": 0}


def test_the_ledger_matches_the_backtester(book):
    spec_path, root = book
    run(spec_path, now=at(3), fetch=fake_fetch, root=root)
    run(spec_path, now=at(4), fetch=fake_fetch, root=root)
    rows = _rows(book, "AAA")
    # Day two fills the anchor's target at the open.
    assert rows["held"].iloc[1] == rows["target_next"].iloc[0]
    assert rows["held"].iloc[1] > 0
    assert reconcile(spec_path, fetch=fake_fetch, root=root, now=at(4)) == []


def test_missed_runs_are_caught_up_and_marked_late(book):
    spec_path, root = book
    run(spec_path, now=at(3), fetch=fake_fetch, root=root)
    assert run(spec_path, now=at(6), fetch=fake_fetch, root=root) == {"AAA": 3, "BBB": 3}
    late = _rows(book, "AAA")["recorded_late"].tolist()
    assert late == [False, True, True, False]  # 3/4 and 3/5 late, 3/6 on time
    assert reconcile(spec_path, fetch=fake_fetch, root=root, now=at(6)) == []


def test_changed_rules_are_refused_but_documentation_is_not(book):
    spec_path, root = book
    run(spec_path, now=at(3), fetch=fake_fetch, root=root)

    edited = {**SPEC, "description": "reworded"}
    spec_path.write_text(json.dumps(edited))
    run(spec_path, now=at(4), fetch=fake_fetch, root=root)

    changed = json.loads(json.dumps(SPEC))
    changed["legs"][0]["params"]["lookback"] = 30
    spec_path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="changed"):
        run(spec_path, now=at(5), fetch=fake_fetch, root=root)


def test_a_failed_download_writes_nothing(book):
    spec_path, root = book
    run(spec_path, now=at(3), fetch=fake_fetch, root=root)

    def flaky(leg):
        if leg.symbol == "BBB":
            raise ConnectionError("rate limited")
        return fake_fetch(leg)

    with pytest.raises(ConnectionError):
        run(spec_path, now=at(4), fetch=flaky, root=root)
    assert len(_rows(book, "AAA")) == 1  # AAA was computed but never written


def test_forming_bars_are_not_recorded():
    stock = DATA["AAA"]
    before_settle = datetime(2026, 3, 4, 21, 30, tzinfo=timezone.utc)  # 16:30 New York
    assert completed_bars(stock, "stock", before_settle).index[-1] == pd.Timestamp("2026-03-03")
    assert completed_bars(stock, "stock", at(4)).index[-1] == pd.Timestamp("2026-03-04")
    crypto = DATA["BBB"]
    assert completed_bars(crypto, "crypto", at(4, 0)).index[-1] == pd.Timestamp("2026-03-03")


def test_dividend_adjusted_history_uses_the_fresh_previous_close(book):
    spec_path, root = book
    run(spec_path, now=at(3), fetch=fake_fetch, root=root)

    def adjusted(leg):
        df = fake_fetch(leg)
        if leg.symbol == "AAA":  # an ex-dividend day: Yahoo scales earlier prices down
            before = df.index < pd.Timestamp("2026-03-04")
            df.loc[before, ["open", "close"]] *= 0.99
        return df

    run(spec_path, now=at(4), fetch=adjusted, root=root)
    rows = _rows(book, "AAA")
    assert rows["prev_close"].iloc[1] == pytest.approx(rows["close"].iloc[0] * 0.99)
    assert reconcile(spec_path, fetch=adjusted, root=root, now=at(4)) == []


def test_reconcile_flags_data_the_ledger_did_not_see(book):
    # The engine check is not vacuous: if the data were revised after the
    # fact, the backtester and the ledger disagree and reconcile says so.
    spec_path, root = book
    run(spec_path, now=at(3), fetch=fake_fetch, root=root)
    run(spec_path, now=at(5), fetch=fake_fetch, root=root)

    def revised(leg):
        df = fake_fetch(leg)
        df.loc[pd.Timestamp("2026-03-05"), "close"] *= 1.02
        return df

    problems = reconcile(spec_path, fetch=revised, root=root, now=at(5))
    assert any("backtester equity differs" in p for p in problems)


def test_a_bar_the_source_fills_in_later_is_skipped_not_fatal(book):
    # Yahoo sometimes drops a day and adds it back later (it skipped BTC's
    # 2026-10-28 bar on the morning of the 29th).
    spec_path, root = book
    run(spec_path, now=at(3), fetch=fake_fetch, root=root)       # BBB anchor 3/2

    def gappy(leg):
        df = fake_fetch(leg)
        return df.drop(pd.Timestamp("2026-03-03")) if leg.symbol == "BBB" else df

    run(spec_path, now=at(5), fetch=gappy, root=root)             # BBB records 3/4
    assert reconcile(spec_path, fetch=fake_fetch, root=root, now=at(5)) == []
    run(spec_path, now=at(6), fetch=fake_fetch, root=root)
    assert _rows(book, "BBB")["date"].dt.day.tolist() == [2, 4, 5]  # 3/3 never inserted


def test_reconcile_catches_an_edited_row(book):
    spec_path, root = book
    run(spec_path, now=at(3), fetch=fake_fetch, root=root)
    run(spec_path, now=at(5), fetch=fake_fetch, root=root)
    path = ledger.leg_path(ledger_dir(load_spec(spec_path), root), "AAA")
    df = pd.read_csv(path)
    df.loc[2, "equity"] *= 1.01
    df.to_csv(path, index=False)
    problems = reconcile(spec_path, fetch=fake_fetch, root=root, now=at(5))
    assert any("equity does not follow" in p for p in problems)
