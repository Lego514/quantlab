"""Combining legs that trade on different calendars.

A stock trades on weekdays; Bitcoin trades every day. The 2026 case study
used to take each leg's daily returns first and then drop every row where
either was missing, which quietly discarded Bitcoin's weekend returns.
"""
import pandas as pd
import pytest

from quantlab.research import leg_returns

DAYS = pd.date_range("2024-01-01", "2024-01-14", freq="D")  # Mon Jan 1 .. Sun Jan 14
WEEKDAYS = DAYS[DAYS.dayofweek < 5]


def _weekend_only_crypto() -> pd.Series:
    """Flat all week, then doubles over Saturday and Sunday."""
    crypto = pd.Series(1.0, index=DAYS)
    crypto[pd.Timestamp("2024-01-06"):] = 1.5
    crypto[pd.Timestamp("2024-01-07"):] = 2.0
    return crypto


def test_weekend_move_of_a_24_7_leg_lands_on_monday():
    rets = leg_returns({"stock": pd.Series(1.0, index=WEEKDAYS),
                        "crypto": _weekend_only_crypto()})

    assert rets.loc["2024-01-08", "crypto"] == pytest.approx(1.0)  # Fri 1.0 -> Mon 2.0
    assert (1 + rets["crypto"]).prod() == pytest.approx(2.0)


def test_returns_are_only_reported_on_days_every_leg_trades():
    rets = leg_returns({"stock": pd.Series(1.0, index=WEEKDAYS),
                        "crypto": _weekend_only_crypto()})

    assert not (rets.index.dayofweek >= 5).any()


def test_the_old_construction_loses_the_weekend():
    # Pins down the bug leg_returns exists to prevent: returns first, then
    # dropna, and the doubling simply disappears.
    old = pd.DataFrame({"stock": pd.Series(1.0, index=WEEKDAYS).pct_change(),
                        "crypto": _weekend_only_crypto().pct_change()}).dropna()

    assert (1 + old["crypto"]).prod() == pytest.approx(1.0)
