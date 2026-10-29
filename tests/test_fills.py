"""When does a signal get filled?

The engine used to earn close(t) -> close(t+1) on a signal computed at
close(t): in effect it bought at the very price that produced the signal.
It now trades at the next bar's open, so an overnight gap after the signal
belongs to whatever position was held BEFORE the signal.
"""
import numpy as np
import pandas as pd
import pytest

from quantlab.engine import Backtester
from quantlab.research.controls import FixedPositions

IDX = pd.bdate_range("2024-01-01", periods=6)


def _gap_prices() -> pd.DataFrame:
    """Flat at 100, then bar 3 gaps up to open at 110 and closes at 110."""
    close = np.array([100, 100, 100, 110, 110, 110], dtype=float)
    opens = np.array([100, 100, 100, 110, 110, 110], dtype=float)
    return pd.DataFrame({"open": opens, "close": close}, index=IDX)


def _long_from(bar: int) -> FixedPositions:
    return FixedPositions(pd.Series([0.0] * bar + [1.0] * (len(IDX) - bar), index=IDX))


def test_signal_at_the_close_before_a_gap_misses_the_gap():
    # Signal at bar 2's close; the jump happens at bar 3's open. Trading at
    # the next open means buying at 110, after the jump.
    result = Backtester(fee=0, slippage=0).run(_gap_prices(), _long_from(2))
    assert result.fill == "next_open"
    assert result.equity.iloc[-1] == pytest.approx(1.0)


def test_close_fill_would_have_captured_it():
    # The old behaviour: filled at bar 2's close (100), so it earns the gap.
    result = Backtester(fee=0, slippage=0, fill="close").run(_gap_prices(), _long_from(2))
    assert result.fill == "close"
    assert result.equity.iloc[-1] == pytest.approx(1.1)


def test_carried_position_earns_the_overnight_gap():
    # Long since bar 0, exit signal at bar 2's close: the exit trades at bar
    # 3's open, after the gap, so the gap is still earned.
    pos = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], index=IDX)
    result = Backtester(fee=0, slippage=0).run(_gap_prices(), FixedPositions(pos))
    assert result.equity.iloc[-1] == pytest.approx(1.1)
    assert result.trade_returns == [pytest.approx(0.1)]


def test_falls_back_to_close_without_opens():
    prices = _gap_prices()[["close"]]
    result = Backtester().run(prices, _long_from(2))
    assert result.fill == "close"


def test_per_bar_accounting_matches_hand_computation():
    close = pd.Series([100.0, 102.0, 101.0, 105.0], index=IDX[:4])
    opens = pd.Series([100.0, 101.0, 103.0, 104.0], index=IDX[:4])
    pos = pd.Series([1.0, 0.5, 0.5, 0.0], index=IDX[:4])
    c = 0.001 + 0.0005
    result = Backtester().run(pd.DataFrame({"open": opens, "close": close}),
                              FixedPositions(pos))
    # bar1: buy 1.0 at the open, earn open->close
    e1 = (1 - 1.0 * c) * (102 / 101)
    # bar2: carry 1.0 over the gap, cut to 0.5 at the open, earn 0.5 of the session
    e2 = e1 * (1 + 1.0 * (103 / 102 - 1)) * (1 - 0.5 * c) * (1 + 0.5 * (101 / 103 - 1))
    # bar3: carry 0.5 over the gap, hold 0.5
    e3 = e2 * (1 + 0.5 * (104 / 101 - 1)) * (1 + 0.5 * (105 / 104 - 1))
    np.testing.assert_allclose(result.equity.to_numpy(), [1.0, e1, e2, e3], rtol=1e-12)
