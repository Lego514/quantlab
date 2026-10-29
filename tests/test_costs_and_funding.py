"""Venue cost presets, perpetual funding and short borrow."""
import pandas as pd
import pytest

from quantlab.data.funding import _to_okx_swap, funding_per_bar
from quantlab.engine import VENUES, Backtester, venue
from quantlab.research.controls import FixedPositions

DAYS = pd.date_range("2024-01-01", periods=5, freq="D")
FLAT = pd.DataFrame({"open": 100.0, "close": 100.0}, index=DAYS)


def test_presets_resolve_and_unknown_names_fail():
    assert venue("coinbase_advanced").fee == pytest.approx(0.006)
    assert all(v.note for v in VENUES.values())
    with pytest.raises(ValueError):
        venue("nowhere")


def _run(pos: float, funding=None, **kw) -> float:
    positions = FixedPositions(pd.Series(pos, index=DAYS))
    result = Backtester(fee=0, slippage=0, allow_short=True, **kw).run(FLAT, positions,
                                                                         funding=funding)
    return result.equity.iloc[-1]


def test_longs_pay_positive_funding_and_shorts_receive_it():
    funding = pd.Series(0.001, index=DAYS)
    # Positions are traded from bar 1, so funding is charged on bars 1..4.
    assert _run(1.0, funding) == pytest.approx(0.999 ** 4)
    assert _run(-1.0, funding) == pytest.approx(1.001 ** 4)


def test_borrow_is_charged_on_shorts_only_by_calendar_day():
    assert _run(1.0, borrow_rate=0.365) == pytest.approx(1.0)
    assert _run(-1.0, borrow_rate=0.365) == pytest.approx(0.999 ** 4)


def test_funding_settlements_are_summed_into_their_bars():
    rates = pd.Series([0.001, 0.002, 0.003],
                      index=pd.to_datetime(["2024-01-02 00:00", "2024-01-02 08:00",
                                            "2024-01-03 16:00"]))
    per_bar, coverage = funding_per_bar(rates, DAYS)
    assert per_bar.tolist() == pytest.approx([0, 0.003, 0.003, 0, 0])
    assert coverage == pytest.approx(4 / 5)  # no data for the first day


@pytest.mark.parametrize("symbol", ["BTC-USD", "BTC/USDT", "btcusdt", "BTC"])
def test_symbols_map_to_okx_swaps(symbol):
    assert _to_okx_swap(symbol) == "BTC-USDT-SWAP"
