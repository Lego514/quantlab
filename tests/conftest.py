"""Shared fixtures for the quantlab test suite.

The engine's correctness claims are all about *timing* and *cost*, so the
fixtures here are deterministic synthetic price series rather than downloaded
market data: a test that depends on the network, or on whatever BTC did last
week, cannot tell you whether the backtester is right.
"""
import numpy as np
import pandas as pd
import pytest


def _frame(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    """OHLCV frame around a close series. High/low bracket the close so that
    range-based strategies (Donchian, Bollinger) have something sane to read."""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(len(close), 1_000_000.0),
        },
        index=index,
    )


@pytest.fixture
def stock_index() -> pd.DatetimeIndex:
    """Weekdays only — the calendar a stock series has (~252 bars/year)."""
    return pd.bdate_range("2020-01-01", periods=1000)


@pytest.fixture
def crypto_index() -> pd.DatetimeIndex:
    """Every calendar day, weekends included (~365 bars/year)."""
    return pd.date_range("2020-01-01", periods=1000, freq="D")


@pytest.fixture
def trending_prices(stock_index) -> pd.DataFrame:
    """Steady uptrend: any trend-following strategy should be long almost all
    of it, which makes cost and timing effects easy to isolate."""
    close = 100 * (1.0005 ** np.arange(len(stock_index)))
    return _frame(stock_index, close)


@pytest.fixture
def flat_prices(stock_index) -> pd.DataFrame:
    """Constant price. Any nonzero return here is a bug: with no price change
    a strategy can only lose money to costs, never make it."""
    return _frame(stock_index, np.full(len(stock_index), 100.0))


@pytest.fixture
def spike_future(trending_prices):
    """Return a copy of the prices with one far-future bar multiplied.

    This is the look-ahead probe: results computed over bars BEFORE the spike
    must be byte-identical whether or not the spike exists. If they differ, the
    engine is letting information travel backwards in time.
    """

    def _spike(at: int = 800, factor: float = 5.0) -> pd.DataFrame:
        out = trending_prices.copy()
        out.iloc[at:, out.columns.get_indexer(["open", "high", "low", "close"])] *= factor
        return out

    return _spike
