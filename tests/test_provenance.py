"""Every result says where its data came from.

Binance answers HTTP 451 to US addresses, and fetch_crypto used to fall back
to Yahoo silently, so nobody could tell which source a backtest ran on.
"""
import pandas as pd
import pytest

from quantlab.data import cache, crypto
from quantlab.engine import Backtester
from quantlab.strategies import BuyHold


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


def _frame():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({"open": 1.0, "close": 1.0}, index=idx)


def _blocked(*_):
    raise RuntimeError("451 Client Error")


def test_fallback_is_labelled_and_cached_under_its_source(isolated_cache, monkeypatch, capsys):
    monkeypatch.setattr(crypto, "_fetch_binance", _blocked)
    monkeypatch.setattr(crypto, "_fetch_yahoo", lambda *_: _frame())

    df = crypto.fetch_crypto("BTC-USD", days=10)
    assert df.attrs["source"] == "yahoo"
    assert "falling back to yahoo" in capsys.readouterr().err
    assert any("yahoo" in p.name for p in isolated_cache.iterdir())

    again = crypto.fetch_crypto("BTC-USD", days=10)  # from cache this time
    assert again.attrs["source"] == "yahoo"


def test_binance_data_is_labelled_binance(isolated_cache, monkeypatch):
    monkeypatch.setattr(crypto, "_fetch_binance", lambda *_: _frame())
    assert crypto.fetch_crypto("BTC-USD", days=10).attrs["source"] == "binance"


def test_backtest_results_carry_the_source():
    prices = _frame()
    prices.attrs["source"] = "yahoo"
    result = Backtester().run(prices, BuyHold())
    assert result.data_source == "yahoo"
    assert "data: yahoo" in result.summary()
