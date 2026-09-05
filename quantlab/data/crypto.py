"""Crypto OHLCV data.

Primary source: Binance public klines API (no API key needed, fine-grained).
Fallback: Yahoo Finance (e.g. BTC-USD) if Binance is unreachable/geo-blocked.
"""
import time

import pandas as pd
import requests

from . import cache

BINANCE_URL = "https://api.binance.com/api/v3/klines"

_INTERVAL_MAP = {  # our interval -> (binance interval, yahoo interval)
    "1h": ("1h", "1h"),
    "4h": ("4h", None),
    "1d": ("1d", "1d"),
    "1w": ("1w", "1wk"),
}


def _to_binance_symbol(symbol: str) -> str:
    """BTC-USD / BTC/USDT / btcusdt -> BTCUSDT"""
    s = symbol.upper().replace("-", "").replace("/", "")
    if s.endswith("USD") and not s.endswith("USDT"):
        s += "T"
    return s


def _fetch_binance(symbol: str, interval: str, limit_days: int) -> pd.DataFrame:
    binance_interval = _INTERVAL_MAP[interval][0]
    end = int(time.time() * 1000)
    start = end - limit_days * 86_400_000
    rows = []
    while start < end:
        resp = requests.get(BINANCE_URL, params={
            "symbol": _to_binance_symbol(symbol),
            "interval": binance_interval,
            "startTime": start,
            "limit": 1000,
        }, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][6] + 1  # close time of last candle + 1ms
        if len(batch) < 1000:
            break
    if not rows:
        raise RuntimeError(f"binance returned no data for {symbol}")
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbb", "tbq", "ignore"])
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def _fetch_yahoo(symbol: str, interval: str, limit_days: int) -> pd.DataFrame:
    from .stocks import fetch_stock
    yahoo_interval = _INTERVAL_MAP[interval][1]
    if yahoo_interval is None:
        raise RuntimeError(f"interval {interval} not supported by yahoo fallback")
    if "-" not in symbol and "/" not in symbol:
        symbol = f"{symbol.upper()}-USD"
    symbol = symbol.replace("/", "-").upper().replace("USDT", "USD")
    period = "max" if limit_days > 1800 else f"{max(limit_days, 30)}d"
    return fetch_stock(symbol, period=period, interval=yahoo_interval)


def fetch_crypto(symbol: str, interval: str = "1d", days: int = 1460) -> pd.DataFrame:
    """Fetch crypto OHLCV. symbol like "BTC-USD", "ETHUSDT", "SOL/USDT"."""
    if interval not in _INTERVAL_MAP:
        raise ValueError(f"interval must be one of {list(_INTERVAL_MAP)}")
    key = f"crypto_{symbol}_{interval}_{days}"
    cached = cache.load(key)
    if cached is not None:
        return cached
    try:
        df = _fetch_binance(symbol, interval, days)
    except Exception:
        df = _fetch_yahoo(symbol, interval, days)
    cache.save(key, df)
    return df
