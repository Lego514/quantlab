"""Crypto OHLCV data.

Primary source: Binance public klines API (no API key needed, fine-grained).
Fallback: Yahoo Finance (e.g. BTC-USD) if Binance is unreachable/geo-blocked.
"""
import sys
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


def _fetch_yahoo(symbol: str, interval: str, limit_days: int,
                 use_cache: bool = True) -> pd.DataFrame:
    from .stocks import fetch_stock
    yahoo_interval = _INTERVAL_MAP[interval][1]
    if yahoo_interval is None:
        raise RuntimeError(f"interval {interval} not supported by yahoo fallback")
    if "-" not in symbol and "/" not in symbol:
        symbol = f"{symbol.upper()}-USD"
    symbol = symbol.replace("/", "-").upper().replace("USDT", "USD")
    period = "max" if limit_days > 1800 else f"{max(limit_days, 30)}d"
    return fetch_stock(symbol, period=period, interval=yahoo_interval, use_cache=use_cache)


SOURCES = ("binance", "yahoo")


def _cache_key(source: str, symbol: str, interval: str, days: int) -> str:
    return f"crypto_{source}_{symbol}_{interval}_{days}"


def fetch_crypto(symbol: str, interval: str = "1d", days: int = 1460,
                 source: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Fetch crypto OHLCV. symbol like "BTC-USD", "ETHUSDT", "SOL/USDT".

    Binance answers HTTP 451 to US addresses, and the fallback to Yahoo used
    to be silent: the 2026 case study's BTC history turned out to be Yahoo's
    (it starts in 2014, three years before Binance existed). The source now
    travels with the data in `df.attrs["source"]`, is part of the cache key,
    and a fallback says why it happened.

    source pins one provider ("binance" or "yahoo") instead of falling back,
    so a record kept over months never mixes two price series.
    use_cache=False always downloads.
    """
    if interval not in _INTERVAL_MAP:
        raise ValueError(f"interval must be one of {list(_INTERVAL_MAP)}")
    if source is not None and source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if use_cache:
        for src in (source,) if source else SOURCES:
            cached = cache.load(_cache_key(src, symbol, interval, days))
            if cached is not None:
                cached.attrs["source"] = src
                return cached
    if source == "yahoo":
        df = _fetch_yahoo(symbol, interval, days, use_cache)
    elif source == "binance":
        df = _fetch_binance(symbol, interval, days)
    else:
        try:
            df, source = _fetch_binance(symbol, interval, days), "binance"
        except Exception as e:
            print(f"  ! binance unavailable for {symbol} ({str(e)[:80]}); "
                  f"falling back to yahoo", file=sys.stderr)
            df, source = _fetch_yahoo(symbol, interval, days, use_cache), "yahoo"
    cache.save(_cache_key(source, symbol, interval, days), df)
    df.attrs["source"] = source
    return df
