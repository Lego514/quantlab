"""Stock (and generic Yahoo Finance) OHLCV data."""
import pandas as pd

from . import cache


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance may return MultiIndex columns even for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].dropna(subset=["close"])


def fetch_stock(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV for a stock ticker via Yahoo Finance, with local cache."""
    key = f"stock_{symbol}_{period}_{interval}"
    cached = cache.load(key)
    if cached is not None:
        return cached

    import yfinance as yf

    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"no data returned for stock symbol {symbol!r}")
    df = _normalize(df)
    cache.save(key, df)
    return df
