from .stocks import fetch_stock
from .crypto import fetch_crypto
from .polymarket import fetch_polymarket, search_polymarket_markets


def fetch(market: str, symbol: str, **kwargs):
    """Unified entry point: fetch OHLCV (or price) data for any market.

    market: "stock" | "crypto" | "polymarket"
    Returns a DataFrame indexed by datetime with at least a "close" column.
    """
    market = market.lower()
    if market == "stock":
        return fetch_stock(symbol, **kwargs)
    if market == "crypto":
        return fetch_crypto(symbol, **kwargs)
    if market == "polymarket":
        return fetch_polymarket(symbol, **kwargs)
    raise ValueError(f"unknown market: {market}")
