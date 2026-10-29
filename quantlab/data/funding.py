"""Perpetual-swap funding rates.

A position held in a perpetual future pays (or receives) funding every few
hours. Spot backtests never see it, so a strategy that would really run on
perps looks better than it is: in a bull market longs pay the rate on every
settlement.

Source: OKX public API, no key needed. OKX only serves about the last three
months of history, so `funding_per_bar` reports how much of a backtest the
data actually covers; bars before the first settlement are charged nothing.
"""
import pandas as pd
import requests

from . import cache

OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate-history"


def _to_okx_swap(symbol: str) -> str:
    """BTC-USD / BTC/USDT / btcusdt -> BTC-USDT-SWAP"""
    s = symbol.upper().replace("/", "-")
    base = s.split("-")[0]
    for quote in ("USDT", "USD"):
        if base.endswith(quote) and base != quote:
            base = base[: -len(quote)]
            break
    return f"{base}-USDT-SWAP"


def fetch_okx_funding(symbol: str) -> pd.Series:
    """Every funding settlement OKX still serves, as a rate per settlement
    indexed by settlement time (UTC, naive)."""
    inst = _to_okx_swap(symbol)
    key = f"funding_okx_{inst}"
    cached = cache.load(key)
    if cached is not None:
        return cached["rate"]

    rows, after = [], None
    while True:
        params = {"instId": inst, "limit": 400}
        if after is not None:
            params["after"] = after
        resp = requests.get(OKX_FUNDING_URL, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json().get("data", [])
        if not batch:
            break
        rows.extend(batch)
        after = min(int(r["fundingTime"]) for r in batch)
    if not rows:
        raise RuntimeError(f"okx returned no funding history for {inst}")
    rates = pd.Series(
        [float(r["realizedRate"] or r["fundingRate"]) for r in rows],
        index=pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms"),
        name="rate",
    ).sort_index()
    rates = rates[~rates.index.duplicated()]
    df = rates.to_frame()
    df.attrs["source"] = "okx"
    cache.save(key, df)
    return rates


def funding_per_bar(rates: pd.Series, index: pd.DatetimeIndex) -> tuple[pd.Series, float]:
    """Sum the settlements that fall inside each bar [t, next t).

    Returns (rate per bar, fraction of bars on or after the first settlement).
    """
    per_bar = pd.Series(0.0, index=index)
    if len(rates) == 0 or len(index) == 0:
        return per_bar, 0.0
    pos = index.searchsorted(rates.index, side="right") - 1
    inside = pos >= 0
    sums = pd.Series(rates.values[inside]).groupby(pos[inside]).sum()
    per_bar.iloc[sums.index.to_numpy()] = sums.to_numpy()
    coverage = float((index >= rates.index[0]).mean())
    return per_bar, coverage
