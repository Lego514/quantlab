"""Polymarket prediction-market data.

Uses the public Gamma API to search markets, and the CLOB API for
historical prices of an outcome token. Prices are probabilities in [0, 1].
"""
import json

import pandas as pd
import requests

from . import cache

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"


def search_polymarket_markets(query: str = "", limit: int = 20,
                              active: bool = True) -> pd.DataFrame:
    """List/search Polymarket markets. Returns question, token ids, volume."""
    params = {
        "limit": limit,
        "order": "volumeNum",
        "ascending": "false",
        "closed": "false" if active else "true",
    }
    resp = requests.get(GAMMA_URL, params=params, timeout=15)
    resp.raise_for_status()
    markets = resp.json()

    rows = []
    q = query.lower()
    for m in markets:
        question = m.get("question", "")
        if q and q not in question.lower():
            continue
        try:
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            outcomes = json.loads(m.get("outcomes", "[]"))
        except (json.JSONDecodeError, TypeError):
            token_ids, outcomes = [], []
        rows.append({
            "question": question,
            "outcomes": ", ".join(outcomes),
            "yes_token_id": token_ids[0] if token_ids else None,
            "volume": float(m.get("volumeNum") or 0),
            "end_date": m.get("endDate", ""),
        })
    return pd.DataFrame(rows)


def fetch_polymarket(token_id: str, fidelity: int = 60, **_) -> pd.DataFrame:
    """Fetch full price history for one outcome token.

    token_id: CLOB token id (get it from search_polymarket_markets).
    fidelity: candle size in minutes (60 = hourly, 1440 = daily).
    Returns DataFrame with a "close" column of probabilities in [0, 1].
    """
    key = f"polymarket_{token_id}_{fidelity}"
    cached = cache.load(key)
    if cached is not None:
        return cached

    resp = requests.get(CLOB_HISTORY_URL, params={
        "market": token_id,
        "interval": "max",
        "fidelity": fidelity,
    }, timeout=20)
    resp.raise_for_status()
    history = resp.json().get("history", [])
    if not history:
        raise RuntimeError(f"no price history for polymarket token {token_id}")

    df = pd.DataFrame(history)
    df.index = pd.to_datetime(df["t"], unit="s")
    df = df.rename(columns={"p": "close"})[["close"]].astype(float)
    cache.save(key, df)
    return df
