"""Test the favorite-longshot bias on resolved Polymarket markets.

For each resolved market: take the YES price `ENTRY_DAYS` before resolution,
bucket it, and compute the return of buying YES at that price and holding to
resolution. If favorites (high prices) systematically return > 0 after the
bucket's fair value, the bias exists and is tradeable.

Run:  venv\\Scripts\\python.exe experiments\\polymarket_favorite_bias.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"

N_MARKETS = 500
ENTRY_DAYS = 14
BUCKETS = [(0.02, 0.10), (0.10, 0.30), (0.30, 0.50), (0.50, 0.70),
           (0.70, 0.90), (0.90, 0.98)]


def fetch_resolved_markets(limit: int) -> list[dict]:
    out, offset = [], 0
    while len(out) < limit:
        resp = requests.get(GAMMA_URL, params={
            "closed": "true", "order": "volumeNum", "ascending": "false",
            "limit": 100, "offset": offset,
        }, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        offset += 100
    return out[:limit]


def yes_history(token_id: str) -> pd.Series | None:
    try:
        resp = requests.get(CLOB_HISTORY_URL, params={
            "market": token_id, "interval": "max", "fidelity": 1440,
        }, timeout=20)
        resp.raise_for_status()
        history = resp.json().get("history", [])
    except requests.RequestException:
        return None
    if len(history) < ENTRY_DAYS + 2:
        return None
    s = pd.Series([h["p"] for h in history],
                  index=pd.to_datetime([h["t"] for h in history], unit="s"))
    return s


def main():
    markets = fetch_resolved_markets(N_MARKETS)
    print(f"fetched {len(markets)} resolved markets, pulling histories...")

    rows = []
    for i, m in enumerate(markets):
        try:
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue
        if not token_ids:
            continue
        hist = yes_history(token_ids[0])
        if hist is None:
            continue
        final = hist.iloc[-1]
        if 0.05 < final < 0.95:
            continue  # ambiguous/unresolved-looking history, skip
        outcome = 1.0 if final >= 0.95 else 0.0
        entry = hist.iloc[-1 - ENTRY_DAYS]
        if not (0.02 <= entry <= 0.98):
            continue  # already decided at entry, no trade
        rows.append({
            "question": m.get("question", "")[:60],
            "entry": entry,
            "outcome": outcome,
            "ret": outcome / entry - 1,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(markets)} processed, {len(rows)} usable")
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no usable samples")
        return
    print(f"\n{len(df)} tradeable samples (entry price {ENTRY_DAYS}d before resolution)\n")

    print(f"{'bucket':>12} {'n':>4} {'avg entry':>10} {'win rate':>9} "
          f"{'avg return':>11} {'edge':>8}")
    for lo, hi in BUCKETS:
        sub = df[(df.entry >= lo) & (df.entry < hi)]
        if len(sub) == 0:
            continue
        avg_entry = sub.entry.mean()
        win = sub.outcome.mean()
        avg_ret = sub.ret.mean()
        # edge = actual win rate minus what the price implied
        print(f"{lo:>5.0%}-{hi:<5.0%} {len(sub):>4d} {avg_entry:>10.2f} "
              f"{win:>9.1%} {avg_ret:>11.1%} {win - avg_entry:>+8.1%}")

    print("\npositive edge in high buckets = favorites underpriced (the classic bias)")
    print("negative edge in low buckets = longshots overpriced (don't buy lottery tickets)")

    out = Path(__file__).parent / "favorite_bias_samples.csv"
    df.to_csv(out, index=False)
    print(f"samples saved: {out}")


if __name__ == "__main__":
    main()
