"""Test the favorite-longshot bias on resolved Polymarket markets.

For each resolved market: find the last YES trade before `--horizon` days
ahead of the scheduled end, bucket that price, and compute the return of
buying YES there and holding to resolution. If favorites (high prices) win
more often than their price implies, the bias exists.

Version 2 (2026-10). The first version read the daily price-history series
and found 16 strong favorites out of 16 winning; the case study called it the
most promising mispricing in the project. A larger study in a sibling project
then found favorites winning LESS often than implied. This version uses that
study's safeguards so the two can be compared on equal terms:

- entry price = an actual trade (data-api), not a history point, and each
  sample records how stale that trade was (hours before the entry time)
- a fresh-only column: an edge that lives only in stale prints is an
  artifact, because nobody can buy at a stale price
- only clean binary markets that resolved on schedule, so "N days before the
  end" is a time a trader could have known in advance
- Wilson 95% intervals per bucket, and the signal-quality gate on favorites

Prices are trade prints, not asks: real entry would cost the spread on top.

Run:  venv\\Scripts\\python.exe experiments\\polymarket_favorite_bias.py --n 600 --horizon 14
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.research import gate, staleness_split

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
TRADES_URL = "https://data-api.polymarket.com/trades"

SINCE = "2026-01-01"   # recent resolutions only: one market regime
VOL_MIN = 5_000        # a dead market's price means nothing
VOL_MAX = 2_000_000    # huge markets trade too often to page back to the entry
MAX_PAGES = 12         # trade pages (500 each) to search per market
BUCKETS = [(0.02, 0.10), (0.10, 0.30), (0.30, 0.50), (0.50, 0.70),
           (0.70, 0.90), (0.90, 0.98)]
FAVORITES = (0.90, 0.98)


def _get(url: str, params: dict, tries: int = 4):
    for i in range(tries):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                time.sleep(2 ** i)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    return []


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def _clean_binary(m: dict) -> dict | None:
    """A binary market that resolved 0/1 on schedule, else None."""
    try:
        prices = [float(x) for x in json.loads(m["outcomePrices"])]
        outcomes = [o.lower() for o in json.loads(m["outcomes"])]
        vol = float(m.get("volumeNum") or 0)
        end = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
        closed = datetime.strptime((m.get("closedTime") or "")[:19],
                                   "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (KeyError, ValueError, TypeError):
        return None
    if outcomes != ["yes", "no"] or sorted(prices) != [0.0, 1.0]:
        return None
    if not VOL_MIN <= vol <= VOL_MAX:
        return None
    # Resolved within two days of the scheduled end. Early resolutions are
    # dropped, which undercounts by-date markets that resolved YES early;
    # that biases high buckets DOWN, so it cannot manufacture a favorite edge.
    if abs((closed - end).total_seconds()) > 2 * 86_400:
        return None
    return {"condition_id": m["conditionId"], "question": m.get("question", "")[:60],
            "yes_won": prices[0] == 1.0, "end_ts": int(end.timestamp()), "volume": vol}


def fetch_resolved_markets(n_target: int) -> list[dict]:
    """Walk back from today in 3-day windows (gamma rejects large offsets)."""
    out = []
    win_end = datetime.now(timezone.utc)
    since = datetime.fromisoformat(SINCE).replace(tzinfo=timezone.utc)
    while len(out) < n_target and win_end > since:
        win_start = win_end - timedelta(days=3)
        offset = 0
        while len(out) < n_target and offset < 2000:
            try:
                batch = _get(GAMMA_URL, {
                    "closed": "true", "limit": 100, "offset": offset,
                    "end_date_min": win_start.strftime("%Y-%m-%d"),
                    "end_date_max": win_end.strftime("%Y-%m-%d"),
                    "order": "endDate", "ascending": "false"})
            except requests.RequestException:
                break
            if not batch:
                break
            out.extend(m for m in map(_clean_binary, batch) if m)
            offset += 100
        win_end = win_start
    return out[:n_target]


def last_trade_before(condition_id: str, ts: int) -> tuple[float, float] | None:
    """(YES price, hours stale) of the last trade before `ts`, or None."""
    offset = 0
    for _ in range(MAX_PAGES):
        try:
            batch = _get(TRADES_URL, {"market": condition_id, "limit": 500, "offset": offset})
        except requests.RequestException:
            return None
        if not batch:
            return None
        for t in batch:  # newest first
            t_ts = t.get("timestamp") or 0
            if t_ts >= ts:
                continue
            try:
                p = float(t["price"])
                idx = int(t.get("outcomeIndex") or 0)
            except (KeyError, ValueError, TypeError):
                continue
            if 0 < p < 1:
                return (p if idx == 0 else 1 - p), (ts - t_ts) / 3600
        if len(batch) < 500:
            return None
        offset += 500
        time.sleep(0.1)
    return None


def bucket_row(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["outcome"] for r in rows)
    lo, hi = wilson(k, n)
    entry = sum(r["entry"] for r in rows) / n
    return {"n": n, "entry": entry, "win": k / n, "ci": (lo, hi),
            "edge": k / n - entry, "ret": sum(r["ret"] for r in rows) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="resolved markets to sample")
    ap.add_argument("--horizon", type=int, default=14,
                    help="days before the scheduled end to take the entry trade")
    args = ap.parse_args()

    markets = fetch_resolved_markets(args.n)
    print(f"{len(markets)} clean binary markets resolved on schedule since {SINCE}; "
          f"finding the last trade {args.horizon}d before each end...")

    rows = []
    for i, m in enumerate(markets):
        got = last_trade_before(m["condition_id"], m["end_ts"] - args.horizon * 86_400)
        if got is not None:
            entry, stale_h = got
            if 0.02 <= entry <= 0.98:
                outcome = 1.0 if m["yes_won"] else 0.0
                rows.append({"question": m["question"], "entry": round(entry, 4),
                             "stale_h": round(stale_h, 2), "outcome": outcome,
                             "ret": outcome / entry - 1, "volume": m["volume"]})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(markets)} processed, {len(rows)} usable", flush=True)

    if not rows:
        print("no usable samples")
        return
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} samples, entry = last trade {args.horizon}d before the "
          f"scheduled end (median staleness {df.stale_h.median():.1f}h)\n")
    print(f"{'bucket':>11} {'n':>4} {'entry':>6} {'win':>6} {'95% CI':>15} {'edge':>7} "
          f"{'avg ret':>8} | {'fresh n':>7} {'fresh edge':>10}")
    for lo, hi in BUCKETS:
        grp = [r for r in rows if lo <= r["entry"] < hi]
        if not grp:
            continue
        b = bucket_row(grp)
        fresh, _ = staleness_split(grp)
        f_txt = f"{len(fresh):>7} {bucket_row(fresh)['edge']:>+10.1%}" if fresh else \
            f"{0:>7} {'':>10}"
        print(f"{lo:>5.0%}-{hi:<5.0%} {b['n']:>4} {b['entry']:>6.2f} {b['win']:>6.1%} "
              f"[{b['ci'][0]:>5.1%},{b['ci'][1]:>6.1%}] {b['edge']:>+7.1%} "
              f"{b['ret']:>+8.1%} | {f_txt}")

    favs = [r["ret"] for r in rows if FAVORITES[0] <= r["entry"] < FAVORITES[1]]
    print(f"\n[buying favorites {FAVORITES[0]:.2f}-{FAVORITES[1]:.2f}, held to resolution]")
    print(gate(favs).report())
    print("\nedge = win rate minus average entry price. Positive in the top bucket would "
          "mean favorites are underpriced;\na fresh-only edge that disappears means the "
          "apparent edge was stale prints.")

    out = Path(__file__).parent / "favorite_bias_samples.csv"
    df.to_csv(out, index=False)
    print(f"samples saved: {out}")


if __name__ == "__main__":
    main()
