"""Cross-sectional crypto rotation: every week, hold the top-N coins by
trailing momentum (only those with positive momentum; otherwise cash).

This is the classic "relative strength" construction — you don't predict
any single coin, you just always ride whichever is currently strongest.

Honesty protocol: grid over (lookback, top_n) is shown for the FULL period,
but the headline number is chosen on data through 2023-12-31 and reported
on 2024+ ONLY (true out-of-sample).

Run:  venv\\Scripts\\python.exe experiments\\crypto_rotation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.data import fetch
from quantlab.engine import compute_metrics

UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
            "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "LTC-USD"]
FEE = 0.0015          # fee + slippage per unit turnover
DAYS = 2000
SPLIT = "2024-01-01"  # in-sample before, out-of-sample after


def load_closes() -> pd.DataFrame:
    closes = {}
    for sym in UNIVERSE:
        try:
            closes[sym] = fetch("crypto", sym, days=DAYS)["close"]
        except Exception as e:
            print(f"  ! {sym}: {e}")
    df = pd.DataFrame(closes)
    df.index = pd.to_datetime(df.index).normalize()
    return df[~df.index.duplicated()]


def rotation_returns(closes: pd.DataFrame, lookback: int, top_n: int,
                     rebalance_days: int = 7) -> pd.Series:
    rets = closes.pct_change()
    mom = closes.pct_change(lookback)

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    current = pd.Series(0.0, index=closes.columns)
    for i, ts in enumerate(closes.index):
        if i % rebalance_days == 0:
            m = mom.iloc[i].dropna()
            winners = m[m > 0].nlargest(top_n)
            current = pd.Series(0.0, index=closes.columns)
            if len(winners) > 0:
                current[winners.index] = 1.0 / top_n
        weights.iloc[i] = current

    held = weights.shift(1).fillna(0.0)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    port = (held * rets).sum(axis=1) - turnover * FEE
    return port.dropna()


def show(label: str, returns: pd.Series) -> dict:
    m = compute_metrics((1 + returns).cumprod())
    print(f"{label:24s} cagr {m['cagr']:>8.1%}  sharpe {m['sharpe']:>5.2f}  "
          f"maxDD {m['max_drawdown']:>7.1%}  calmar {m['calmar']:>5.2f}")
    return m


def main():
    closes = load_closes()
    print(f"universe loaded: {closes.shape[1]} coins, "
          f"{closes.index[0].date()} -> {closes.index[-1].date()}\n")

    print("=== full-period grid (in-sample, for sensitivity only) ===")
    grid_results = {}
    for lookback in (14, 30, 60, 90):
        for top_n in (1, 2, 3):
            r = rotation_returns(closes, lookback, top_n)
            m = show(f"lb={lookback} top{top_n}", r)
            grid_results[(lookback, top_n)] = m

    print("\n=== honest test: pick best config on <2024, report 2024+ only ===")
    is_closes = closes.loc[:SPLIT]
    best_cfg, best_sharpe = None, -9e9
    for lookback in (14, 30, 60, 90):
        for top_n in (1, 2, 3):
            m = compute_metrics(
                (1 + rotation_returns(is_closes, lookback, top_n)).cumprod())
            if m.get("sharpe", -9e9) > best_sharpe:
                best_sharpe, best_cfg = m["sharpe"], (lookback, top_n)
    print(f"config chosen in-sample: lookback={best_cfg[0]}, top_n={best_cfg[1]} "
          f"(IS sharpe {best_sharpe:.2f})")

    full = rotation_returns(closes, *best_cfg)
    oos = full.loc[SPLIT:]
    show("OOS 2024+ rotation", oos)
    bench = closes["BTC-USD"].pct_change().loc[SPLIT:].dropna()
    show("OOS 2024+ BTC hold", bench)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot((1 + full).cumprod(), label=f"rotation lb={best_cfg[0]} top{best_cfg[1]}")
    ax.plot((1 + closes["BTC-USD"].pct_change().reindex(full.index).fillna(0)).cumprod(),
            alpha=0.6, label="BTC hold")
    ax.axvline(pd.Timestamp(SPLIT), color="red", linestyle="--", alpha=0.6,
               label="IS/OOS split")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("crypto cross-sectional rotation (weekly, top-N momentum)")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "reports" / "crypto_rotation.png"
    fig.savefig(out, dpi=110)
    print(f"\nchart saved: {out}")


if __name__ == "__main__":
    main()
