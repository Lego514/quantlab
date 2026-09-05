"""How far can leverage push the validated vol-momentum portfolio?

Takes the walk-forward OUT-OF-SAMPLE daily returns of the 50/50 QQQ+BTC
vol_momentum portfolio and mechanically levers them 1x..4x, charging
realistic financing on the borrowed portion (margin rate / perp funding).

This answers: "what CAGR is reachable, and what drawdown does it cost?"

Run:  venv\\Scripts\\python.exe experiments\\leverage_sweep.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.data import fetch
from quantlab.engine import Backtester, compute_metrics
from quantlab.research import walk_forward
from quantlab.strategies import VolMomentum

FUNDING_ANNUAL = 0.08  # financing cost on borrowed portion (margin/funding)
LEVERS = [1.0, 1.5, 2.0, 3.0, 4.0]


def main():
    bt = Backtester(fee=0.001, slippage=0.0005)
    legs = {}
    for symbol, kind, kwargs in [("QQQ", "stock", {"period": "10y"}),
                                 ("BTC-USD", "crypto", {"days": 2000})]:
        prices = fetch(kind, symbol, **kwargs)
        legs[symbol] = walk_forward(prices, VolMomentum, bt)["oos_equity"].pct_change()

    rets = pd.DataFrame(legs).dropna()
    base = 0.5 * rets["QQQ"] + 0.5 * rets["BTC-USD"]
    daily_funding = FUNDING_ANNUAL / 365

    print("levered 50/50 QQQ+BTC vol-momentum (OOS returns, funding "
          f"{FUNDING_ANNUAL:.0%}/yr on borrowed portion)\n")
    print(f"{'lever':>6} {'cagr':>8} {'sharpe':>7} {'maxDD':>8} {'calmar':>7} {'ruin?':>6}")
    for L in LEVERS:
        lr = L * base - max(L - 1, 0) * daily_funding
        equity = (1 + lr).cumprod()
        m = compute_metrics(equity)
        ruined = "YES" if (equity <= 0.05).any() or (1 + lr <= 0).any() else "no"
        print(f"{L:>5.1f}x {m['cagr']:>8.1%} {m['sharpe']:>7.2f} "
              f"{m['max_drawdown']:>8.1%} {m['calmar']:>7.2f} {ruined:>6}")

    print("\nnote: leverage multiplies CAGR sub-linearly (vol drag) but multiplies")
    print("drawdown roughly linearly. calmar (cagr/maxDD) is the honest score.")


if __name__ == "__main__":
    main()
