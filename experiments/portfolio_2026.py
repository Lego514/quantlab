"""The "2026 portfolio": 50/50 QQQ + BTC vol-targeted momentum,
built ONLY from walk-forward out-of-sample returns (no in-sample flattery).

Run:  venv\\Scripts\\python.exe experiments\\portfolio_2026.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.data import fetch
from quantlab.engine import Backtester, compute_metrics
from quantlab.research import leg_returns, walk_forward
from quantlab.strategies import VolMomentum

WEIGHTS = {"QQQ": 0.5, "BTC-USD": 0.5}


def main():
    bt = Backtester(fee=0.001, slippage=0.0005)

    equity, closes = {}, {}
    for symbol in WEIGHTS:
        if symbol == "QQQ":
            prices = fetch("stock", symbol, period="10y", interval="1d")
        else:
            prices = fetch("crypto", symbol, days=2000)
        closes[symbol] = prices["close"]
        wf = walk_forward(prices, VolMomentum, bt)
        equity[symbol] = wf["oos_equity"]
        m = wf["oos_metrics"]
        print(f"{symbol:8s} OOS: sharpe {m['sharpe']:.2f}  cagr {m['cagr']:.1%}  "
              f"maxDD {m['max_drawdown']:.1%}")

    # Overlap period, on the days both legs trade; BTC's weekend moves land on
    # Monday instead of being dropped.
    rets = leg_returns(equity)
    print(f"\noverlap: {rets.index[0].date()} -> {rets.index[-1].date()} "
          f"({len(rets)} days)")
    print(f"leg correlation: {rets.corr().iloc[0, 1]:.2f}")

    port_rets = sum(w * rets[s] for s, w in WEIGHTS.items())
    port_equity = (1 + port_rets).cumprod()
    # The portfolio rebalances only on days both legs trade: a stock calendar.
    m = compute_metrics(port_equity, periods_per_year=252)
    print(f"\n[50/50 portfolio, daily rebalance, out-of-sample only]")
    print(f"  total_return   {m['total_return']:>10.2%}")
    print(f"  cagr           {m['cagr']:>10.2%}")
    print(f"  sharpe         {m['sharpe']:>10.2f}")
    print(f"  sortino        {m['sortino']:>10.2f}")
    print(f"  max_drawdown   {m['max_drawdown']:>10.2%}")

    # The bar the portfolio has to clear: simply holding each asset over the
    # same window.
    print(f"\n[buy & hold, same window]")
    for symbol, close in closes.items():
        window = close[(close.index >= rets.index[0]) & (close.index <= rets.index[-1])]
        b = compute_metrics(window / window.iloc[0])
        print(f"  {symbol:8s} sharpe {b['sharpe']:.2f}  cagr {b['cagr']:.1%}  "
              f"maxDD {b['max_drawdown']:.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for s in WEIGHTS:
        ax.plot((1 + rets[s]).cumprod(), alpha=0.6, label=f"{s} leg")
    ax.plot(port_equity, linewidth=2, color="black",
            label=f"50/50 portfolio (sharpe {m['sharpe']:.2f})")
    ax.set_title("2026 strategy: vol-targeted momentum, OOS only")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "reports" / "portfolio_2026.png"
    fig.savefig(out, dpi=110)
    print(f"\nchart saved: {out}")


if __name__ == "__main__":
    main()
