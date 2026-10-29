"""The "2026 portfolio": 50/50 QQQ + BTC vol-targeted momentum,
built ONLY from walk-forward out-of-sample returns (no in-sample flattery).

Run:  venv\\Scripts\\python.exe experiments\\portfolio_2026.py [--end 2026-09-30] [--fill close]

--end cuts the data at a date, so a restatement can hold the window fixed
while it changes the method. --fill close reproduces the engine's pre-2026-10
fill (at the signal bar's own close) for comparison.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.data import fetch
from quantlab.engine import Backtester, compute_metrics
from quantlab.engine.backtester import FILLS
from quantlab.research import block_bootstrap_sharpe, gate, leg_returns, walk_forward, years_needed
from quantlab.strategies import VolMomentum

WEIGHTS = {"QQQ": 0.5, "BTC-USD": 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", help="last date of data to use, e.g. 2026-09-30")
    ap.add_argument("--fill", choices=FILLS, default="next_open")
    args = ap.parse_args()

    bt = Backtester(fee=0.001, slippage=0.0005, fill=args.fill)

    equity, closes, trades = {}, {}, []
    for symbol in WEIGHTS:
        if symbol == "QQQ":
            prices = fetch("stock", symbol, period="10y", interval="1d")
        else:
            prices = fetch("crypto", symbol, days=2000)
        if args.end:
            prices = prices.loc[:args.end]
        closes[symbol] = prices["close"]
        wf = walk_forward(prices, VolMomentum, bt)
        equity[symbol] = wf["oos_equity"]
        trades.extend(wf["oos_trade_returns"])
        m = wf["oos_metrics"]
        print(f"{symbol:8s} OOS: sharpe {m['sharpe']:.2f}  cagr {m['cagr']:.1%}  "
              f"maxDD {m['max_drawdown']:.1%}  ({len(wf['oos_trade_returns'])} trades, "
              f"data {wf['data_source']}, fill {wf['fill']})")

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

    # The error bars a single Sharpe hides.
    ci = block_bootstrap_sharpe(port_rets, periods_per_year=252)
    print(f"  sharpe 95% CI  [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]"
          f"  (block bootstrap, 20-day blocks)")
    need = years_needed(m["sharpe"])
    if need is not None:
        print(f"  years before this sharpe is significant: {need:.1f} "
              f"(have {m['years']:.1f})")

    # The bar the portfolio has to clear: simply holding each asset over the
    # same window.
    print(f"\n[buy & hold, same window]")
    for symbol, close in closes.items():
        window = close[(close.index >= rets.index[0]) & (close.index <= rets.index[-1])]
        b = compute_metrics(window / window.iloc[0])
        print(f"  {symbol:8s} sharpe {b['sharpe']:.2f}  cagr {b['cagr']:.1%}  "
              f"maxDD {b['max_drawdown']:.1%}")

    print(f"\n[signal-quality gate, both legs' out-of-sample trades]")
    print(gate(trades).report())

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
