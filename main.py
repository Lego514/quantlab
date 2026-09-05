"""quantlab CLI - multi-market strategy backtesting.

Examples:
  python main.py backtest --market crypto --symbol BTC-USD --strategy sma_cross
  python main.py backtest --market stock --symbol AAPL --strategy rsi_reversion --params window=14,lower=30
  python main.py optimize --market crypto --symbol ETH-USD --strategy momentum
  python main.py walkforward --market stock --symbol QQQ --strategy sma_cross
  python main.py scan --market crypto --symbols BTC-USD,ETH-USD,SOL-USD
  python main.py pm-search --query "fed"
  python main.py backtest --market polymarket --symbol <token_id> --strategy momentum
"""
import argparse
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from quantlab.data import fetch, search_polymarket_markets
from quantlab.engine import Backtester
from quantlab.strategies import REGISTRY, make_strategy
from quantlab.research import grid_search, walk_forward
from quantlab.report import plot_backtest, plot_walkforward


def parse_params(s: str | None) -> dict:
    if not s:
        return {}
    out = {}
    for pair in s.split(","):
        k, v = pair.split("=")
        try:
            out[k.strip()] = int(v)
        except ValueError:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def get_data(args):
    kwargs = {}
    if args.market == "stock":
        kwargs = {"period": args.period, "interval": args.interval}
    elif args.market == "crypto":
        kwargs = {"interval": args.interval, "days": args.days}
    elif args.market == "polymarket":
        kwargs = {"fidelity": 1440 if args.interval == "1d" else 60}
    return fetch(args.market, args.symbol, **kwargs)


def make_backtester(args) -> Backtester:
    return Backtester(fee=args.fee, slippage=args.slippage,
                      max_leverage=getattr(args, "leverage", 1.0))


def cmd_backtest(args):
    prices = get_data(args)
    strategy = make_strategy(args.strategy, **parse_params(args.params))
    result = make_backtester(args).run(prices, strategy, symbol=args.symbol)
    print(result.summary())
    path = plot_backtest(result)
    print(f"\nchart saved: {path}")


def cmd_optimize(args):
    prices = get_data(args)
    cls = REGISTRY[args.strategy]
    ranked = grid_search(prices, cls, make_backtester(args))
    print(f"grid search: {args.strategy} on {args.symbol} "
          f"({len(ranked)} combos, in-sample!)\n")
    print(ranked.head(10).to_string())
    print("\nNOTE: in-sample rankings overfit. Run `walkforward` to validate.")


def cmd_walkforward(args):
    prices = get_data(args)
    cls = REGISTRY[args.strategy]
    wf = walk_forward(prices, cls, make_backtester(args), n_splits=args.splits)
    print(f"walk-forward: {args.strategy} on {args.symbol}\n")
    for f in wf["folds"]:
        oos = f["oos_sharpe"]
        oos_txt = f"{oos:.2f}" if oos is not None else "n/a"
        print(f"  fold {f['fold']}: test {f['test_span']}  "
              f"params={f['params']}  IS sharpe={f['is_sharpe']:.2f}  "
              f"OOS sharpe={oos_txt}")
    m = wf["oos_metrics"]
    print(f"\n[stitched out-of-sample performance]")
    print(f"  total_return   {m.get('total_return', 0):>10.2%}")
    print(f"  cagr           {m.get('cagr', 0):>10.2%}")
    print(f"  sharpe         {m.get('sharpe', 0):>10.2f}")
    print(f"  max_drawdown   {m.get('max_drawdown', 0):>10.2%}")
    path = plot_walkforward(wf, args.symbol, args.strategy)
    print(f"\nchart saved: {path}")


def cmd_scan(args):
    """Run every strategy on every symbol; rank by OOS-ish robustness (sharpe)."""
    symbols = [s.strip() for s in args.symbols.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")] if args.strategies \
        else [n for n in REGISTRY if n != "buy_hold"]
    bt = make_backtester(args)

    rows = []
    for symbol in symbols:
        try:
            a = argparse.Namespace(**{**vars(args), "symbol": symbol})
            prices = get_data(a)
        except Exception as e:
            print(f"  ! {symbol}: data error: {e}", file=sys.stderr)
            continue
        for name in strategies:
            result = bt.run(prices, make_strategy(name), symbol=symbol)
            rows.append({
                "symbol": symbol,
                "strategy": name,
                "sharpe": result.metrics.get("sharpe", 0),
                "cagr": result.metrics.get("cagr", 0),
                "max_dd": result.metrics.get("max_drawdown", 0),
                "trades": result.metrics.get("trades", 0),
                "bh_sharpe": result.benchmark_metrics.get("sharpe", 0),
            })
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(df.to_string(index=False,
                       formatters={"sharpe": "{:.2f}".format,
                                   "cagr": "{:.1%}".format,
                                   "max_dd": "{:.1%}".format,
                                   "bh_sharpe": "{:.2f}".format}))
    print("\ndefault params only — optimize + walkforward the promising cells.")


def cmd_pm_search(args):
    df = search_polymarket_markets(args.query, limit=args.limit)
    if df.empty:
        print("no matching markets")
        return
    with __import__("pandas").option_context("display.max_colwidth", 60):
        print(df.to_string(index=False))
    print("\nuse yes_token_id as --symbol with --market polymarket")


def build_parser():
    p = argparse.ArgumentParser(description="quantlab: multi-market backtesting")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, need_symbol=True):
        sp.add_argument("--market", choices=["stock", "crypto", "polymarket"],
                        default="stock")
        if need_symbol:
            sp.add_argument("--symbol", required=True)
        sp.add_argument("--period", default="5y", help="stock: yfinance period")
        sp.add_argument("--interval", default="1d")
        sp.add_argument("--days", type=int, default=1460, help="crypto lookback days")
        sp.add_argument("--fee", type=float, default=0.001)
        sp.add_argument("--slippage", type=float, default=0.0005)
        sp.add_argument("--leverage", type=float, default=1.0,
                        help="max position size (2 = up to 2x leverage)")

    sp = sub.add_parser("backtest", help="run one strategy on one symbol")
    common(sp)
    sp.add_argument("--strategy", choices=list(REGISTRY), default="sma_cross")
    sp.add_argument("--params", help="e.g. fast=20,slow=100")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("optimize", help="grid-search strategy parameters (in-sample)")
    common(sp)
    sp.add_argument("--strategy", choices=list(REGISTRY), default="sma_cross")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("walkforward", help="out-of-sample validation")
    common(sp)
    sp.add_argument("--strategy", choices=list(REGISTRY), default="sma_cross")
    sp.add_argument("--splits", type=int, default=4)
    sp.set_defaults(func=cmd_walkforward)

    sp = sub.add_parser("scan", help="all strategies x many symbols, ranked")
    common(sp, need_symbol=False)
    sp.add_argument("--symbols", required=True, help="comma separated")
    sp.add_argument("--strategies", help="comma separated, default: all")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("pm-search", help="search Polymarket markets")
    sp.add_argument("--query", default="")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_pm_search)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
