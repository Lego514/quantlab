"""quantlab CLI - multi-market strategy backtesting.

Examples:
  python main.py backtest --market crypto --symbol BTC-USD --strategy sma_cross
  python main.py backtest --market stock --symbol AAPL --strategy rsi_reversion --params window=14,lower=30
  python main.py backtest --market crypto --symbol BTC-USD --strategy momentum --venue coinbase_advanced
  python main.py backtest --market crypto --symbol ETH-USD --strategy sma_cross --perp --allow-short
  python main.py optimize --market crypto --symbol ETH-USD --strategy momentum
  python main.py walkforward --market stock --symbol QQQ --strategy sma_cross
  python main.py scan --market crypto --symbols BTC-USD,ETH-USD,SOL-USD --control 100
  python main.py pm-search --query "fed"
  python main.py backtest --market polymarket --symbol <token_id> --strategy momentum
"""
import argparse
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from quantlab.data import fetch, search_polymarket_markets
from quantlab.data.funding import fetch_okx_funding, funding_per_bar
from quantlab.engine import VENUES, Backtester, venue
from quantlab.engine.metrics import _periods_per_year
from quantlab.strategies import REGISTRY, make_strategy
from quantlab.research import (block_bootstrap_sharpe, gate, grid_search, shift_control,
                               trades_needed, walk_forward, years_needed)
from quantlab.report import plot_backtest, plot_walkforward

POLYMARKET_WARNING = (
    "  ! Polymarket history is a last-trade series: no bid-ask spread, and thin\n"
    "    markets print stale prices nobody could trade at. The flat fee and\n"
    "    slippage understate real costs (Polymarket US quotes had a median spread\n"
    "    near 13% in a 2026-07-30 survey). Treat results as directional at best.")


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
        print(POLYMARKET_WARNING, file=sys.stderr)
    prices = fetch(args.market, args.symbol, **kwargs)
    return prices.loc[:args.end] if args.end else prices


def get_funding(args, prices):
    """Per-bar OKX funding for --perp runs, else None."""
    if not getattr(args, "perp", False):
        return None
    if args.market != "crypto":
        raise SystemExit("--perp only applies to --market crypto")
    per_bar, coverage = funding_per_bar(fetch_okx_funding(args.symbol), prices.index)
    if coverage < 1:
        print(f"  ! OKX funding history covers {coverage:.0%} of {args.symbol}'s bars; "
              "earlier bars are charged no funding", file=sys.stderr)
    return per_bar


def make_backtester(args) -> Backtester:
    v = venue(args.venue)
    return Backtester(fee=v.fee if args.fee is None else args.fee,
                      slippage=v.slippage if args.slippage is None else args.slippage,
                      allow_short=args.allow_short,
                      max_leverage=getattr(args, "leverage", 1.0),
                      fill=args.fill, borrow_rate=args.borrow_rate)


def cmd_backtest(args):
    prices = get_data(args)
    strategy = make_strategy(args.strategy, **parse_params(args.params))
    result = make_backtester(args).run(prices, strategy, symbol=args.symbol,
                                       funding=get_funding(args, prices))
    print(result.summary())
    path = plot_backtest(result)
    print(f"\nchart saved: {path}")


def cmd_optimize(args):
    prices = get_data(args)
    cls = REGISTRY[args.strategy]
    ranked = grid_search(prices, cls, make_backtester(args),
                         funding=get_funding(args, prices))
    print(f"grid search: {args.strategy} on {args.symbol} "
          f"({len(ranked)} combos, in-sample!)\n")
    print(ranked.head(10).to_string())
    print("\nNOTE: in-sample rankings overfit. Run `walkforward` to validate.")


def cmd_walkforward(args):
    prices = get_data(args)
    cls = REGISTRY[args.strategy]
    wf = walk_forward(prices, cls, make_backtester(args), n_splits=args.splits,
                      funding=get_funding(args, prices))
    print(f"walk-forward: {args.strategy} on {args.symbol}  "
          f"(data: {wf['data_source'] or 'unknown'}, fill: {wf['fill']})\n")
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

    # How much should anyone trust that sharpe? Error bars, then the gate.
    equity = wf["oos_equity"]
    try:
        ci = block_bootstrap_sharpe(equity.pct_change().dropna(),
                                    _periods_per_year(equity.index))
        print(f"  sharpe 95% CI  [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]  (block bootstrap)")
    except ValueError:
        pass
    need_years = years_needed(m.get("sharpe", 0))
    if need_years is not None:
        print(f"  years of data before this sharpe is significant: "
              f"{need_years:.1f} (have {m.get('years', 0):.1f})")
    trades = wf["oos_trade_returns"]
    print("\n" + gate(trades).report())
    need = trades_needed(trades)
    if need is not None:
        print(f"  trades before the average trade clears zero: {need} (have {len(trades)})")

    path = plot_walkforward(wf, args.symbol, args.strategy)
    print(f"\nchart saved: {path}")


def cmd_scan(args):
    """Run every strategy on every symbol with default parameters, ranked by
    Sharpe. Every number here is IN-SAMPLE: it says where to look, not what
    works. --control N adds a random-timing control (research/controls.py)."""
    symbols = [s.strip() for s in args.symbols.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")] if args.strategies \
        else [n for n in REGISTRY if n != "buy_hold"]
    bt = make_backtester(args)

    rows = []
    for symbol in symbols:
        try:
            a = argparse.Namespace(**{**vars(args), "symbol": symbol})
            prices = get_data(a)
            funding = get_funding(a, prices)
        except Exception as e:
            print(f"  ! {symbol}: data error: {e}", file=sys.stderr)
            continue
        for name in strategies:
            result = bt.run(prices, make_strategy(name), symbol=symbol, funding=funding)
            row = {
                "symbol": symbol,
                "strategy": name,
                "sharpe": result.metrics.get("sharpe", 0),
                "cagr": result.metrics.get("cagr", 0),
                "max_dd": result.metrics.get("max_drawdown", 0),
                "trades": result.metrics.get("trades", 0),
                "bh_sharpe": result.benchmark_metrics.get("sharpe", 0),
            }
            if args.control:
                c = shift_control(prices, make_strategy(name), bt, n=args.control,
                                  funding=funding)
                row["ctrl_p"] = c["p_value"]
            rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("IN-SAMPLE scan, default parameters\n")
    print(df.to_string(index=False,
                       formatters={"sharpe": "{:.2f}".format,
                                   "cagr": "{:.1%}".format,
                                   "max_dd": "{:.1%}".format,
                                   "bh_sharpe": "{:.2f}".format,
                                   "ctrl_p": "{:.3f}".format}))
    if args.control:
        print(f"\nctrl_p: share of {args.control} randomly shifted copies of each strategy "
              "that scored at least as well. Near 1 = its timing adds nothing.")
    print("\nTop rows are the luckiest of many tries. optimize + walkforward them before "
          "believing any.")


def cmd_forward(args):
    from quantlab import forward
    if args.action == "run":
        appended = forward.run(args.spec)
        for symbol, n in appended.items():
            print(f"  {symbol}: {n} new row(s)")
        print()
        print(forward.summary_text(forward.load_spec(args.spec)))
    elif args.action == "reconcile":
        problems = forward.reconcile(args.spec)
        if problems:
            print("reconcile FAILED:")
            for p in problems:
                print(f"  x {p}")
            sys.exit(1)
        print("reconcile OK: the ledger follows from its own prices, and the backtester "
              "agrees with it on fresh data")
    else:
        print(forward.summary_text(forward.load_spec(args.spec)))


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
        sp.add_argument("--end", help="last date of data to use, e.g. 2026-09-30")
        sp.add_argument("--venue", choices=list(VENUES), default="default",
                        help="fee + slippage preset; --fee/--slippage override it")
        sp.add_argument("--fee", type=float, default=None)
        sp.add_argument("--slippage", type=float, default=None)
        sp.add_argument("--fill", choices=["next_open", "close"], default="next_open",
                        help="trade at the next bar's open (default) or the signal bar's close")
        sp.add_argument("--leverage", type=float, default=1.0,
                        help="max position size (2 = up to 2x leverage)")
        sp.add_argument("--allow-short", action="store_true")
        sp.add_argument("--borrow-rate", type=float, default=0.0,
                        help="annual borrow rate charged on short exposure")
        sp.add_argument("--perp", action="store_true",
                        help="crypto: charge OKX perpetual funding (about 3 months of history)")

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

    sp = sub.add_parser("scan", help="all strategies x many symbols, ranked (in-sample)")
    common(sp, need_symbol=False)
    sp.add_argument("--symbols", required=True, help="comma separated")
    sp.add_argument("--strategies", help="comma separated, default: all")
    sp.add_argument("--control", type=int, default=0,
                    help="random-timing control runs per cell (0 = off)")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("forward", help="frozen-rule forward ledger (see forward/)")
    sp.add_argument("action", choices=["run", "reconcile", "status"])
    sp.add_argument("--spec", default="forward/specs/portfolio_2026.json")
    sp.set_defaults(func=cmd_forward)

    sp = sub.add_parser("pm-search", help="search Polymarket markets")
    sp.add_argument("--query", default="")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_pm_search)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
