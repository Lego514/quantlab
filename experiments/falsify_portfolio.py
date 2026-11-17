"""P1: try to falsify the 2026 case study's claim of a timing edge.

The case study (STRATEGY_2026.md) beats buy & hold on risk-adjusted terms.
That alone does not show the signal times anything: a strategy that holds
0.4x on average would also halve buy & hold's drawdown. This experiment asks
whether the rules beat holding the SAME average exposure all the time.

Everything is out of sample, reconstructed from the walk-forward's own
per-fold parameters:
  - benchmarks: 50/50 buy & hold, and static exposure (each leg held at the
    strategy's own average position, every day)
  - ablation: momentum signal only / volatility target only / both
  - random-timing control per leg (shift_control, n=500)
  - stress: costs x3, fills one bar later, lookback 30-250 on the same window

Pass (fixed 2026-11-16 in Quantlab_Research_Plan.md, before this was run):
  portfolio Sharpe(strategy) - Sharpe(static) > 0 with paired block-bootstrap
  p < 0.10, AND each leg's random-timing control p < 0.10.

Run:  venv\\Scripts\\python.exe experiments\\falsify_portfolio.py --end 2026-09-30
Writes reports/falsify_p1.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.data import fetch
from quantlab.engine import Backtester, compute_metrics
from quantlab.engine.metrics import _periods_per_year
from quantlab.research import (leg_returns, paired_block_bootstrap_sharpe_diff, shift_control,
                               walk_forward)
from quantlab.research.controls import FixedPositions
from quantlab.strategies import VolMomentum

LEGS = {"QQQ": ("stock", {"period": "10y"}), "BTC-USD": ("crypto", {"days": 2000})}
WEIGHT = 0.5
PASS_P = 0.10
CONTROL_N = 500
LOOKBACKS = [30, 60, 90, 120, 150, 200, 250]
STRESS_TARGET_VOL = 0.15
VARIANTS = ("both", "signal", "voltarget", "static", "buyhold")


def realized_vol(close: pd.Series, window: int = 30) -> pd.Series:
    return close.pct_change().rolling(window).std() * np.sqrt(_periods_per_year(close.index))


def rule_positions(close: pd.Series, lookback: int, target_vol: float, variant: str) -> pd.Series:
    """The case study's rule with parts switched off. 'both' is VolMomentum."""
    trend = close.pct_change(lookback) > 0
    size = (target_vol / realized_vol(close)).clip(upper=1.0)
    rule = {"both": size.where(trend, 0.0), "signal": trend.astype(float), "voltarget": size}
    return rule[variant].fillna(0.0)


def oos_positions(prices: pd.DataFrame, folds: list, variant: str) -> pd.Series:
    """Positions over every test window, each with its own fold's parameters."""
    pieces = []
    for f in folds:
        start, end = (pd.Timestamp(s) for s in f["test_span"].split(" -> "))
        pos = rule_positions(prices["close"], f["params"]["lookback"],
                             f["params"]["target_vol"], variant)
        pieces.append(pos.loc[start:end])
    return pd.concat(pieces)


def equity(prices: pd.DataFrame, positions: pd.Series, bt: Backtester) -> pd.Series:
    window = prices.loc[positions.index[0]:positions.index[-1]]
    return bt.run(window, FixedPositions(positions)).equity


def leg_study(prices: pd.DataFrame, bt: Backtester) -> dict:
    """Walk-forward, then every variant's OOS positions and equity."""
    wf = walk_forward(prices, VolMomentum, bt)
    for f in wf["folds"]:  # 'both' must be the strategy itself, not a lookalike
        assert np.allclose(rule_positions(prices["close"], f["params"]["lookback"],
                                          f["params"]["target_vol"], "both"),
                           VolMomentum(**f["params"]).generate_positions(prices))
    pos = {v: oos_positions(prices, wf["folds"], v) for v in ("both", "signal", "voltarget")}
    both = pos["both"]
    pos["static"] = pd.Series(both.mean(), index=both.index)
    pos["buyhold"] = pd.Series(1.0, index=both.index)
    return {"wf": wf, "pos": pos, "eq": {v: equity(prices, p, bt) for v, p in pos.items()}}


def portfolio_returns(legs: dict, variant: str) -> pd.Series:
    rets = leg_returns({s: legs[s]["eq"][variant] for s in legs})
    return sum(WEIGHT * rets[s] for s in legs)


def sharpe(rets: pd.Series, ppy: float) -> float:
    return compute_metrics((1 + rets).cumprod(), periods_per_year=ppy)["sharpe"]


def strategy_vs_static(legs: dict) -> dict:
    return paired_block_bootstrap_sharpe_diff(portfolio_returns(legs, "both"),
                                              portfolio_returns(legs, "static"), 252)


def delayed(legs: dict, prices: dict, bt: Backtester) -> dict:
    """Same legs with every position filled one bar later."""
    out = {}
    for s, leg in legs.items():
        late = leg["pos"]["both"].shift(1).fillna(0.0)
        out[s] = {"eq": {"both": equity(prices[s], late, bt),
                         "static": leg["eq"]["static"]}}
    return out


def lookback_grid(prices: pd.DataFrame, window_index: pd.DatetimeIndex, bt: Backtester) -> dict:
    """Sharpe on the OOS window for fixed lookbacks: a plateau or a spike?"""
    out = {}
    for lb in LOOKBACKS:
        pos = rule_positions(prices["close"], lb, STRESS_TARGET_VOL, "both").loc[window_index]
        out[lb] = compute_metrics(equity(prices, pos, bt))["sharpe"]
    return out


def report(legs, controls, main, cost3, late, grids, verdict) -> str:
    lines = ["# P1 falsification: does the case study time anything?", "",
             "Generated by `experiments/falsify_portfolio.py --end 2026-09-30`. "
             "Thresholds fixed in `Quantlab_Research_Plan.md` before the run.", "",
             "## Each leg, out of sample", "",
             "| Leg | Strategy | Signal only | Vol target only | Static (same avg exposure) "
             "| Buy & hold | Avg exposure | Control p |", "|---|---|---|---|---|---|---|---|"]
    for s, leg in legs.items():
        sh = {v: compute_metrics(leg["eq"][v])["sharpe"] for v in VARIANTS}
        lines.append(f"| {s} | {sh['both']:.2f} | {sh['signal']:.2f} | {sh['voltarget']:.2f} | "
                     f"{sh['static']:.2f} | {sh['buyhold']:.2f} | "
                     f"{leg['pos']['both'].mean():.2f} | {controls[s]['p_value']:.3f} |")
    lines += ["", "## Portfolio (50/50, daily rebalance)", "",
              "| Variant | Sharpe |", "|---|---|"]
    for v in VARIANTS:
        lines.append(f"| {v} | {sharpe(portfolio_returns(legs, v), 252):.2f} |")
    lines += ["", "## Strategy minus static, paired block bootstrap", "",
              "| Case | Sharpe diff | 95% interval | p (one-sided) |", "|---|---|---|---|"]
    for name, r in (("as tested", main), ("costs x3", cost3), ("fills one bar later", late)):
        lines.append(f"| {name} | {r['diff']:+.2f} | [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] "
                     f"| {r['p_value']:.3f} |")
    lines += ["", f"## Lookback sweep (target vol {STRESS_TARGET_VOL:.0%}, same OOS window)", "",
              "| Lookback | " + " | ".join(grids) + " |", "|---|" + "---|" * len(grids)]
    for lb in LOOKBACKS:
        lines.append(f"| {lb} | " + " | ".join(f"{grids[s][lb]:.2f}" for s in grids) + " |")
    lines += ["", f"## Verdict: {'PASS' if verdict['passed'] else 'FAIL'}", ""]
    lines += [f"- {c}" for c in verdict["checks"]]
    lines += ["", "## Notes", "",
              "- Positions are rebuilt from each fold's parameters over contiguous test "
              "windows, so a parameter change at a fold boundary is traded and charged. "
              "The walk-forward itself drops each fold's first bar and that switch, which "
              "is why its leg Sharpes differ slightly (printed at the top of the run).",
              "- The static benchmark knows the strategy's average exposure in advance. That "
              "favours the benchmark a little; it is the price of asking only about timing."]
    return "\n".join(lines) + "\n"


def judge(main: dict, controls: dict) -> dict:
    checks = [f"portfolio Sharpe diff {main['diff']:+.2f} > 0 and p {main['p_value']:.3f} "
              f"< {PASS_P}: {'yes' if main['diff'] > 0 and main['p_value'] < PASS_P else 'NO'}"]
    checks += [f"{s} random-timing control p {c['p_value']:.3f} < {PASS_P}: "
               f"{'yes' if c['p_value'] < PASS_P else 'NO'}" for s, c in controls.items()]
    return {"passed": all(c.endswith("yes") for c in checks), "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-09-30")
    args = ap.parse_args()

    bt, bt3 = Backtester(fee=0.001, slippage=0.0005), Backtester(fee=0.003, slippage=0.0015)
    prices = {s: fetch(k, s, **kw).loc[:args.end] for s, (k, kw) in LEGS.items()}
    legs = {s: leg_study(p, bt) for s, p in prices.items()}
    for s, leg in legs.items():  # the reconstruction must reproduce the walk-forward
        print(f"{s}: walk-forward OOS sharpe {leg['wf']['oos_metrics']['sharpe']:.3f}, "
              f"reconstructed {compute_metrics(leg['eq']['both'])['sharpe']:.3f}")

    controls = {s: shift_control(prices[s].loc[leg["pos"]["both"].index],
                                 FixedPositions(leg["pos"]["both"]), bt, n=CONTROL_N)
                for s, leg in legs.items()}
    main_result = strategy_vs_static(legs)
    cost3 = strategy_vs_static({s: leg_study(p, bt3) for s, p in prices.items()})
    late = strategy_vs_static(delayed(legs, prices, bt))
    grids = {s: lookback_grid(prices[s], leg["pos"]["both"].index, bt) for s, leg in legs.items()}

    verdict = judge(main_result, controls)
    text = report(legs, controls, main_result, cost3, late, grids, verdict)
    out = Path(__file__).resolve().parents[1] / "reports" / "falsify_p1.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
