"""Quarterly review of the forward ledgers (rules in forward/GOVERNANCE.md).

For each ledger: where its live return so far falls in the distribution of
same-length returns its frozen rules produced before it started (a block
bootstrap of the backtest), its drawdown against the backtest's worst, and
whether a kill rule fires. Across ledgers: the correlation of their daily
returns and an equal-weight combined book.

The reference is the frozen rules run over the REF_YEARS before each ledger
started. For portfolio_2026 the parameters were chosen on part of that
history, so its reference is optimistic: a low percentile there is a weaker
signal of failure than it looks, a high one stronger.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine import Backtester
from ..research.portfolio import leg_returns
from ..strategies import make_strategy
from . import ledger
from .runner import ROOT, ledger_dir, portfolio_equity, spec_paths
from .spec import load_spec

REF_YEARS = 8
BLOCK = 20
N_BOOT = 2000
KILL_PERCENTILE = 5.0
KILL_DRAWDOWN_MULTIPLE = 1.5


def reference_fetch(leg) -> pd.DataFrame:
    """Full history, unlike the ledger's own short signal window."""
    from ..data import fetch
    if leg.market == "crypto":
        return fetch("crypto", leg.symbol, days=6000, source=leg.source, use_cache=False)
    return fetch("stock", leg.symbol, period="max", use_cache=False)


def reference_returns(spec, anchors: dict, fetch=reference_fetch) -> pd.Series:
    """Daily returns of the frozen rules before the ledger began, legs weighted
    and rebalanced on common days as the ledger's portfolio is."""
    bt = Backtester(fee=spec.fee, slippage=spec.slippage, fill="next_open")
    equity = {}
    for leg in spec.legs:
        prices = fetch(leg).loc[:anchors[leg.symbol]]
        eq = bt.run(prices, make_strategy(leg.strategy, **leg.params)).equity
        equity[leg.symbol] = eq.loc[eq.index[-1] - pd.DateOffset(years=REF_YEARS):]
    rets = leg_returns(equity)
    return sum(leg.weight * rets[leg.symbol] for leg in spec.legs)


def percentile_of(live_total: float, reference: pd.Series, n_days: int,
                  seed: int = 0) -> float:
    """Share (0-100) of bootstrapped n_days-long reference runs that did no better."""
    r = reference.dropna().to_numpy(dtype=float)
    if n_days < 1 or len(r) < 2 * BLOCK:
        return float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n_days // BLOCK)
    totals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, len(r) - BLOCK + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + BLOCK] for s in starts])[:n_days]
        totals[i] = np.prod(1 + sample) - 1
    return float((totals <= live_total).mean() * 100)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def kill_reasons(history: list[float], live_dd: float, reference_dd: float) -> list[str]:
    """history: this ledger's recorded quarter-end percentiles, oldest first."""
    reasons = []
    if len(history) >= 2 and all(p < KILL_PERCENTILE for p in history[-2:]):
        reasons.append(f"below the {KILL_PERCENTILE:.0f}th percentile at two quarter-ends in a row")
    if reference_dd < 0 and live_dd < KILL_DRAWDOWN_MULTIPLE * reference_dd:
        reasons.append(f"drawdown {live_dd:.1%} is beyond {KILL_DRAWDOWN_MULTIPLE}x the "
                       f"backtest's worst ({reference_dd:.1%})")
    return reasons


def history_path(root: Path | None = None) -> Path:
    return (root or ROOT) / "forward" / "reviews" / "history.json"


def load_history(root: Path | None = None) -> dict:
    p = history_path(root)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def evaluate_ledger(path, history: dict, root=None, fetch=reference_fetch) -> dict | None:
    spec = load_spec(path)
    d = ledger_dir(spec, root)
    if ledger.read_lock(d) is None:
        return None
    anchors = {leg.symbol: ledger.read_leg(d, leg.symbol)["date"].iloc[0] for leg in spec.legs}
    live = portfolio_equity(spec, root)
    ref = reference_returns(spec, anchors, fetch)
    days = len(live) - 1
    live_total = float(live.iloc[-1] - 1) if days > 0 else 0.0
    live_dd, ref_dd = max_drawdown(live), max_drawdown((1 + ref).cumprod())
    past = [h["percentile"] for h in history.get(spec.name, [])]
    return {"name": spec.name, "days": days, "live_return": live_total,
            "percentile": percentile_of(live_total, ref, days),
            "live_dd": live_dd, "reference_dd": ref_dd,
            "kill": kill_reasons(past, live_dd, ref_dd), "returns": live.pct_change().dropna()}


def evaluate_all(root=None, fetch=reference_fetch, record: bool = False,
                 today: str | None = None) -> str:
    """Review every started ledger; with record=True (quarter-ends only) the
    percentiles join the history the two-quarter kill rule reads."""
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history = load_history(root)
    results = [r for p in spec_paths(root) if (r := evaluate_ledger(p, history, root, fetch))]
    if record:
        for r in results:
            history.setdefault(r["name"], []).append({"date": today, "percentile": r["percentile"]})
            r["kill"] = kill_reasons([h["percentile"] for h in history[r["name"]]],
                                     r["live_dd"], r["reference_dd"])
        history_path(root).parent.mkdir(parents=True, exist_ok=True)
        history_path(root).write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return review_text(results, today, record)


def review_text(results: list[dict], today: str, record: bool) -> str:
    lines = [f"# Forward review, {today}", "",
             "Recorded quarter-end review." if record else
             "Not a quarter-end: informational only, nothing recorded, nothing to act on.", "",
             "| Ledger | Days | Return | Percentile vs backtest | Drawdown | Backtest worst | Kill |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        kill = "; ".join(r["kill"]) if r["kill"] else "no"
        lines.append(f"| {r['name']} | {r['days']} | {r['live_return']:+.2%} | "
                     f"{r['percentile']:.0f} | {r['live_dd']:.1%} | {r['reference_dd']:.1%} | {kill} |")
    rets = pd.DataFrame({r["name"]: r["returns"] for r in results}).dropna()
    if len(rets) > 2 and rets.shape[1] > 1:
        corr = rets.corr()
        book = (1 + rets.mean(axis=1)).prod() - 1
        lines += ["", f"## Correlation of daily returns ({len(rets)} common days)", "",
                  "| | " + " | ".join(corr.columns) + " |", "|---|" + "---|" * len(corr.columns)]
        lines += [f"| {i} | " + " | ".join(f"{v:+.2f}" for v in row) + " |"
                  for i, row in corr.iterrows()]
        lines += ["", f"Equal-weight combined book over those days: {book:+.2%}."]
    return "\n".join(lines) + "\n"
