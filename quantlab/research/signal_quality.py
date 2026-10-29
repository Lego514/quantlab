"""Signal-quality gate: the checks an idea has to pass before it earns a
paper-trading ledger.

Ported from a sibling research project (Polymarket and funding-rate
scanners) that rejected six ideas in a row, all the same way: each was
selected on extreme values. A spike is a spike because its price was stale,
because it did not last, or because nobody could trade it. Each time it took
weeks of data collection to find out. These checks move the verdict to day
one:

  1. net of costs: what the gross number leaves after a round trip
  2. trimmed mean and median: a positive mean with a non-positive trimmed
     mean or median is carried by a few outliers (a lottery-ticket shape)
  3. sample size and t-statistic
  4. versus a control: entries with the same exposure but random timing
  5. decay: realized return / what the signal claimed when it fired
  6. concentration: share of total P&L earned by the best 5% of trades

`staleness_split` is the companion for any study built on last-trade prices:
compute the statistic separately on fresh and stale observations. An edge
that lives only in the stale group is an artifact, because a stale print is
a price nobody can trade at.

The thresholds are fixed in THRESHOLDS before looking at any result.
Loosening one means passing `thresholds=` explicitly, in the open.
"""
import math
import statistics
from dataclasses import dataclass

THRESHOLDS = {
    "min_n": 30,                 # below this, no verdict either way
    "min_trimmed_mean": 0.0,     # trimmed mean must be positive
    "min_median": 0.0,           # median must be positive (no lottery tickets)
    "min_t": 2.0,
    "min_edge_vs_control": 0.0,  # must beat the control
    "min_decay": 0.5,            # realized / signal >= 50%
    "max_top5_share": 0.5,       # best 5% of trades may earn at most half the P&L
}

STALE_HOURS = 6  # "fresh" = traded within 6 hours of the observation


def trade_stats(returns: list[float], trim: float = 0.05) -> dict | None:
    """Mean, median, trimmed mean, win rate and t-statistic of per-trade returns."""
    if not returns:
        return None
    s = sorted(returns)
    k = max(int(len(s) * trim), 1)
    trimmed = s[k:-k] if len(s) > 2 * k else s
    sd = statistics.pstdev(returns)
    mean = statistics.mean(returns)
    if len(returns) < 2:
        t = 0.0
    elif sd > 0:
        t = mean / (sd / math.sqrt(len(returns)))
    else:  # identical returns: no noise, so the sign is certain
        t = math.copysign(math.inf, mean) if mean else 0.0
    return {
        "n": len(returns),
        "mean": mean,
        "median": statistics.median(returns),
        "trimmed_mean": statistics.mean(trimmed),
        "win": sum(1 for r in returns if r > 0) / len(returns),
        "t": t,
    }


def top_share(returns: list[float], frac: float = 0.05) -> float:
    """Share of total P&L earned by the best `frac` of trades.

    Above 1 means the rest lost money in aggregate. 1.0 when the total is not
    positive, since then nothing is being earned to share out."""
    total = sum(returns)
    if not returns or total <= 0:
        return 1.0
    k = max(1, int(len(returns) * frac))
    return sum(sorted(returns)[-k:]) / total


@dataclass
class Verdict:
    passed: bool
    reasons: list
    stats: dict

    def report(self) -> str:
        s = self.stats
        lines = [f"signal-quality gate: {'PASS' if self.passed else 'FAIL'}"]
        if s.get("n"):
            lines.append(
                f"  n={s['n']}  mean {s['mean']:+.2%}  trimmed {s['trimmed_mean']:+.2%}  "
                f"median {s['median']:+.2%}  win {s['win']:.0%}  t={s['t']:.2f}")
        if s.get("control_mean") is not None:
            lines.append(f"  control mean {s['control_mean']:+.2%} (n={s['control_n']})"
                         f" -> edge {s['edge_vs_control']:+.2%}")
        if s.get("decay_median") is not None:
            lines.append(f"  median decay {s['decay_median']:.0%} (realized / signal)")
        if s.get("n"):
            lines.append(f"  best 5% of trades earn {s['top5_share']:.0%} of total P&L")
        lines.extend(f"  x {r}" for r in self.reasons)
        return "\n".join(lines)


def gate(returns: list[float], control: list[float] | None = None,
         signal: list[float | None] | None = None, cost_per_round: float = 0.0,
         thresholds: dict | None = None) -> Verdict:
    """Run every check and return a Verdict.

    returns:  per-trade returns; gross if `cost_per_round` is given, else net
    control:  per-trade returns of the control, same units
    signal:   what the signal claimed for each trade (aligned with `returns`,
              None where unknown), for the decay check
    """
    th = {**THRESHOLDS, **(thresholds or {})}
    net = [r - cost_per_round for r in returns]
    st = trade_stats(net)
    if not st:
        return Verdict(False, ["no trades"], {"n": 0})
    ctl = [c - cost_per_round for c in (control or [])]
    ctl_mean = statistics.mean(ctl) if ctl else None
    decays = [r / s for r, s in zip(returns, signal or []) if s not in (None, 0)]
    stats = {
        **st,
        "control_n": len(ctl),
        "control_mean": ctl_mean,
        "edge_vs_control": st["mean"] - ctl_mean if ctl_mean is not None else None,
        "decay_median": statistics.median(decays) if decays else None,
        "top5_share": top_share(net),
    }

    reasons = []
    if st["n"] < th["min_n"]:
        reasons.append(f"too few trades ({st['n']} < {th['min_n']})")
    if st["trimmed_mean"] <= th["min_trimmed_mean"]:
        reasons.append(f"trimmed mean not positive ({st['trimmed_mean']:+.2%}): outlier-driven")
    if st["median"] <= th["min_median"]:
        reasons.append(f"median not positive ({st['median']:+.2%}): lottery-ticket shape")
    if st["t"] < th["min_t"]:
        reasons.append(f"t-statistic too low ({st['t']:.2f} < {th['min_t']})")
    if stats["edge_vs_control"] is not None and \
            stats["edge_vs_control"] <= th["min_edge_vs_control"]:
        reasons.append(f"does not beat the control (edge {stats['edge_vs_control']:+.2%})")
    if stats["decay_median"] is not None and stats["decay_median"] < th["min_decay"]:
        reasons.append(f"signal decays too fast (realizes {stats['decay_median']:.0%} "
                       "of what it claimed): it caught a momentary peak")
    if stats["top5_share"] > th["max_top5_share"]:
        reasons.append(f"P&L concentrated in a few trades (best 5% earn "
                       f"{stats['top5_share']:.0%})")
    return Verdict(not reasons, reasons, stats)


def staleness_split(rows: list[dict], key: str = "stale_h",
                    hours: float = STALE_HOURS) -> tuple[list[dict], list[dict]]:
    """Split observations into (fresh, stale) by hours since the last trade.

    Rows without the field go to the stale group: a price that cannot be
    shown to be fresh is treated as the worst case."""
    fresh, stale = [], []
    for r in rows:
        v = r.get(key)
        (fresh if v is not None and v <= hours else stale).append(r)
    return fresh, stale
