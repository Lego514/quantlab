"""How uncertain is a backtest number? Resampling answers it.

A Sharpe ratio is a point estimate. The 2026 case study's portfolio trades
about 25 round trips in eight years, so its headline number carries wide
error bars that a single figure hides. These helpers put the bars back:

- `block_bootstrap_sharpe` resamples daily returns in blocks (to keep the
  autocorrelation that trend strategies have) and reports a 95% interval.
- `bootstrap_trades` resamples per-trade returns: interval for the average
  trade and profit factor, and how often the resampled history loses money.
- `trades_needed` / `years_needed` turn the same question around: how much
  evidence it would take to tell this result apart from zero.
"""
import math

import numpy as np
import pandas as pd


def _sharpe(r: np.ndarray, periods_per_year: float) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0


def block_bootstrap_sharpe(returns: pd.Series, periods_per_year: float,
                           n: int = 2000, block: int = 20, seed: int = 0) -> dict:
    """Moving-block bootstrap of an annualized Sharpe ratio."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 2 * block:
        raise ValueError("series too short for the block size")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(len(r) / block)
    sharpes = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(r) - block + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:len(r)]
        sharpes[i] = _sharpe(sample, periods_per_year)
    return {
        "sharpe": _sharpe(r, periods_per_year),
        "ci_low": float(np.percentile(sharpes, 2.5)),
        "ci_high": float(np.percentile(sharpes, 97.5)),
        "p_nonpositive": float((sharpes <= 0).mean()),
    }


def _profit_factor(r: np.ndarray) -> float:
    loss = -r[r <= 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else float("inf")


def bootstrap_trades(trade_returns: list[float], n: int = 2000, seed: int = 0) -> dict:
    """Resample whole trades with replacement."""
    r = np.asarray(trade_returns, dtype=float)
    if len(r) < 2:
        raise ValueError("need at least two trades")
    rng = np.random.default_rng(seed)
    means, pfs, losing = np.empty(n), np.empty(n), 0
    for i in range(n):
        q = rng.choice(r, size=len(r), replace=True)
        means[i] = q.mean()
        pfs[i] = _profit_factor(q)
        losing += np.prod(1 + q) < 1.0
    return {
        "mean": float(r.mean()),
        "mean_ci_low": float(np.percentile(means, 2.5)),
        "mean_ci_high": float(np.percentile(means, 97.5)),
        "pf_p5": float(np.percentile(pfs, 5)),
        "pf_p50": float(np.percentile(pfs, 50)),
        "p_unprofitable": losing / n,
    }


def trades_needed(trade_returns: list[float], z: float = 1.96) -> int | None:
    """Trades needed before the average trade's 95% interval excludes zero,
    if trades keep looking like these. None when the average is not positive."""
    r = np.asarray(trade_returns, dtype=float)
    if len(r) < 2 or r.mean() <= 0:
        return None
    return math.ceil((z * r.std(ddof=1) / r.mean()) ** 2)


def years_needed(sharpe: float, z: float = 1.96) -> float | None:
    """Years of returns before an annualized Sharpe this size is significant:
    its t-statistic is roughly sharpe * sqrt(years)."""
    return (z / sharpe) ** 2 if sharpe > 0 else None
