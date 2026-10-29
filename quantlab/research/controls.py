"""Random-timing control: does a signal pick better moments than chance?

The control circularly shifts the strategy's own position series by a random
offset and runs it again on the same prices. It keeps everything about the
strategy except its timing: the same exposure, holding periods, turnover and
costs. If the real ordering does not beat most of the shifts, the signal is
not selecting anything; it is just being in the market.

A sibling project found the cost of skipping this: of 57 trend-following
parameter sets, 74% looked good in-sample and only 7% held up out of sample.
`scan` ranks in-sample, so this control is the cheapest check it can carry.
"""
import numpy as np
import pandas as pd

from ..strategies.base import Strategy


class FixedPositions(Strategy):
    """A strategy that replays a given position series."""
    name = "fixed_positions"

    def __init__(self, positions: pd.Series):
        self._positions = positions

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        return self._positions


def shift_control(prices: pd.DataFrame, strategy, backtester, n: int = 200,
                  seed: int = 0, min_shift: int | None = None,
                  funding: pd.Series | None = None) -> dict:
    """Sharpe of the strategy against `n` randomly shifted copies of itself.

    p_value is the share of shifts that do at least as well, counting the real
    run once (so it is never exactly zero).
    """
    actual = backtester.run(prices, strategy, funding=funding)
    positions = strategy.generate_positions(prices).reindex(prices.index).fillna(0.0)
    size = len(positions)
    min_shift = min_shift or max(20, size // 20)
    if size <= 2 * min_shift:
        raise ValueError("series too short for a shift control")

    rng = np.random.default_rng(seed)
    sharpes, control_trades = [], []
    for k in rng.integers(min_shift, size - min_shift, size=n):
        shifted = pd.Series(np.roll(positions.to_numpy(), k), index=positions.index)
        r = backtester.run(prices, FixedPositions(shifted), funding=funding)
        sharpes.append(r.metrics.get("sharpe", 0.0))
        control_trades.extend(r.trade_returns)

    real = float(actual.metrics.get("sharpe", 0.0))
    sharpes = np.asarray(sharpes)
    return {
        "sharpe": real,
        "control_median": float(np.median(sharpes)),
        "control_p95": float(np.percentile(sharpes, 95)),
        "p_value": float((1 + (sharpes >= real).sum()) / (n + 1)),
        "trade_returns": actual.trade_returns,
        "control_trade_returns": control_trades,
    }
