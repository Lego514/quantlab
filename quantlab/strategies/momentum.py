import pandas as pd

from .base import Strategy


class Momentum(Strategy):
    """Time-series momentum: long when trailing return over `lookback` bars
    is positive (above `threshold`). Works well on crypto and trending stocks;
    on Polymarket it expresses "news momentum" in probabilities."""
    name = "momentum"
    param_grid = {"lookback": [20, 60, 90, 120], "threshold": [0.0, 0.05]}

    def __init__(self, lookback: int = 90, threshold: float = 0.0):
        self.lookback = int(lookback)
        self.threshold = threshold

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        trailing = prices["close"].pct_change(self.lookback)
        return (trailing > self.threshold).astype(float)
