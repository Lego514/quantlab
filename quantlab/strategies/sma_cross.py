import pandas as pd

from .base import Strategy


class SmaCross(Strategy):
    """Golden cross trend following: long while fast SMA > slow SMA."""
    name = "sma_cross"
    param_grid = {"fast": [10, 20, 50], "slow": [50, 100, 200]}

    def __init__(self, fast: int = 20, slow: int = 100):
        self.fast = int(fast)
        self.slow = int(slow)

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if self.fast >= self.slow:
            return pd.Series(0.0, index=prices.index)
        close = prices["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        return (fast > slow).astype(float)
