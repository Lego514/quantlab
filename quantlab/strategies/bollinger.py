import pandas as pd

from .base import Strategy


class BollingerReversion(Strategy):
    """Mean reversion: buy when close drops below the lower Bollinger band,
    exit when it crosses back above the middle band."""
    name = "bollinger"
    param_grid = {"window": [20, 40], "n_std": [1.5, 2.0, 2.5]}

    def __init__(self, window: int = 20, n_std: float = 2.0):
        self.window = int(window)
        self.n_std = n_std

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        mid = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        lower = mid - self.n_std * std

        positions = pd.Series(index=prices.index, dtype=float)
        positions[close < lower] = 1.0
        positions[close > mid] = 0.0
        return positions.ffill().fillna(0.0)
