import pandas as pd

from .base import Strategy


class DonchianBreakout(Strategy):
    """Channel breakout (turtle-style): go long when close makes a new
    `entry`-bar high, exit when it makes a new `exit`-bar low. On intraday
    bars this trades far more often than daily momentum."""
    name = "donchian"
    param_grid = {"entry": [30, 55, 100], "exit": [15, 25, 50]}

    def __init__(self, entry: int = 55, exit: int = 25):
        self.entry = int(entry)
        self.exit = int(exit)

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        upper = close.rolling(self.entry).max().shift(1)
        lower = close.rolling(self.exit).min().shift(1)

        positions = pd.Series(index=prices.index, dtype=float)
        positions[close > upper] = 1.0
        positions[close < lower] = 0.0
        return positions.ffill().fillna(0.0)
