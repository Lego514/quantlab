import pandas as pd

from .base import Strategy


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).fillna(50)


class RsiReversion(Strategy):
    """Mean reversion: buy oversold (RSI < lower), exit when RSI recovers."""
    name = "rsi_reversion"
    param_grid = {"window": [7, 14, 21], "lower": [20, 30], "exit": [50, 60]}

    def __init__(self, window: int = 14, lower: float = 30, exit: float = 55):
        self.window = int(window)
        self.lower = lower
        self.exit = exit

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        r = rsi(prices["close"], self.window)
        positions = pd.Series(index=prices.index, dtype=float)
        positions[r < self.lower] = 1.0
        positions[r > self.exit] = 0.0
        return positions.ffill().fillna(0.0)
