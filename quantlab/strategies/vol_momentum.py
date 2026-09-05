import numpy as np
import pandas as pd

from .base import Strategy


class VolMomentum(Strategy):
    """Time-series momentum with volatility targeting.

    Long when trailing return is positive, but size the position so the
    portfolio targets `target_vol` annualized: position = target_vol / realized_vol
    (capped at 1). High-volatility chop gets small positions; calm uptrends get
    full size. This is the classic "risk-managed momentum" construction.
    """
    name = "vol_momentum"
    param_grid = {"lookback": [60, 90, 120], "target_vol": [0.15, 0.25, 0.40]}

    def __init__(self, lookback: int = 90, target_vol: float = 0.25,
                 vol_window: int = 30, max_position: float = 1.0):
        self.lookback = int(lookback)
        self.target_vol = target_vol
        self.vol_window = int(vol_window)
        self.max_position = max_position

    def _periods_per_year(self, index: pd.DatetimeIndex) -> float:
        if len(index) < 2:
            return 252.0
        seconds = pd.Series(index).diff().median().total_seconds()
        if seconds <= 0:
            return 252.0
        per_day = 86_400 / seconds
        if per_day >= 2:
            return per_day * 365
        return 365 if seconds <= 86_400 else 252

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        trailing = close.pct_change(self.lookback)
        ppy = self._periods_per_year(close.index)
        realized_vol = close.pct_change().rolling(self.vol_window).std() * np.sqrt(ppy)
        size = (self.target_vol / realized_vol).clip(upper=self.max_position)
        return size.where(trailing > 0, 0.0).fillna(0.0)
