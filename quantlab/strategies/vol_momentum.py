import numpy as np
import pandas as pd

from ..engine.metrics import _periods_per_year
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

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        trailing = close.pct_change(self.lookback)
        # Same calendar rule the metrics use. This used to be a private copy
        # that returned 365 for every daily series, which overstated a stock's
        # annualized vol by sqrt(365/252) ~ 1.20 and sized its positions ~17%
        # smaller than target_vol asked for.
        ppy = _periods_per_year(close.index)
        realized_vol = close.pct_change().rolling(self.vol_window).std() * np.sqrt(ppy)
        size = (self.target_vol / realized_vol).clip(upper=self.max_position)
        return size.where(trailing > 0, 0.0).fillna(0.0)
