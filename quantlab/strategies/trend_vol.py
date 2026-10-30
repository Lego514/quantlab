import numpy as np
import pandas as pd

from ..engine.metrics import _periods_per_year
from .base import Strategy


class TrendVolTarget(Strategy):
    """Long above the long moving average, sized to a volatility target.

    Ported from a sibling project's crypto trend ledger (rules frozen
    2026-07-03): hold while the close is above the simple average of the last
    `ma` closes, today's included, at min(1, target_vol / realized vol).
    Realized vol is the population standard deviation of the last
    `vol_window` daily returns, annualized on the series' own calendar (365
    for crypto), exactly as the original computed it.
    """
    name = "trend_vol"
    param_grid = {"ma": [100, 150, 200], "target_vol": [0.15, 0.20, 0.30]}

    def __init__(self, ma: int = 200, target_vol: float = 0.20, vol_window: int = 30):
        self.ma = int(ma)
        self.target_vol = target_vol
        self.vol_window = int(vol_window)

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        average = close.rolling(self.ma).mean()
        ppy = _periods_per_year(close.index)
        vol = close.pct_change().rolling(self.vol_window).std(ddof=0) * np.sqrt(ppy)
        size = (self.target_vol / vol.where(vol > 0)).clip(upper=1.0)
        return size.where(close > average, 0.0).fillna(0.0)
