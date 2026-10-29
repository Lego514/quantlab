import numpy as np
import pandas as pd

from .base import Strategy


def wilder_rsi(close: np.ndarray, period: int = 2) -> np.ndarray:
    """Wilder's RSI: the first average is a simple mean of `period` changes,
    then each step smooths by (period - 1) / period. NaN until defined.

    rsi_reversion's `rsi` seeds its average from the first change instead, so
    the two disagree on short series; this one matches the original ledger.
    """
    out = np.full(len(close), np.nan)
    gains = losses = 0.0
    avg_gain = avg_loss = 0.0
    for i in range(1, len(close)):
        change = close[i] - close[i - 1]
        g, l = max(change, 0.0), max(-change, 0.0)
        if i <= period:
            gains += g
            losses += l
            if i < period:
                continue
            avg_gain, avg_loss = gains / period, losses / period
        else:
            avg_gain = (avg_gain * (period - 1) + g) / period
            avg_loss = (avg_loss * (period - 1) + l) / period
        out[i] = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 100.0
    return out


class Rsi2(Strategy):
    """Connors-style RSI(2) mean reversion for index ETFs.

    Ported from a sibling project's ledger (deployed settings: threshold 10
    with the 200-day filter). Buy at a close where RSI(2) is below
    `threshold` and the close is above the average of the PREVIOUS
    `ma_filter` closes; sell at the first later close above the average of
    the previous `ma_exit` closes.

    The position is path-dependent (in or out), so it is replayed from the
    start of the history given. Exits come within days, so two histories that
    start at different dates agree once each has seen its first exit.
    """
    name = "rsi2"
    param_grid = {"threshold": [5, 10, 15], "ma_exit": [5, 10]}

    def __init__(self, threshold: float = 10, ma_filter: int = 200, ma_exit: int = 5,
                 period: int = 2):
        self.threshold = threshold
        self.ma_filter = int(ma_filter)
        self.ma_exit = int(ma_exit)
        self.period = int(period)

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"].to_numpy(dtype=float)
        rsi = wilder_rsi(close, self.period)
        prior = pd.Series(close)
        filter_avg = prior.rolling(self.ma_filter).mean().shift(1).to_numpy()
        exit_avg = prior.rolling(self.ma_exit).mean().shift(1).to_numpy()

        positions = np.zeros(len(close))
        holding = False
        for i in range(self.ma_filter, len(close)):
            if not holding and rsi[i] < self.threshold and close[i] > filter_avg[i]:
                holding = True
            elif holding and close[i] > exit_avg[i]:
                holding = False
            positions[i] = 1.0 if holding else 0.0
        return pd.Series(positions, index=prices.index)
