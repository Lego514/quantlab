import pandas as pd

from .base import Strategy


class BuyHold(Strategy):
    """Always fully long. The benchmark every strategy has to beat."""
    name = "buy_hold"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index)
