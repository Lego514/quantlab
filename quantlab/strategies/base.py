"""Strategy base class.

A strategy converts a price DataFrame (must contain "close") into a position
Series: 0 = flat, 1 = fully long, values in between = partial. The backtester
handles execution lag and costs, so strategies stay pure signal logic.
"""
from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name = "strategy"

    #: parameter grid used by the optimizer, e.g. {"fast": [10, 20], "slow": [50, 100]}
    param_grid: dict = {}

    @abstractmethod
    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        ...

    def describe(self) -> str:
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in params.items())})"
