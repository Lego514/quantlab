"""Grid search over a strategy's parameter grid."""
import itertools

import pandas as pd

from ..engine import Backtester


def grid_search(prices: pd.DataFrame, strategy_cls, backtester: Backtester,
                param_grid: dict | None = None, objective: str = "sharpe") -> pd.DataFrame:
    """Run a backtest for every parameter combination, ranked by `objective`.

    Returns a DataFrame with one row per combination, best first.
    WARNING: in-sample results overstate real performance — validate the
    winner with walk_forward before trusting it.
    """
    grid = param_grid or strategy_cls.param_grid
    if not grid:
        raise ValueError(f"{strategy_cls.__name__} has no param_grid")

    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        strategy = strategy_cls(**params)
        result = backtester.run(prices, strategy)
        row = {**params,
               "sharpe": result.metrics.get("sharpe", 0),
               "cagr": result.metrics.get("cagr", 0),
               "max_drawdown": result.metrics.get("max_drawdown", 0),
               "calmar": result.metrics.get("calmar", 0),
               "trades": result.metrics.get("trades", 0)}
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(objective, ascending=False).reset_index(drop=True)
