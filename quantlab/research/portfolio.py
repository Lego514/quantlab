"""Combining legs that trade on different calendars into one portfolio."""
import pandas as pd


def leg_returns(equity: dict[str, pd.Series]) -> pd.DataFrame:
    """Daily returns of each leg, on the days every leg trades.

    Align the equity *levels* first, then take returns. The old construction
    took each leg's returns first and then dropped every row where a leg was
    missing, which silently threw away a 24/7 asset's weekend: QQQ has no
    Saturday or Sunday bar, so Bitcoin's Friday-to-Sunday move was dropped
    instead of landing on Monday.
    """
    levels = pd.DataFrame(equity).dropna()
    return levels.pct_change().dropna()
