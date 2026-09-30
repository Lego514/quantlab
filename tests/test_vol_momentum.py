"""Vol targeting has to annualize on the series' own calendar.

VolMomentum used to carry a private copy of the periods-per-year rule that
returned 365 for every daily series. For a stock that overstated realized vol
by sqrt(365/252) and sized every position ~17% smaller than target_vol asked.
"""
import numpy as np
import pandas as pd
import pytest

from quantlab.strategies import VolMomentum


@pytest.mark.parametrize("calendar, periods", [("stock_index", 252), ("crypto_index", 365)])
def test_position_size_uses_the_series_own_calendar(calendar, periods, request):
    index = request.getfixturevalue(calendar)
    rng = np.random.default_rng(7)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.002, 0.01, len(index))), index=index)

    # A low target keeps every position below the 1.0 cap, so the vol math is visible.
    positions = VolMomentum(lookback=60, target_vol=0.05, vol_window=30).generate_positions(
        pd.DataFrame({"close": close})
    )

    realized = close.pct_change().rolling(30).std() * np.sqrt(periods)
    expected = (0.05 / realized).clip(upper=1.0)
    long = positions > 0
    assert long.sum() > 100
    np.testing.assert_allclose(positions[long], expected[long])
