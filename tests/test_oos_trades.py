"""walk_forward reports the trades its test windows opened, so the gate and
the trade bootstrap run on out-of-sample trades only."""
import numpy as np
import pandas as pd

from quantlab.engine import Backtester
from quantlab.research import walk_forward
from quantlab.strategies import Momentum


def test_oos_trades_open_inside_a_test_window(trending_prices):
    rng = np.random.default_rng(3)
    prices = trending_prices.copy()
    prices["close"] = prices["close"] * np.exp(np.cumsum(rng.normal(0, 0.01, len(prices))))
    prices["open"] = prices["close"].shift(1).fillna(prices["close"].iloc[0])

    wf = walk_forward(prices, Momentum, Backtester())
    assert wf["fill"] == "next_open"
    entries = wf["oos_trade_entries"]
    assert len(entries) == len(wf["oos_trade_returns"]) > 0

    spans = [tuple(pd.Timestamp(d) for d in f["test_span"].split(" -> "))
             for f in wf["folds"]]
    for t in entries:
        assert any(start <= t <= end for start, end in spans), t
