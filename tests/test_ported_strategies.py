"""trend_vol and rsi2 were ported from a sibling project's ledgers. Moving a
rule between codebases is where it silently changes, so each is checked bar
for bar against a transcription of the original code."""
import math
import statistics

import numpy as np
import pandas as pd
import pytest

from quantlab.strategies import Rsi2, TrendVolTarget


# --- the original rules, transcribed from the sibling project --------------

def original_trend_signal(closes, ma=200, vol_window=30, vol_target=0.20):
    """trend/paper.py signal(): position for the next day, from closes so far."""
    if len(closes) < max(ma, vol_window) + 1:
        return 0.0
    if closes[-1] <= sum(closes[-ma:]) / ma:
        return 0.0
    daily = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - vol_window, len(closes))]
    rv = statistics.pstdev(daily) * math.sqrt(365)
    return min(1.0, vol_target / rv) if rv > 0 else 0.0


def original_rsi(closes, period=2):
    """trend/rsi_backtest.py rsi()."""
    out = [None] * len(closes)
    gains = losses = 0.0
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        g, l = max(ch, 0), max(-ch, 0)
        if i <= period:
            gains += g
            losses += l
            if i == period:
                ag, al = gains / period, losses / period
                out[i] = 100 - 100 / (1 + ag / al) if al > 0 else 100.0
        else:
            ag = (ag * (period - 1) + g) / period
            al = (al * (period - 1) + l) / period
            out[i] = 100 - 100 / (1 + ag / al) if al > 0 else 100.0
    return out


def original_rsi2_positions(closes, threshold=10, ma_filter=200, ma_exit=5):
    """trend/rsi_paper.py process_symbol(), replayed from the first bar."""
    r2 = original_rsi(closes, 2)
    holding, out = False, [0.0] * len(closes)
    for i in range(ma_filter, len(closes)):
        ma200 = sum(closes[i - ma_filter:i]) / ma_filter
        ma5 = sum(closes[i - ma_exit:i]) / ma_exit
        if not holding and r2[i] is not None and r2[i] < threshold and closes[i] > ma200:
            holding = True
        elif holding and closes[i] > ma5:
            holding = False
        out[i] = 1.0 if holding else 0.0
    return out


# --- synthetic markets with both regimes -----------------------------------

def _walk(index, seed, vol):
    rng = np.random.default_rng(seed)
    drift = np.where(np.arange(len(index)) % 400 < 250, 0.002, -0.002)  # up, then down
    close = 100 * np.cumprod(1 + drift + rng.normal(0, vol, len(index)))
    return pd.DataFrame({"open": close, "close": close}, index=index)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_trend_vol_matches_the_original_rule_every_day(seed):
    prices = _walk(pd.date_range("2020-01-01", periods=900, freq="D"), seed, 0.03)
    ours = TrendVolTarget().generate_positions(prices).to_numpy()
    closes = prices["close"].tolist()
    # The original needs one bar more than the average does before it speaks.
    theirs = [original_trend_signal(closes[:i + 1]) for i in range(len(closes))]
    np.testing.assert_allclose(ours[201:], theirs[201:], rtol=1e-9, atol=1e-12)
    assert (ours > 0).sum() > 100 and (ours == 0).sum() > 100  # both regimes exercised


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_rsi2_matches_the_original_rule_every_day(seed):
    prices = _walk(pd.bdate_range("2015-01-01", periods=1500), seed, 0.012)
    ours = Rsi2().generate_positions(prices).tolist()
    theirs = original_rsi2_positions(prices["close"].tolist())
    assert ours == theirs
    assert 20 < sum(1 for a, b in zip(theirs, theirs[1:]) if b > a)  # many entries


def test_rsi2_waits_out_the_filter_and_exits_above_the_short_average():
    # 205 calm rising bars, then a two-day dip that stays above the 200-day
    # average, then a bounce.
    up = list(100 + 0.1 * np.arange(205))            # ends at 120.4
    closes = up + [119.4, 118.4, 121.4, 121.9]
    prices = pd.DataFrame({"close": closes}, index=pd.bdate_range("2020-01-01", periods=len(closes)))
    pos = Rsi2().generate_positions(prices).tolist()
    assert pos[:205] == [0.0] * 205
    # 119.4: RSI(2) = 9.1 < 10 and above the average -> buy.
    # 118.4: not above the prior 5-close average (120.08) -> hold.
    # 121.4: above the prior 5-close average (119.74) -> sell.
    # 121.9: RSI(2) is high again -> stay out.
    assert pos[205:] == [1.0, 1.0, 0.0, 0.0]
