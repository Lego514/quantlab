"""The signal-quality gate and its companions: bootstrap error bars,
random-timing control, staleness split."""
import numpy as np
import pandas as pd
import pytest

from quantlab.engine import Backtester
from quantlab.research import (block_bootstrap_sharpe, bootstrap_trades, gate, shift_control,
                               staleness_split, trades_needed, years_needed)
from quantlab.research.signal_quality import top_share
from quantlab.strategies import BuyHold
from quantlab.strategies.base import Strategy


def test_steady_edge_passes():
    rng = np.random.default_rng(0)
    returns = list(rng.normal(0.01, 0.01, 200))
    assert gate(returns).passed


def test_outlier_driven_mean_fails_on_median_and_concentration():
    # 95 small losers and 5 huge winners: positive mean, lottery-ticket shape.
    returns = [-0.01] * 95 + [0.5] * 5
    verdict = gate(returns)
    assert not verdict.passed
    text = " ".join(verdict.reasons)
    assert "median" in text and "concentrated" in text


def test_costs_and_control_are_applied():
    returns = [0.004] * 50
    assert gate(returns).passed
    assert not gate(returns, cost_per_round=0.005).passed
    assert not gate(returns, control=[0.006] * 50).passed


def test_decay_flags_signals_that_claim_more_than_they_realize():
    returns = [0.01] * 50
    assert not gate(returns, signal=[0.05] * 50).passed
    assert gate(returns, signal=[0.012] * 50).passed


def test_top_share():
    assert top_share([1.0] * 20) == pytest.approx(0.05)
    assert top_share([-1.0, -1.0, 3.0]) == pytest.approx(3.0)


def test_staleness_split_puts_unknown_freshness_in_the_stale_group():
    fresh, stale = staleness_split([{"stale_h": 1}, {"stale_h": 12}, {}])
    assert [r.get("stale_h") for r in fresh] == [1]
    assert len(stale) == 2


def test_block_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    ci = block_bootstrap_sharpe(r, periods_per_year=252, n=300)
    assert ci["ci_low"] < ci["sharpe"] < ci["ci_high"]


def test_bootstrap_trades_and_sample_size_helpers():
    out = bootstrap_trades([0.02, -0.01] * 30, n=300)
    assert out["mean_ci_low"] < out["mean"] < out["mean_ci_high"]
    # mean 0.005, sd ~0.0151: (1.96 * sd / mean)^2 ~ 35 trades
    assert trades_needed([0.02, -0.01] * 30) == pytest.approx(35, abs=1)
    assert trades_needed([-0.01] * 10) is None
    assert years_needed(1.0) == pytest.approx(1.96 ** 2)


class _Peeker(Strategy):
    """Cheats: long exactly on the bars before an up move."""
    name = "peeker"

    def generate_positions(self, prices):
        return (prices["close"].shift(-1) > prices["close"]).astype(float)


def test_shift_control_separates_real_timing_from_exposure(trending_prices):
    rng = np.random.default_rng(2)
    noisy = trending_prices.copy()
    noisy["close"] = noisy["close"] * np.exp(rng.normal(0, 0.01, len(noisy)))
    noisy["open"] = noisy["close"]
    bt = Backtester(fill="close")
    assert shift_control(noisy, _Peeker(), bt, n=50)["p_value"] < 0.05
    # Constant exposure has no timing at all: every shift is identical.
    assert shift_control(noisy, BuyHold(), bt, n=50)["p_value"] == pytest.approx(1.0)
