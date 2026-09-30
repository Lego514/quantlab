"""Annualization tests — the worked example for the rest of the suite.

Background: `_periods_per_year` used to return 365 for every daily series,
including stock series that only print ~252 bars a year. Because Sharpe is
scaled by sqrt(periods_per_year) and n_years is derived from it too, that
inflated the reported Sharpe of every equity backtest by sqrt(365/252) ~ 1.20
and overstated CAGR by more. STRATEGY_2026.md records the case study restated
on corrected figures.

These tests exist so that never silently comes back.
"""
import numpy as np
import pandas as pd
import pytest

from quantlab.engine.metrics import _periods_per_year, compute_metrics


class TestPeriodsPerYear:
    def test_weekday_only_series_annualizes_on_trading_days(self, stock_index):
        assert _periods_per_year(stock_index) == 252.0

    def test_series_with_weekend_bars_annualizes_on_calendar_days(self, crypto_index):
        assert _periods_per_year(crypto_index) == 365.0

    def test_stock_and_crypto_are_distinguished_despite_equal_step(
        self, stock_index, crypto_index
    ):
        # Both have a one-day median step: the discriminator has to be the
        # presence of weekend bars, not the step size.
        step = lambda idx: pd.Series(idx).diff().median()
        assert step(stock_index) == step(crypto_index)
        assert _periods_per_year(stock_index) != _periods_per_year(crypto_index)

    def test_weekly_bars_fall_back_to_step_derived_rate(self):
        weekly = pd.date_range("2020-01-06", periods=200, freq="7D")
        assert _periods_per_year(weekly) == pytest.approx(52.14, abs=0.1)

    def test_degenerate_index_does_not_explode(self):
        assert _periods_per_year(pd.DatetimeIndex(["2020-01-01"])) == 252.0


class TestComputeMetrics:
    def test_sharpe_scales_with_the_annualization_factor(self, stock_index):
        rng = np.random.default_rng(0)
        equity = pd.Series(
            (1 + rng.normal(0.0004, 0.01, len(stock_index))).cumprod(), index=stock_index
        )
        on_252 = compute_metrics(equity, periods_per_year=252)["sharpe"]
        on_365 = compute_metrics(equity, periods_per_year=365)["sharpe"]
        # This ratio is exactly the size of the bug that was fixed.
        assert on_365 / on_252 == pytest.approx(np.sqrt(365 / 252), rel=1e-9)

    def test_explicit_override_beats_inference(self, stock_index):
        equity = pd.Series(np.linspace(1.0, 2.0, len(stock_index)), index=stock_index)
        inferred = compute_metrics(equity)["years"]
        overridden = compute_metrics(equity, periods_per_year=365)["years"]
        assert inferred != overridden

    def test_years_matches_the_real_calendar_span(self, stock_index):
        # 1000 weekdays is about 3.97 calendar years; if n_years is computed on
        # 365 instead of 252 it comes out near 2.7 and CAGR inflates with it.
        equity = pd.Series(np.linspace(1.0, 2.0, len(stock_index)), index=stock_index)
        span_years = (stock_index[-1] - stock_index[0]).days / 365.25
        assert compute_metrics(equity)["years"] == pytest.approx(span_years, rel=0.05)
