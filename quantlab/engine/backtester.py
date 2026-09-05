"""Vectorized backtester with fees and slippage.

Contract with strategies: a strategy takes a price DataFrame and returns a
position Series in [-1, 1] (fraction of equity long/short). The backtester
shifts positions by one bar (no look-ahead: you trade on the NEXT bar after
a signal) and charges costs on position changes.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import compute_metrics


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    positions: pd.Series
    metrics: dict = field(default_factory=dict)
    benchmark_metrics: dict = field(default_factory=dict)
    strategy_name: str = ""
    symbol: str = ""

    def summary(self) -> str:
        def fmt(d: dict) -> str:
            parts = []
            for k in ("total_return", "cagr", "sharpe", "sortino",
                      "max_drawdown", "calmar", "win_rate", "profit_factor", "trades"):
                if k not in d:
                    continue
                v = d[k]
                if k in ("total_return", "cagr", "max_drawdown", "win_rate"):
                    parts.append(f"  {k:15s} {v:>10.2%}")
                elif k == "trades":
                    parts.append(f"  {k:15s} {v:>10d}")
                else:
                    parts.append(f"  {k:15s} {v:>10.2f}")
            return "\n".join(parts)

        lines = [
            f"=== {self.strategy_name} on {self.symbol} ===",
            f"[strategy]",
            fmt(self.metrics),
            f"[buy & hold benchmark]",
            fmt(self.benchmark_metrics),
        ]
        return "\n".join(lines)


class Backtester:
    def __init__(self, fee: float = 0.001, slippage: float = 0.0005,
                 allow_short: bool = False, max_leverage: float = 1.0):
        """fee/slippage are per-trade fractions (0.001 = 0.1%)."""
        self.fee = fee
        self.slippage = slippage
        self.allow_short = allow_short
        self.max_leverage = max_leverage

    def run(self, prices: pd.DataFrame, strategy, symbol: str = "") -> BacktestResult:
        close = prices["close"]
        positions = strategy.generate_positions(prices).reindex(close.index).fillna(0.0)
        positions = positions.clip(-self.max_leverage if self.allow_short else 0,
                                   self.max_leverage)

        # trade on next bar after signal -> shift(1)
        held = positions.shift(1).fillna(0.0)
        asset_returns = close.pct_change().fillna(0.0)

        turnover = held.diff().abs().fillna(held.abs())
        cost = turnover * (self.fee + self.slippage)

        strat_returns = held * asset_returns - cost
        equity = (1 + strat_returns).cumprod()
        benchmark = (1 + asset_returns).cumprod()

        trades, trade_returns = self._extract_trades(held, asset_returns)

        result = BacktestResult(
            equity=equity,
            benchmark=benchmark,
            positions=held,
            strategy_name=getattr(strategy, "name", strategy.__class__.__name__),
            symbol=symbol,
        )
        result.metrics = compute_metrics(equity, trades, trade_returns)
        result.benchmark_metrics = compute_metrics(benchmark)
        return result

    def _extract_trades(self, held: pd.Series, asset_returns: pd.Series):
        """Group consecutive nonzero holdings into trades, net of costs."""
        trades = 0
        trade_returns = []
        in_trade = False
        acc = 1.0
        prev = 0.0
        for pos, ret in zip(held.values, asset_returns.values):
            if pos != 0 and not in_trade:
                in_trade = True
                acc = 1.0
            if in_trade:
                acc *= 1 + pos * ret
            if pos == 0 and in_trade:
                in_trade = False
                trades += 1
                round_trip_cost = 2 * abs(prev) * (self.fee + self.slippage)
                trade_returns.append(acc - 1 - round_trip_cost)
            prev = pos
        if in_trade:
            trades += 1
            trade_returns.append(acc - 1 - 2 * (self.fee + self.slippage))
        return trades, trade_returns
