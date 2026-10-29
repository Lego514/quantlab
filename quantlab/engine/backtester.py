"""Vectorized backtester with fees, slippage and financing.

Contract with strategies: a strategy takes a price DataFrame and returns a
position Series in [-1, 1] (fraction of equity long/short), decided at each
bar's close. The backtester trades that decision at the NEXT bar's open when
the data has opens (fill="next_open"), so a signal is never filled at the
price that produced it. Without opens (Polymarket history), or with
fill="close", it fills at the signal bar's own close: the engine's only
behaviour before 2026-10, kept so the two can be compared.

Per bar, with E the equity at the previous close:
    overnight   E *= 1 + old * gap        gap   = open / previous close - 1
    rebalance   E *= 1 - |new - old| * (fee + slippage)
    session     E *= 1 + new * intra      intra = close / open - 1
    financing   E *= 1 - new * funding - max(-new, 0) * borrow
where old is the position carried overnight and new is the one set at the
open. In close mode gap is 0 and intra is the whole close-to-close move.
"""
from dataclasses import dataclass, field

import pandas as pd

from .metrics import compute_metrics

FILLS = ("next_open", "close")


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    positions: pd.Series
    metrics: dict = field(default_factory=dict)
    benchmark_metrics: dict = field(default_factory=dict)
    strategy_name: str = ""
    symbol: str = ""
    trade_returns: list = field(default_factory=list)
    trade_entries: pd.DatetimeIndex | None = None
    fill: str = ""
    data_source: str = ""

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
            f"fill: {self.fill}   data: {self.data_source or 'unknown'}",
            f"[strategy]",
            fmt(self.metrics),
            f"[buy & hold benchmark]",
            fmt(self.benchmark_metrics),
        ]
        return "\n".join(lines)


def trade_entry_times(held: pd.Series) -> pd.DatetimeIndex:
    """Bars where a trade opens, in the order `_extract_trades` reports them."""
    opened = (held != 0) & (held.shift(1).fillna(0.0) == 0)
    return held.index[opened]


class Backtester:
    def __init__(self, fee: float = 0.001, slippage: float = 0.0005,
                 allow_short: bool = False, max_leverage: float = 1.0,
                 fill: str = "next_open", borrow_rate: float = 0.0):
        """fee/slippage are per-trade fractions (0.001 = 0.1%).

        fill: "next_open" or "close" (see the module docstring).
        borrow_rate: annual rate charged on short exposure per calendar day.
        """
        if fill not in FILLS:
            raise ValueError(f"fill must be one of {FILLS}")
        self.fee = fee
        self.slippage = slippage
        self.allow_short = allow_short
        self.max_leverage = max_leverage
        self.fill = fill
        self.borrow_rate = borrow_rate

    def run(self, prices: pd.DataFrame, strategy, symbol: str = "",
            funding: pd.Series | None = None) -> BacktestResult:
        """funding: optional funding rate per bar (perpetuals), indexed like
        `prices`. Longs pay a positive rate and shorts receive it."""
        close = prices["close"]
        positions = strategy.generate_positions(prices).reindex(close.index).fillna(0.0)
        positions = positions.clip(-self.max_leverage if self.allow_short else 0,
                                   self.max_leverage)

        # decided at the close of bar t, traded on bar t+1 -> shift(1)
        held = positions.shift(1).fillna(0.0)
        carried = held.shift(1).fillna(0.0)
        asset_returns = close.pct_change().fillna(0.0)
        gap, intra, fill = self._session_returns(prices)

        turnover = (held - carried).abs()
        cost = turnover * (self.fee + self.slippage)
        financing = self._financing(held, close.index, funding)

        strat_returns = ((1 + carried * gap) * (1 - cost) * (1 + held * intra)
                         * (1 - financing) - 1)
        equity = (1 + strat_returns).cumprod()
        benchmark = (1 + asset_returns).cumprod()

        trades, trade_returns = self._extract_trades(held, intra, gap)

        result = BacktestResult(
            equity=equity,
            benchmark=benchmark,
            positions=held,
            strategy_name=getattr(strategy, "name", strategy.__class__.__name__),
            symbol=symbol,
            trade_returns=trade_returns,
            trade_entries=trade_entry_times(held),
            fill=fill,
            data_source=prices.attrs.get("source", ""),
        )
        result.metrics = compute_metrics(equity, trades, trade_returns)
        result.benchmark_metrics = compute_metrics(benchmark)
        return result

    def _session_returns(self, prices: pd.DataFrame):
        """(overnight gap, open-to-close move, fill actually used) per bar."""
        close = prices["close"]
        opens = prices["open"] if "open" in prices.columns else None
        usable = opens is not None and bool(opens.notna().all()) and bool((opens > 0).all())
        if self.fill == "next_open" and usable:
            gap = (opens / close.shift(1) - 1).fillna(0.0)
            return gap, close / opens - 1, "next_open"
        zero = pd.Series(0.0, index=close.index)
        return zero, close.pct_change().fillna(0.0), "close"

    def _financing(self, held: pd.Series, index: pd.DatetimeIndex,
                   funding: pd.Series | None) -> pd.Series:
        """Funding paid on the position plus borrow on short exposure, as a
        fraction of equity per bar."""
        out = pd.Series(0.0, index=index)
        if funding is not None:
            out = out + held * funding.reindex(index).fillna(0.0)
        if self.borrow_rate:
            # A Monday bar spans the weekend: charge the calendar time it covers.
            days = index.to_series().diff().dt.total_seconds() / 86_400
            days = days.fillna(days.median() if days.notna().any() else 1.0)
            out = out + (-held).clip(lower=0) * self.borrow_rate * days / 365
        return out

    def _extract_trades(self, held: pd.Series, asset_returns: pd.Series,
                        gap: pd.Series | None = None):
        """Group consecutive nonzero holdings into trades, net of costs.

        `asset_returns` is what the position set on each bar earns (the whole
        bar in close mode, open-to-close otherwise). `gap` is the overnight
        move the previous position earns before it can be changed, so a
        trade's last gap belongs to it even though it exits at that open.
        """
        if gap is None:
            gap = pd.Series(0.0, index=held.index)
        trades = 0
        trade_returns = []
        in_trade = False
        acc = 1.0
        prev = 0.0
        for pos, ret, g in zip(held.values, asset_returns.values, gap.values):
            if in_trade:
                acc *= 1 + prev * g
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
