"""Performance metrics for an equity curve."""
import numpy as np
import pandas as pd


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    median_step = pd.Series(index).diff().median()
    seconds = median_step.total_seconds()
    if seconds <= 0:
        return 252.0
    per_day = 86_400 / seconds
    if per_day >= 2:          # intraday data trades around the clock
        return per_day * 365
    if per_day > 0.5:         # daily data: assume trading days for stocks-ish cadence
        return 365 if seconds <= 86_400 else 252
    return 365 / (seconds / 86_400)


def compute_metrics(equity: pd.Series, trades: int = 0,
                    trade_returns: list | None = None) -> dict:
    """Compute standard performance stats from an equity curve (start=1.0)."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {"error": "not enough data"}

    returns = equity.pct_change().dropna()
    ppy = _periods_per_year(equity.index)
    n_years = len(returns) / ppy

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 1e-9)) - 1

    vol = returns.std() * np.sqrt(ppy)
    sharpe = (returns.mean() / returns.std() * np.sqrt(ppy)) if returns.std() > 0 else 0.0

    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std() * np.sqrt(ppy)) if len(downside) > 1 and downside.std() > 0 else float("inf") if returns.mean() > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd < 0 else float("inf") if cagr > 0 else 0.0

    stats = {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_periods": len(returns),
        "years": n_years,
        "trades": trades,
    }

    if trade_returns:
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        stats["win_rate"] = len(wins) / len(trade_returns)
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        stats["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else float("inf")
        stats["avg_trade"] = float(np.mean(trade_returns))
    return stats
