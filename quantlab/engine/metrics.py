"""Performance metrics for an equity curve."""
import numpy as np
import pandas as pd


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    """Observations per year, used to annualize Sharpe, volatility and CAGR.

    A daily stock series and a daily 24/7 crypto series BOTH have a one-day
    median step, so step size alone cannot tell them apart. Weekend timestamps
    are the discriminator: a market that prints bars on Saturdays trades ~365
    days a year, one that does not trades ~252. Choosing wrong scales Sharpe by
    sqrt(365/252) ~ 1.20 and distorts CAGR by more, because n_years is derived
    from this too.
    """
    if len(index) < 2:
        return 252.0
    index = pd.DatetimeIndex(index)
    median_step = pd.Series(index).diff().median()
    seconds = median_step.total_seconds()
    if seconds <= 0:
        return 252.0

    sessions_per_year = 365.0 if bool(index.dayofweek.isin((5, 6)).any()) else 252.0
    per_day = 86_400 / seconds

    if per_day >= 2:  # intraday: bars per session x sessions per year
        return per_day * sessions_per_year
    if seconds <= 86_400:  # daily bars
        return sessions_per_year
    # Coarser than daily (weekly, monthly): derive straight from the step.
    return 365 / (seconds / 86_400)


def compute_metrics(equity: pd.Series, trades: int = 0,
                    trade_returns: list | None = None,
                    periods_per_year: float | None = None) -> dict:
    """Compute standard performance stats from an equity curve (start=1.0).

    `periods_per_year` overrides the inference in `_periods_per_year` — pass it
    when the calendar is known, rather than trusting the index to reveal it.
    """
    equity = equity.dropna()
    if len(equity) < 2:
        return {"error": "not enough data"}

    returns = equity.pct_change().dropna()
    ppy = periods_per_year if periods_per_year else _periods_per_year(equity.index)
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
