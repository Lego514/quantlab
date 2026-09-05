"""Walk-forward validation: the honest test for "is this actually profitable?"

Repeatedly: optimize parameters on a training window, then trade them
unchanged on the following out-of-sample window. Only the stitched
out-of-sample equity counts. If a strategy only wins in-sample, it's
overfit, not profitable.
"""
import pandas as pd

from ..engine import Backtester, compute_metrics
from .optimizer import grid_search


def walk_forward(prices: pd.DataFrame, strategy_cls, backtester: Backtester,
                 n_splits: int = 4, train_ratio: float = 0.7,
                 objective: str = "sharpe") -> dict:
    """Anchored walk-forward.

    Splits data into n_splits folds; for each fold, trains on everything
    before it and tests on the fold itself.

    Returns {"oos_equity", "oos_metrics", "folds": [per-fold info]}.
    """
    n = len(prices)
    min_train = int(n / (n_splits + 1) * train_ratio) + 1
    fold_size = (n - min_train) // n_splits
    if fold_size < 30:
        raise ValueError("not enough data for that many walk-forward splits")

    oos_returns = []
    folds = []
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_end = train_end + fold_size if i < n_splits - 1 else n
        train = prices.iloc[:train_end]
        test = prices.iloc[train_end:test_end]

        ranked = grid_search(train, strategy_cls, backtester, objective=objective)
        best = ranked.iloc[0]
        param_names = [c for c in ranked.columns
                       if c not in ("sharpe", "cagr", "max_drawdown", "calmar", "trades")]
        params = {}
        for k in param_names:
            v = best[k]
            try:
                f = float(v)
                params[k] = int(f) if f.is_integer() else f
            except (TypeError, ValueError):
                params[k] = v

        # include a little pre-test history so indicators warm up, but only
        # keep returns from the test window
        warmup = min(250, train_end)
        ctx = prices.iloc[train_end - warmup:test_end]
        result = backtester.run(ctx, strategy_cls(**params))
        oos_equity_fold = result.equity.loc[test.index[0]:]
        fold_returns = oos_equity_fold.pct_change().dropna()
        oos_returns.append(fold_returns)

        folds.append({
            "fold": i + 1,
            "train_span": f"{train.index[0].date()} -> {train.index[-1].date()}",
            "test_span": f"{test.index[0].date()} -> {test.index[-1].date()}",
            "params": params,
            "is_sharpe": float(best[objective]) if objective in best else None,
            "oos_sharpe": compute_metrics((1 + fold_returns).cumprod()).get("sharpe"),
        })

    all_returns = pd.concat(oos_returns)
    oos_equity = (1 + all_returns).cumprod()
    return {
        "oos_equity": oos_equity,
        "oos_metrics": compute_metrics(oos_equity),
        "folds": folds,
    }
