"""Chart generation for backtest results."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def plot_backtest(result, filename: str | None = None) -> Path:
    """Equity curve vs benchmark + drawdown + position chart."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    ax = axes[0]
    ax.plot(result.equity.index, result.equity.values,
            label=f"{result.strategy_name} (sharpe {result.metrics.get('sharpe', 0):.2f})")
    ax.plot(result.benchmark.index, result.benchmark.values,
            label=f"buy & hold (sharpe {result.benchmark_metrics.get('sharpe', 0):.2f})",
            alpha=0.7)
    ax.set_title(f"{result.strategy_name} on {result.symbol}")
    ax.set_ylabel("equity (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)

    dd = result.equity / result.equity.cummax() - 1
    axes[1].fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.5)
    axes[1].set_ylabel("drawdown")
    axes[1].grid(alpha=0.3)

    axes[2].fill_between(result.positions.index, result.positions.values, 0,
                         color="tab:blue", alpha=0.5, step="post")
    axes[2].set_ylabel("position")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    name = filename or f"{result.symbol}_{result.strategy_name}.png".replace("/", "-")
    path = REPORT_DIR / name
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_walkforward(wf: dict, symbol: str, strategy_name: str) -> Path:
    equity = wf["oos_equity"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values,
            label=f"out-of-sample equity (sharpe {wf['oos_metrics'].get('sharpe', 0):.2f})")
    ax.set_title(f"walk-forward OOS: {strategy_name} on {symbol}")
    ax.set_ylabel("equity (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = REPORT_DIR / f"wf_{symbol}_{strategy_name}.png".replace("/", "-")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
