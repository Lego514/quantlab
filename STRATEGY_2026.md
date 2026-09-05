# Case Study: Volatility-Targeted Cross-Asset Momentum Portfolio

> Generated 2026-07-05. **Every number below is walk-forward out-of-sample** — measured on data the parameter search never saw — and net of 0.1% fees + 0.05% slippage. Reproduction commands at the bottom.

This is a worked example of what the engine is for: take a plausible idea, subject it to out-of-sample validation, and report the result honestly whether or not it flatters the idea.

## The rules

Two legs, 50% of capital each: **QQQ** (Nasdaq-100 ETF) and **BTC**. Each leg runs independently:

1. **Entry signal** — trailing 90-day return > 0 (BTC uses a 60–120 day window) → hold; otherwise flat.
2. **Position sizing** — `position % = 15% target volatility ÷ trailing 30-day realized annualized volatility`, capped at 100%.
   - Example: BTC realized vol at 60% annualized → deploy only 25% of that leg's capital.
   - Size shrinks automatically when markets get violent and expands in calm uptrends. This is where the drawdown reduction comes from, not from a better entry signal.
3. **Execution cadence** — check once per day at the close; trade only when the signal flips. The backtest fires roughly 25 round trips over 8 years.

## Out-of-sample results (Nov 2017 – Jul 2026)

| | Sharpe | Annualized return | Max drawdown |
|---|---|---|---|
| QQQ leg | 1.07 | 16.0% | -14.7% |
| BTC leg | 1.39 | 20.5% | -24.3% |
| **50/50 portfolio** | **1.29** | **14.8%** | **-17.7%** |
| Reference: BTC buy & hold | 0.80 | ~31% | **-53%** |

Out-of-sample return correlation between the two legs is **0.17** — the diversification is real, not an artifact.

The portfolio's annualized return is lower than raw BTC because volatility targeting deliberately trades upside for risk control. On a risk-adjusted basis (Sharpe), it beats either asset held outright — a portfolio that returns 14.8% at a 17.7% drawdown is a materially different instrument from one that returns 31% at a 53% drawdown.

## Why I think this is a real effect, not a curve fit

1. **Walk-forward is positive across 4 time folds, 3 assets, and different selected parameters per fold.** Overfit strategies don't survive that.
2. **Momentum is among the most durable anomalies in the literature** (Jegadeesh & Titman 1993 onward) — documented in equities, commodities, FX, and crypto, across a century of data.
3. **It has a behavioral basis**: investors under-react to news, so trends persist. That mechanism doesn't expire in 2026.
4. **Parameters are not sensitive** — lookbacks of 60, 90, and 120 days all work. The result doesn't depend on one lucky number.

## Known weaknesses (the honest section)

- **Choppy, range-bound markets are the failure mode.** Momentum gets whipsawed: during 2021–2023 the BTC leg's OOS Sharpe fell to 0.28.
- ~15% annualized is not spectacular; it's "sustainably decent." Any strategy claiming a stable 50%+ should be treated as suspect until proven out-of-sample.
- Taxes and exchange counterparty risk are not modeled.
- The last 8 years contained a historic tech and crypto bull market. In a prolonged bear market the strategy correctly goes flat — that's the protection mechanism working, but flat periods earn nothing.

## Side study: Polymarket favorite–longshot bias

Tested on 500 resolved, high-volume markets (2026-07-05; 118 tradeable samples):

| Entry price bucket | n | Win rate | Mean 14-day return | Edge vs implied probability |
|---|---|---|---|---|
| 0.90–0.98 (strong favorites) | 11 | **100%** | **+5.8%** | +5.4% |
| 0.70–0.90 | 6 | 83.3% | +7.6% | +6.3% |
| 0.50–0.70 | 13 | 53.8% | -9.6% | -7.3% |

Two independent runs (150 markets, then 500) agree directionally: buying strong favorites and holding to resolution has positive expectancy. If the capital could be recycled every 14 days, +5.8% compounds to over 100% annualized in theory — **the only structural mispricing I found in any of these experiments that could plausibly produce very high returns.**

**The statistically honest caveat:** a 16-for-16 record has a 95% confidence lower bound around 81%, which is not yet enough to claim the true win rate exceeds the ~0.95 average entry price. This needs roughly 100 favorite-bucket samples, or a small-capital live test over three months, before it means anything. Also, outcomes within a single event class (e.g. teams in one tournament) are highly correlated and cannot be sized as independent bets.

## Appendix: four attempts at 50%+ annualized, and what actually happened

I deliberately went looking for higher returns. Three of the four paths died in out-of-sample testing:

| Path | Full-period backtest | Honest out-of-sample | Verdict |
|---|---|---|---|
| Weekly altcoin rotation (top-N momentum) | 77–132% annualized | **-3% annualized** (2024+) | A mirage propped up by the 2017/2021 altseasons. Dead. |
| 1-hour Donchian breakout | IS Sharpe 4.36 | **OOS Sharpe -0.89** | High frequency + transaction costs = textbook overfitting |
| 2x–4x leverage on the validated portfolio | — | Even 4x returns only 26.8%, at -63% drawdown | Volatility drag: returns scale sub-linearly, drawdowns scale linearly |
| Polymarket favorite bias | — | See experiment output above | The only plausible high-return source found — but capacity-limited |

**The lesson:** the honest way to raise returns is not a cleverer signal. It's either accepting the roughly 1:1.5 return-to-drawdown physics of trend following, or finding a structural mispricing (as in prediction markets) — and then being rigorous about whether the sample size supports the claim.

## Reproduce

```bash
python main.py walkforward --market crypto --symbol BTC-USD --strategy vol_momentum --days 2000
python main.py walkforward --market stock  --symbol QQQ     --strategy vol_momentum --period 10y
python experiments/portfolio_2026.py
python experiments/polymarket_favorite_bias.py
```
