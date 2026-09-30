# Case Study: Volatility-Targeted Cross-Asset Momentum Portfolio

> First published 2026-07-05; **restated 2026-09-30** — see the correction below. **Every number is walk-forward out-of-sample** — measured on data the parameter search never saw — and net of 0.1% fees + 0.05% slippage. Reproduction commands at the bottom.

This is a worked example of what the engine is for: take a plausible idea, subject it to out-of-sample validation, and report the result honestly whether or not it flatters the idea.

## Correction (2026-09-30)

The first version of this page reported a portfolio Sharpe of **1.29**. The figure is now **1.05**. Two separate things moved it:

- **Three more months of out-of-sample data** account for most of the change. Re-running the original, unfixed pipeline on data through 2026-09-30 gives 1.09.
- **Three calendar bugs**, all now fixed and covered by tests in `tests/`:
  1. **Annualization.** `metrics.py` annualized every daily series over 365 periods, including stock series that only trade about 252 days a year. That overstated equity Sharpe by about 20% (√(365/252)) and CAGR by more.
  2. **The same rule inside the strategy.** `VolMomentum` kept its own copy of that rule, so it overestimated QQQ's volatility and held positions about 17% smaller than its volatility target asked for.
  3. **Dropped weekends.** The portfolio took each leg's daily returns and then dropped every day where either leg was missing. QQQ has no weekend bars, so Bitcoin's Saturday and Sunday returns were discarded instead of carried into Monday.

The bugs pushed in opposite directions. On today's data, fixing annualization alone takes the Sharpe from 1.09 to 0.91; restoring the weekend returns brings it to 1.01; fixing the strategy's sizing brings it to 1.05. Fixing only the most obvious bug would have understated the result.

The leverage experiment had a related fourth bug: it charged financing once per trading day, so weekends were free. That is fixed too, and it matters more there (see the appendix).

## The rules

Two legs, 50% of capital each: **QQQ** (Nasdaq-100 ETF) and **BTC**. Each leg runs independently:

1. **Entry signal** — trailing return over 60–120 days > 0 → hold; otherwise flat. The walk-forward picks the lookback for each fold.
2. **Position sizing** — `position % = target volatility ÷ trailing 30-day realized annualized volatility`, capped at 100%. The walk-forward chose a 15% target in most folds.
   - Example: BTC realized vol at 60% annualized → deploy only 25% of that leg's capital.
   - Size shrinks automatically when markets get violent and expands in calm uptrends. This is where the drawdown reduction comes from, not from a better entry signal.
3. **Execution cadence** — check once per day at the close; trade only when the signal flips. The backtest fires roughly 25 round trips over 8 years.

## Out-of-sample results (Feb 2018 – Sep 2026)

| Same window, Feb 2018 – Sep 2026 | Sharpe | Annualized return | Max drawdown |
|---|---|---|---|
| **50/50 portfolio** | **1.05** | **10.8%** | **-16.4%** |
| Reference: QQQ buy & hold | 0.88 | 19.8% | -35.1% |
| Reference: BTC buy & hold | 0.71 | 27.5% | **-76.6%** |

Each leg over its own out-of-sample span: **QQQ** 0.78 Sharpe, 10.1% a year, -16.0% max drawdown (from Feb 2018); **BTC** 1.34 Sharpe, 19.4% a year, -24.3% max drawdown (from May 2016).

Out-of-sample return correlation between the two legs is **0.16** — the diversification is real, not an artifact.

The portfolio's annualized return is lower than either asset held outright because volatility targeting deliberately trades upside for risk control. On a risk-adjusted basis it beats both: a portfolio that returns 10.8% at a 16.4% drawdown is a materially different instrument from Bitcoin returning 27.5% at a 76.6% drawdown, or QQQ returning 19.8% at 35.1%.

## Why I think this is a real effect, not a curve fit

1. **Walk-forward is positive in every one of 4 time folds, on 3 assets** (QQQ, BTC, ETH), with different selected parameters per fold. Overfit strategies don't survive that.
2. **Momentum is among the most durable anomalies in the literature** (Jegadeesh & Titman 1993 onward) — documented in equities, commodities, FX, and crypto, across a century of data.
3. **It has a behavioral basis**: investors under-react to news, so trends persist. That mechanism doesn't expire in 2026.
4. **Parameters are not sensitive** — lookbacks of 60, 90, and 120 days all work. The result doesn't depend on one lucky number.

## Known weaknesses (the honest section)

- **On its own, the QQQ leg does not beat holding QQQ**: 0.78 Sharpe against 0.88. What it does is halve the drawdown (-16.0% against -35.1%). The portfolio's edge comes from combining two weakly correlated legs, not from a better signal on either one.
- **Choppy, range-bound markets are the failure mode.** Momentum gets whipsawed: the BTC leg's out-of-sample Sharpe fell to 0.56 in 2021–2024 and to 0.22 in 2024–2026.
- **One fold looks overfit.** QQQ's first fold (2018–2020, through the COVID crash) scored 2.05 in-sample and only 0.11 out-of-sample.
- **The volatility target is barely identified.** While positions stay under the 100% cap, a higher target just scales exposure, so Sharpe hardly moves and the grid's pick between 15%, 25% and 40% can flip between runs — it did for one ETH fold, with an identical out-of-sample Sharpe (0.32) either way. Drawdown changes a lot (ETH's stitched drawdown went from -16% to -37%), so the target is really a risk choice the backtest cannot optimize, not a parameter it discovers.
- ~11% annualized is not spectacular; it's "sustainably decent." Any strategy claiming a stable 50%+ should be treated as suspect until proven out-of-sample.
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
| 1.5x–4x leverage on the validated portfolio | — | 4x returns 11.2% a year at -64% drawdown; the best, 12.2% at 3x, costs -52% | Volatility drag and financing: returns barely move, drawdowns scale almost linearly. Dead. |
| Polymarket favorite bias | — | See experiment output above | The only plausible high-return source found — but capacity-limited |

The leverage row changed the most in the restatement. The first version reported 26.8% a year at 4x; charging financing on weekends and fixing the calendar bugs above brought that down to 11.2%. The verdict did not change; it got stronger.

**The lesson:** the honest way to raise returns is not a cleverer signal. It's either accepting the roughly 1:1.5 return-to-drawdown physics of trend following, or finding a structural mispricing (as in prediction markets) — and then being rigorous about whether the sample size supports the claim.

## Reproduce

```bash
pytest                                   # offline tests, including the calendar fixes
python main.py walkforward --market stock  --symbol QQQ     --strategy vol_momentum --period 10y
python main.py walkforward --market crypto --symbol BTC-USD --strategy vol_momentum --days 2000
python main.py walkforward --market crypto --symbol ETH-USD --strategy vol_momentum --days 2000
python experiments/portfolio_2026.py
python experiments/leverage_sweep.py
python experiments/polymarket_favorite_bias.py
```
