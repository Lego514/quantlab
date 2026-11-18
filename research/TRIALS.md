# Trial registry

Every strategy configuration ever backtested, across this repo and three
sibling research projects. A Sharpe ratio means less the more things were
tried before it was found; this is the count the Deflated Sharpe Ratio needs.

**Rule (from 2026-11-16):** every new backtest run adds a row before its result
is read. Counts below 2026-11-17 are backfilled from the projects' own records,
so they are lower bounds: anything run and never written down is missing.

## Backfilled (to 2026-11-17)

| Project | Study | Configurations | Source |
|---|---|---|---|
| Quantitative Trading | funding-rate arbitrage parameter sets | 288 | BACKTEST_RESULTS.md |
| Quantitative Trading | trend parameter sets (4h EMA, ATR exits) | 58 | BACKTEST_RESULTS.md |
| Quantitative Trading | high-return search (breakout, bands, Donchian, Supertrend, RSI, rotation) | 65 | HIGH_RETURN_RESULTS.md |
| Quantitative Trading | Chan theory entries x exit plans | 192 | CHAN_RESULTS.md |
| Quantitative Trading | harmonic patterns | 320 | HARMONIC_RESULTS.md |
| Quantitative Trading | momentum rotation variants | 6 | MIX_RESULTS.md |
| Quantitative Trading | strategy portfolios (V1 chosen from these) | 15 | MIX_RESULTS.md |
| Quantitative Trading | event, weather, wallet, insider, listing studies | 18 | reports/*PROTOCOL*, *RESULTS* |
| Quant Trading (archived) | ideas carried to a verdict (9 rejected, 6 passed) | 15 | ROADMAP.md |
| Quant Trading (archived) | PO3 grid | 18 | ROADMAP.md round 8 |
| Quant Trading (archived) | trend parameter grid | 9 | ROADMAP.md round 4 |
| Quant Trading (archived) | market-making quote distances x markets | 135 | ROADMAP.md round 10 (5 x 27) |
| quantlab | vol_momentum walk-forward grid, per fold (3 assets x 4 folds x 9) | 108 | STRATEGY_2026.md |
| quantlab | other registry strategies' default grids, scanned | 59 | quantlab/strategies |
| quantlab | altcoin rotation, 1h Donchian, leverage sweep, favorite-bias buckets | 20 | STRATEGY_2026.md appendix |
| quantlab | P1 variants and lookbacks, P2 assets, P3 grid | 25 + 13 + 273 | reports/falsify_p1.md, falsify_p2.md, pbo_p3.md |
| Signal Lab | harmonic pattern families | 12 | VERIFICATION.md |
| **Total** | | **at least 1,649** | |

The P1-P3 rows are falsification runs of one existing claim, not searches for
new ones; they are listed because they are still trials.

## Which number to use

- **Strict:** everything above, about 1,650 (1,649). The case study is one survivor of all of it.
- **Lenient:** only the family the case study comes from: momentum and trend
  variants on daily bars (Quantitative Trading trend 58 + rotation 6, Quant
  Trading trend grid 9, quantlab vol_momentum 108 + P3 grid 273): about 450.

Either way it is hundreds, not one.

## From here on

| Date | Study | Configurations | Result read? |
|---|---|---|---|
| | | | |
