# Preregistration

Every new research idea is written down here, in full, **before any backtest
of it runs**. A result is only evidence for the hypothesis that was stated
before it was seen.

## Rules

- At most **two** new ideas a quarter.
- An entry is committed before its first backtest; the commit time is the proof.
- Once committed, an entry is never edited. A changed idea is a new entry and
  counts toward the quarter's two.
- Every configuration run is added to `research/TRIALS.md`.
- The pass threshold is the one written here. A failure is reported as a
  failure, in the README if the idea was ever mentioned there.
- A new forward ledger may only come from an entry that passed.

## Status of the case study (2026-11-18)

The planned multi-asset trend ledger (`trend_multi_asset`) was conditional on
P2 passing. P2 failed (`reports/falsify_p2.md`), so it is **not** created. No
idea is preregistered yet.

## Template

Copy this block, fill every field, commit, then run.

```
### <short name>  (registered YYYY-MM-DD, quarter YYYY-Qn, idea k of 2)

Hypothesis:       one sentence a reader could prove false
Why it might work: the economic or behavioral reason, with a source
Universe:         the exact list, fixed now
Period:           in-sample dates / out-of-sample dates
Rules:            entry, exit, sizing, costs (fee, slippage, funding)
Parameters:       every value to be tried (this is the trial count)
Benchmark:        what it must beat (e.g. static exposure, buy and hold)
Metric and test:  e.g. Sharpe difference, paired block bootstrap
Pass threshold:   e.g. difference > 0 with p < 0.10, on the out-of-sample period
Kill threshold:   what result ends the idea for good
```

## Entries

(none yet)
