# Forward ledger governance

Rules fixed on 2026-11-18, before any ledger had a quarter-end review. They
exist so that a bad month is not a reason to change anything, and a dead
ledger is not kept alive by hope.

## Reviews

- **When:** the last day of each quarter (Mar 31, Jun 30, Sep 30, Dec 31). The
  first is **2026-12-31**. Between reviews the ledgers are not judged and nothing
  about them changes.
- **How:** `python main.py forward evaluate --record`. It writes
  `forward/reviews/<date>.md` and adds each ledger's percentile to
  `forward/reviews/history.json`. Run without `--record` any time to look;
  that output is informational and cannot trigger anything.
- **What it measures, per ledger:**
  - the live return so far, as a percentile of same-length returns from a block
    bootstrap of the frozen rules' backtest over the 8 years before the
    ledger started;
  - the live drawdown against the backtest's worst;
  - across ledgers: correlation of daily returns and an equal-weight book.

For `portfolio_2026` the backtest's parameters were chosen on part of that
history, so its reference is optimistic.

## Kill rules

A ledger stops when either holds at a recorded review:

1. its percentile is below **5** at **two quarter-ends in a row**; or
2. its drawdown is deeper than **1.5 times** the backtest's worst.

Stopping means: the spec stays, the ledger stops being recorded (its spec is
moved to `forward/retired/`), and the review states the cause. A stopped ledger
is not restarted.

## What never happens

- A spec is never edited. New rules are a new ledger with a new name.
- No new ledger without a preregistration (`research/PREREGISTRATION.md`),
  at most two new ideas a quarter.
- Thresholds above are not changed after a review is seen.

## Reading the numbers honestly

- The three ledgers are close to one bet: in their first weeks their daily
  returns correlated at 0.57 to 0.86. They are not three pieces of evidence.
- A Sharpe around 1 needs years to tell from zero. A few months of results,
  good or bad, are mostly noise; that is why reviews are quarterly and the kill
  rule needs two of them.
