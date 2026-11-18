# Deflated Sharpe Ratio: your exercise

The plan says you write this one yourself (it is question 3 in your
`tests/test_walkforward.py`). This page gives the idea, the formula and the
inputs, already computed. The code is yours.

## The idea in one paragraph

If you try N strategies that have no edge at all, the best of them will still
show a positive Sharpe, just by luck, and the more you try the higher it gets.
The DSR asks: given how many things were tried, what is the probability that
the Sharpe you are reporting beats that "best of N lucky ones" benchmark? It
also corrects for fat tails and skew, which make a Sharpe estimate noisier.

## The formula (Bailey & Lopez de Prado, 2014)

1. Expected maximum Sharpe of N zero-edge trials (per period, not annualized):

   SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))

   where V is the variance of the Sharpe estimates across the trials, g is the
   Euler-Mascheroni constant (0.5772...), Z is the inverse standard normal CDF
   (`scipy.stats.norm.ppf`), and e is Euler's number.

2. Probabilistic Sharpe of the observed SR against that benchmark:

   DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2) )

   with Phi the standard normal CDF, T the number of returns, skew and kurt
   (non-excess, so a normal distribution has 3) of the returns.

Everything is per period (daily here). Do not annualize before plugging in.

## Inputs for the case-study portfolio (already computed, 2026-11-17)

Out-of-sample daily returns of the 50/50 portfolio, data through 2026-09-30:

| Input | Value |
|---|---|
| T | 2,160 |
| SR (daily) | 0.06775 (annualized 1.075) |
| skew | 0.194 |
| kurtosis (non-excess) | 13.117 |
| N | strict about 1,650, lenient about 450 (see `research/TRIALS.md`) |
| V | yours to estimate: the variance of the daily Sharpe across the trials. A defensible start: the 21 configurations in `experiments/pbo_grid.py` on QQQ and BTC (`grid_returns`), each config's daily Sharpe, then their variance |

## What to write

1. `quantlab/research/dsr.py` with `expected_max_sharpe(n_trials, var_sr)` and
   `deflated_sharpe(sr, n_trials, var_sr, t, skew, kurt)`.
2. Tests first: with N = 1 the expected maximum is 0; DSR rises with T; with
   zero skew and kurtosis 3 it reduces to the plain probabilistic Sharpe.
3. Run it for both N. The plan's rule: only if DSR > 0.95 may any document call
   the result "significant". Expect it not to be, given P1 and P2.

Be ready to explain in English, in two minutes, why trying more things lowers
the bar a lucky result has to clear.
