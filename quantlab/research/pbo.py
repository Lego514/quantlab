"""Probability of backtest overfitting, by combinatorially symmetric
cross-validation (Bailey, Borwein, Lopez de Prado & Zhu, 2015).

Take the daily returns of N configurations over T days. Cut the days into S
blocks. For every way of choosing half the blocks as "in sample": pick the
configuration with the best in-sample Sharpe and see where it ranks on the
other half. PBO is the share of splits where the in-sample winner lands in
the bottom half out of sample. Near 0: picking the best config works. Near
0.5 or above: the best in-sample config is no better than a random one.
"""
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def pbo(returns: pd.DataFrame, n_blocks: int = 16) -> dict:
    """returns: one column per configuration, one row per day, no NaN."""
    r = returns.dropna().to_numpy(dtype=float)
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks must be even and at least 4")
    if r.shape[1] < 2 or len(r) < 2 * n_blocks:
        raise ValueError("need at least two configurations and two days per block")
    blocks = np.array_split(np.arange(len(r)), n_blocks)
    # Per block sums let every split's Sharpe come from additions, not a rescan.
    count = np.array([len(b) for b in blocks], dtype=float)
    total = np.array([r[b].sum(axis=0) for b in blocks])
    square = np.array([(r[b] ** 2).sum(axis=0) for b in blocks])

    def sharpe(idx):
        n = count[idx].sum()
        mean = total[idx].sum(axis=0) / n
        var = square[idx].sum(axis=0) / n - mean ** 2
        return mean / np.sqrt(np.where(var > 0, var, np.nan))

    logits = []
    everything = np.arange(n_blocks)
    for chosen in combinations(range(n_blocks), n_blocks // 2):
        ins = np.array(chosen)
        oos = np.setdiff1d(everything, ins)
        best = int(np.nanargmax(sharpe(ins)))
        oos_sr = sharpe(oos)
        # relative rank of the in-sample winner out of sample, in (0, 1)
        rank = (np.sum(oos_sr < oos_sr[best]) + 1) / (len(oos_sr) + 1)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "splits": comb(n_blocks, n_blocks // 2),
            "median_logit": float(np.median(logits)), "configs": r.shape[1]}
