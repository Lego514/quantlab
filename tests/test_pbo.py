"""Probability of backtest overfitting (CSCV)."""
import numpy as np
import pandas as pd
import pytest

from quantlab.research.pbo import pbo


def _noise(n_days=1600, n_configs=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (n_days, n_configs)))


def test_pure_noise_configs_are_often_overfit():
    # No config is better in population. Within one finite sample some got
    # lucky over the whole period, and both halves share that luck, so PBO
    # sits below one half; but it is far from the near-zero of a real edge.
    results = [pbo(_noise(seed=s), n_blocks=10) for s in range(6)]
    assert all(r["pbo"] > 0.1 for r in results)
    assert np.mean([r["pbo"] for r in results]) > 0.25
    assert results[0]["splits"] == 252  # C(10, 5)


def test_a_real_edge_is_not_overfitting():
    returns = _noise(seed=1)
    returns[0] += 0.002  # one config earns 20 bp a day more, in every block
    out = pbo(returns, n_blocks=10)
    assert out["pbo"] < 0.05
    assert out["median_logit"] > 0


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        pbo(_noise(), n_blocks=7)
    with pytest.raises(ValueError):
        pbo(_noise(n_configs=1))
