"""phase-52.3: the Ledoit-Wolf SR-difference test (sharpe_diff_test) -- correctness.

Pins the TEST's behavior on synthetic data (not the empirical 52wh verdict, which is the
live_check's job): identical paired series -> no significant difference (high one-sided p);
a clearly-dominant series -> p<0.05, positive delta, CI lower bound > 0; deterministic (seeded).
$0, no network.
"""
import numpy as np

from backend.backtest.analytics import sharpe_diff_test


def test_identical_series_not_significant():
    rng = np.random.default_rng(0)
    x = list(rng.normal(0.01, 0.04, 60))
    r = sharpe_diff_test(x, x, periods_per_year=12, n_boot=2000, seed=1)
    assert abs(r["delta"]) < 1e-9          # same series -> zero SR difference
    assert r["p_one_sided"] > 0.5          # cannot reject H0: SR_a <= SR_b
    assert r["ci_low"] <= 0 <= r["ci_high"]


def test_clearly_better_series_is_significant():
    # a = b + a consistent positive per-period increment -> SR_a strictly > SR_b every resample
    rng = np.random.default_rng(0)
    b = rng.normal(0.005, 0.04, 60)
    a = b + 0.02
    r = sharpe_diff_test(list(a), list(b), periods_per_year=12, n_boot=2000, seed=1)
    assert r["delta"] > 0
    assert r["p_one_sided"] < 0.05         # significant edge
    assert r["ci_low"] > 0                 # CI lower bound positive (R2 magnitude leg)


def test_worse_series_not_significant():
    # a is WORSE than b -> one-sided H0 (SR_a <= SR_b) cannot be rejected -> high p
    rng = np.random.default_rng(0)
    b = rng.normal(0.01, 0.04, 60)
    a = b - 0.02
    r = sharpe_diff_test(list(a), list(b), periods_per_year=12, n_boot=2000, seed=1)
    assert r["delta"] < 0
    assert r["p_one_sided"] > 0.5


def test_deterministic_with_seed():
    rng = np.random.default_rng(0)
    a = list(rng.normal(0.01, 0.05, 50)); b = list(rng.normal(0.008, 0.05, 50))
    r1 = sharpe_diff_test(a, b, n_boot=1500, seed=7)
    r2 = sharpe_diff_test(a, b, n_boot=1500, seed=7)
    assert r1["p_one_sided"] == r2["p_one_sided"] and r1["ci_low"] == r2["ci_low"]


def test_handles_none_and_short():
    assert sharpe_diff_test([None, 1.0], [0.5, None], n_boot=100)["p_one_sided"] == 1.0  # <10 -> safe default
