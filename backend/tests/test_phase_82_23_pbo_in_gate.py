"""phase-82.23 -- PBO reaches the promotion gate, and cannot arrive as a lie.

The gate was never fail-open: `PromotionGate.evaluate` drops a candidate with no
`pbo` as "missing_dsr_or_pbo". The defect was upstream -- `compute_pbo` returns
**0.0, the best possible value, when N < 2 or T < S*2**, and 0.0 passes every
ceiling. So an undersized matrix does not fail to inform; it manufactures a PASS.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.autoresearch.gate import PromotionGate
from backend.backtest.analytics import (
    PBO_CEILING_CANONICAL,
    PBO_CEILING_LIVE,
    PBO_MIN_TRIALS_GATE_GRADE,
    compute_pbo,
    compute_pbo_checked,
)


def _matrix(T: int, N: int, seed: int = 82_23) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(T, N))


# ── the false-good 0.0, which is the whole defect ────────────────────

@pytest.mark.parametrize("T,N,why", [
    (100, 1, "N < 2"),
    (20, 8, "T < S*2"),
    (10, 1, "both"),
])
def test_raw_compute_pbo_returns_a_false_good_zero(T, N, why):
    """Pin the behaviour being defended against, so the wrapper's value is
    visible and a future 'simplification' back to the raw call is loud."""
    assert compute_pbo(_matrix(T, N)) == 0.0, why
    assert 0.0 <= PBO_CEILING_LIVE, "0.0 passes the live ceiling -- that is the danger"


@pytest.mark.parametrize("T,N", [(100, 1), (20, 8), (10, 1)])
def test_checked_wrapper_refuses_instead_of_returning_zero(T, N):
    r = compute_pbo_checked(_matrix(T, N))
    assert r["pbo"] is None, f"expected refusal, got {r}"
    assert r["refused"], "a refusal must say why"
    assert r["gate_grade"] is False


def test_checked_wrapper_returns_a_real_value_when_well_sized():
    r = compute_pbo_checked(_matrix(1000, 16))
    assert r["pbo"] is not None and 0.0 <= r["pbo"] <= 1.0
    assert r["n_trials"] == 16 and r["n_obs"] == 1000
    assert r["refused"] is None


def test_checked_wrapper_rejects_a_non_2d_input():
    assert compute_pbo_checked(np.zeros(10))["pbo"] is None


# ── N travels WITH the value ─────────────────────────────────────────

def test_value_always_carries_its_trial_count():
    """A bare float cannot be told apart from a gate-grade one. The wrapper
    returns a dict specifically so a consumer cannot receive PBO without N."""
    r = compute_pbo_checked(_matrix(1000, 8))
    assert set(r) >= {"pbo", "n_trials", "n_obs", "gate_grade", "refused"}


def test_undersized_trial_count_is_marked_not_gate_grade():
    """N=8 -- the phase-82.3 shape -- is directional, not promotion-grade."""
    assert compute_pbo_checked(_matrix(1000, 8))["gate_grade"] is False
    assert compute_pbo_checked(_matrix(1000, PBO_MIN_TRIALS_GATE_GRADE))["gate_grade"] is True


# ── the gate ─────────────────────────────────────────────────────────

def test_gate_still_fails_closed_on_a_missing_pbo():
    """Unchanged behaviour, pinned: a missing PBO has never promoted anything."""
    v = PromotionGate().evaluate({"dsr": 0.99, "trial_id": "t1"})
    assert v["promoted"] is False and v["reason"] == "missing_dsr_or_pbo"


def test_gate_refuses_a_pbo_computed_from_too_few_trials():
    """The phase-82.23 addition. A sound number from 8 trials is still not
    promotion-grade."""
    v = PromotionGate().evaluate(
        {"dsr": 0.99, "pbo": 0.05, "pbo_n_trials": 8, "trial_id": "t2"})
    assert v["promoted"] is False
    assert "pbo_trials_below_min" in v["reason"]


def test_gate_promotes_when_pbo_is_good_and_well_sized():
    """Mirror guard -- the addition must not make the gate always refuse."""
    v = PromotionGate().evaluate(
        {"dsr": 0.99, "pbo": 0.05, "pbo_n_trials": 32, "trial_id": "t3"})
    assert v["promoted"] is True, v


def test_gate_is_unchanged_for_producers_that_do_not_report_n():
    """Additive: a producer that never emitted `pbo_n_trials` must behave
    exactly as before, or this step breaks every existing caller."""
    v = PromotionGate().evaluate({"dsr": 0.99, "pbo": 0.05, "trial_id": "t4"})
    assert v["promoted"] is True


def test_gate_rejects_the_measured_incumbent_pbo():
    """The phase-82.3 measurement, end to end: triple_barrier's PBO 0.7486
    must not promote under ANY of the repo's three ceilings."""
    v = PromotionGate().evaluate(
        {"dsr": 0.99, "pbo": 0.7486, "pbo_n_trials": 32, "trial_id": "incumbent"})
    assert v["promoted"] is False
    assert "pbo_above_max" in v["reason"]
    assert 0.7486 > PBO_CEILING_LIVE and 0.7486 > PBO_CEILING_CANONICAL


def test_gate_rejects_a_non_numeric_trial_count():
    v = PromotionGate().evaluate(
        {"dsr": 0.99, "pbo": 0.05, "pbo_n_trials": "many", "trial_id": "t5"})
    assert v["promoted"] is False and "non_numeric_pbo_n_trials" in v["reason"]


# ── the reconciled thresholds ────────────────────────────────────────

def test_the_live_ceiling_is_the_strict_one():
    """Three ceilings existed (0.20 live, 0.5 dead, 0.5 documented). The
    constants must record which is enforced, or the next reader re-derives the
    confusion."""
    assert PBO_CEILING_LIVE == 0.20
    assert PBO_CEILING_CANONICAL == 0.50
    assert PBO_CEILING_LIVE < PBO_CEILING_CANONICAL
    assert PromotionGate().max_pbo == PBO_CEILING_LIVE, (
        "the live gate's ceiling has drifted from the recorded constant"
    )


# ── PBO cannot live in generate_report ───────────────────────────────

def test_generate_report_still_does_not_emit_a_pbo():
    """Deliberate. `generate_report` receives ONE BacktestResult; PBO needs N
    configurations. Emitting one there would mean N=1 -> a hard-coded 0.0 ->
    an unconditional PASS on every single run. This test exists so a future
    'why doesn't the report have PBO?' does not get answered by adding it."""
    import inspect

    from backend.backtest import analytics

    src = inspect.getsource(analytics.generate_report)
    assert "compute_pbo" not in src, (
        "generate_report now computes PBO, but it sees a single run: N=1 makes "
        "compute_pbo return a false-good 0.0 that passes every ceiling"
    )


# ── criterion 4: trial diversity accompanies every reported PBO ──────

def test_every_reported_pbo_carries_a_column_correlation():
    """CSCV ranks the N columns against each other across time subsets, so
    near-identical columns make that ranking noise-driven HOWEVER LARGE N IS.
    A PBO without this number cannot be told apart from one computed over a
    genuinely diverse configuration set."""
    r = compute_pbo_checked(_matrix(1000, 12))
    assert r["pbo"] is not None
    assert r["column_corr_mean"] is not None, "no trial-diversity number reported"
    assert r["column_corr_max"] is not None
    assert "columns_diverse" in r


def test_diversity_number_discriminates_independent_from_duplicate_columns():
    """A diagnostic that reports the same value for both is decoration."""
    rng = np.random.default_rng(3)
    independent = compute_pbo_checked(rng.normal(size=(1000, 12)))

    base = rng.normal(size=(1000, 1))
    near_dup = compute_pbo_checked(base + rng.normal(size=(1000, 12)) * 0.01)

    assert independent["column_corr_mean"] < 0.1, independent["column_corr_mean"]
    assert near_dup["column_corr_mean"] > 0.99, near_dup["column_corr_mean"]
    assert independent["columns_diverse"] is True
    assert near_dup["columns_diverse"] is False, (
        "near-duplicate columns were reported as a diverse configuration set"
    )


def test_adapter_forwards_the_diversity_number_to_the_gate():
    """The value must reach the consumer, not merely exist in the helper."""
    import inspect

    from backend.autoresearch import strategy_backtest_adapter as ad

    src = inspect.getsource(ad)
    assert "pbo_column_corr_mean" in src, (
        "the adapter computes diversity but never emits it, so the gate cannot see it"
    )
    assert "pbo_columns_diverse" in src
