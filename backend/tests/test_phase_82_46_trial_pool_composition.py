"""phase-82.46 -- the optimizer's trial pool, decided rather than inherited.

THE STEP'S PREMISE WAS FALSE AND THIS FILE PINS THE TRUTH. The step (and the
82.16 comment it inherited) says the trial POOL is a direct input to Deflated
Sharpe. It is not: `compute_deflated_sharpe` has no pool parameter and
`num_trials` is incremented once per ITERATION. `test_pool_size_does_not_enter_dsr`
proves it by identity.

What IS true is sharper, and it is what the removals in this step act on: the N
increment happens BEFORE the try/except around `run_backtest`, so an experiment
that CRASHES still costs a trial -- and DSR is steep in N. A pool member that
cannot run on the configured window therefore burns Deflated Sharpe on every
iteration that selects it. `test_crashed_experiment_still_costs_a_trial` pins
the ordering that makes that true.

GUARD-DESIGN NOTE. Criterion 4 asks for a guard that fails when the pool changes
without the decision being updated. A test that merely restated the list would
be a tautology. Instead the pool is DERIVED from an executable rule and the
decision record is a dict compared against it, so both directions fail: a member
without a rationale, and a rationale without a member.

And nothing here source-scans for "blend" -- it appears in comments in both
`quant_optimizer.py` and `backtest_engine.py`, which is exactly the trap that
bit 82.43. Every assertion is on imported VALUES or on `resolve_strategy`
BEHAVIOUR.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.backtest.backtest_engine import (
    NON_COMPARABLE_STRATEGIES,
    STRATEGY_REGISTRY,
    resolve_strategy,
)
from backend.backtest.quant_optimizer import (
    AVAILABLE_STRATEGIES,
    POOL_DECISION,
    _PARAM_BOUNDS,
    _selectable_strategies,
    selectable_strategies_for_window,
)

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "handoff" / "current" / "contract_82.46.md"


# ---------------------------------------------------------------------------
# Criterion 1 -- the set is DERIVED from a documented decision
# ---------------------------------------------------------------------------


def test_pool_is_derived_from_the_rule_not_restated():
    """The rule: registry key AND not demoted. Asserted against the live
    registry so the pool cannot drift from it the way the pre-82.16 literal
    did."""
    assert AVAILABLE_STRATEGIES, "the pool must not be empty"
    expected = [s for s in STRATEGY_REGISTRY if s not in NON_COMPARABLE_STRATEGIES]
    assert AVAILABLE_STRATEGIES == expected == _selectable_strategies()
    # The demoted names must be absent, and they must really BE demoted --
    # otherwise this asserts nothing.
    assert NON_COMPARABLE_STRATEGIES, "fixture precondition: something is demoted"
    for name in NON_COMPARABLE_STRATEGIES:
        assert name not in AVAILABLE_STRATEGIES, name


def test_the_demoted_exclusion_clause_is_not_a_no_op():
    """A mutation run caught this: the demoted names are NOT in the live
    registry (82.16 removed them), so `if s not in _NON_COMPARABLE` currently
    filters nothing and deleting it leaves every live assertion green. The
    clause is defensive -- it matters the moment a demoted strategy is
    re-registered -- but a defensive clause with no observable effect is
    indistinguishable from dead code.

    So drive it with a SYNTHETIC registry in which a demoted name IS present.
    """
    import backend.backtest.quant_optimizer as qo

    assert NON_COMPARABLE_STRATEGIES, "fixture precondition"
    demoted = sorted(NON_COMPARABLE_STRATEGIES)[0]
    assert demoted not in STRATEGY_REGISTRY, (
        "the live registry now contains a demoted name -- the clause is no "
        "longer a no-op and this guard's premise must be re-derived")

    fake_registry = dict(STRATEGY_REGISTRY)
    fake_registry[demoted] = "_compute_triple_barrier_label"
    orig = qo._STRATEGY_REGISTRY
    try:
        qo._STRATEGY_REGISTRY = fake_registry
        got = qo._selectable_strategies()
    finally:
        qo._STRATEGY_REGISTRY = orig
    assert demoted not in got, (
        f"{demoted} is demoted but the rule offered it: {got}")
    assert set(got) == set(fake_registry) - set(NON_COMPARABLE_STRATEGIES)


def test_every_pool_member_has_a_recorded_rationale():
    """CRITERION 4, direction 1: a member without a decision fails."""
    assert set(POOL_DECISION) == set(AVAILABLE_STRATEGIES), (
        f"pool and decision record disagree: "
        f"pool-only={sorted(set(AVAILABLE_STRATEGIES) - set(POOL_DECISION))}, "
        f"record-only={sorted(set(POOL_DECISION) - set(AVAILABLE_STRATEGIES))}")
    for name, why in POOL_DECISION.items():
        assert isinstance(why, str) and len(why) > 40, (
            f"{name}'s rationale is too thin to be a decision: {why!r}")


def test_a_rationale_for_a_non_member_also_fails():
    """CRITERION 4, direction 2, driven through the REAL comparison.

    An earlier version built a LOCAL dict, added a key to it, and asserted the
    local dict differed from the pool -- true by construction and unable to fail
    for any production defect. The 82.46 Q/A named it. This version injects the
    stale record into the module and re-runs the production check, the same way
    `test_the_demoted_exclusion_clause_is_not_a_no_op` injects a registry.
    """
    import backend.backtest.quant_optimizer as qo

    def _decision_matches_pool() -> bool:
        return set(qo.POOL_DECISION) == set(qo.AVAILABLE_STRATEGIES)

    assert _decision_matches_pool(), "precondition: the live record is consistent"

    orig = qo.POOL_DECISION
    try:
        qo.POOL_DECISION = dict(orig, a_strategy_not_in_the_pool="x" * 50)
        assert not _decision_matches_pool(), (
            "a rationale for a NON-member must break the comparison -- if this "
            "passes, the check is a subset test, not a set equality")
        qo.POOL_DECISION = {k: v for k, v in orig.items()
                            if k != sorted(orig)[0]}
        assert not _decision_matches_pool(), (
            "a member with NO rationale must break it too")
    finally:
        qo.POOL_DECISION = orig
    assert _decision_matches_pool(), "the injection must be fully undone"


def test_the_decision_is_recorded_in_the_step_artifact():
    """CRITERION 1 requires the rationale in the artifact, not only in code."""
    assert CONTRACT.is_file(), CONTRACT
    text = CONTRACT.read_text(encoding="utf-8")
    assert "## 3. THE DECISION" in text
    for name in AVAILABLE_STRATEGIES:
        assert name in text, f"{name} is selectable but unexplained in the artifact"


# ---------------------------------------------------------------------------
# Criterion 3 -- no selectable name resolves to a different strategy
# ---------------------------------------------------------------------------


def test_no_selectable_name_resolves_to_a_different_strategy():
    """CRITERION 3, driven through the production `resolve_strategy` -- NOT a
    source scan. `blend` appears in comments in two modules, so a text search
    would be satisfied by prose."""
    for name in AVAILABLE_STRATEGIES:
        effective, was_demoted = resolve_strategy(name)
        assert effective == name, (
            f"{name!r} is offered but RUNS as {effective!r}; results would be "
            "scored under the wrong name")
        assert was_demoted is False, name


def test_blend_is_gone_from_the_pool_and_would_still_be_coerced():
    """`blend` had no implementation and resolved to the incumbent. It is out of
    the pool -- and the coercion it would have suffered is still demonstrated,
    so the reason for removing it is visible rather than asserted."""
    assert "blend" not in AVAILABLE_STRATEGIES
    assert "blend" not in STRATEGY_REGISTRY, (
        "if blend is ever implemented, this step's decision must be revisited")
    effective, _ = resolve_strategy("blend")
    assert effective == "triple_barrier", (
        "the coercion is the defect: an offered name that runs as another and "
        "is then scored under the requested one")


def test_the_dead_blend_weight_params_are_gone():
    """They had no engine reader and were 4 of 24 proposable params, so ~1
    proposal in 6 spent a full walk-forward run and a DSR-costing trial on a
    parameter nothing consumed."""
    for dead in ("tb_weight", "qm_weight", "mr_weight", "fm_weight"):
        assert dead not in _PARAM_BOUNDS, dead
    assert len(_PARAM_BOUNDS) == 20, (
        f"expected 20 proposable params after removing 4 of 24, got "
        f"{len(_PARAM_BOUNDS)} -- re-measure before changing this")
    # The surviving params must still be a real set, or the assertion above
    # could pass on an emptied dict.
    assert "tp_pct" in _PARAM_BOUNDS and "sl_pct" in _PARAM_BOUNDS


# ---------------------------------------------------------------------------
# Criterion 2 -- the DSR impact, MEASURED
# ---------------------------------------------------------------------------


def test_pool_size_does_not_enter_dsr():
    """CRITERION 2, and it refutes the step's own premise.

    `compute_deflated_sharpe` takes `num_trials`, not a pool. Two calls at the
    same iteration count are bit-identical however the pool changed -- which is
    a stronger measurement than a sampled before/after comparison, because it
    holds for every input rather than the ones I happened to try.
    """
    import inspect

    from backend.backtest.analytics import compute_deflated_sharpe

    params = inspect.signature(compute_deflated_sharpe).parameters
    assert "num_trials" in params
    assert not any("pool" in p or "strateg" in p for p in params), (
        f"a pool parameter appeared; the premise of this step changes: {list(params)}")

    # What this CAN establish: holding the DSR inputs fixed, the pool has no
    # separate term. Stated as a signature fact, NOT dressed up as a before/after
    # measurement -- an earlier version called the same function twice with
    # byte-identical arguments and labelled the two results
    # "DSR(pool_before)"/"DSR(pool_after)". That was f(x)==f(x): unfalsifiable,
    # and the labels invented a provenance neither call had. The 82.46 Q/A named
    # it and was right.
    assert compute_deflated_sharpe(1.5, 26, variance_of_srs=0.5, T=252) == pytest.approx(
        0.796298, abs=1e-6)


def test_dsr_IS_pool_dependent_through_its_inputs():
    """The correction to my own over-generalisation, pinned so it cannot be
    re-made.

    "No pool parameter" does NOT mean "the pool cannot affect DSR". At the real
    call site `analytics.generate_report` feeds DSR
    `observed_sr=result.aggregate_sharpe` and
    `variance_of_srs=np.var(window_sharpes)` -- both functions of WHICH strategy
    was sampled. So the pool moves DSR through its inputs, and any claim that the
    pool change is DSR-neutral needs an empirical before/after, not a signature.
    """
    from backend.backtest.analytics import compute_deflated_sharpe as f

    # Same trial count, different realised inputs -> different DSR.
    a = f(1.5, 26, variance_of_srs=0.5, T=252)
    b = f(1.2, 26, variance_of_srs=0.5, T=252)
    c = f(1.5, 26, variance_of_srs=1.5, T=252)
    assert a != b, "observed_sr moves DSR at a fixed trial count"
    assert a != c, "variance_of_srs moves DSR at a fixed trial count"

    src = (REPO / "backend" / "backtest" / "analytics.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "observed_sr=result.aggregate_sharpe" in code, (
        "the DSR call no longer reads aggregate_sharpe; re-derive this argument")
    assert "variance_of_srs=sr_variance" in code


def test_dsr_is_steep_in_the_trial_COUNT():
    """The mechanism that DOES matter: N, not pool size. These are the numbers
    recorded in the step artifact."""
    from backend.backtest.analytics import compute_deflated_sharpe

    got = {n: round(compute_deflated_sharpe(1.5, n, variance_of_srs=0.5, T=252), 4)
           for n in (1, 10, 26, 50, 100)}
    assert got[1] == 1.0 and got[10] == 1.0, got
    assert got[26] == pytest.approx(0.7963, abs=1e-4), got
    assert got[50] == pytest.approx(0.1164, abs=1e-4), got
    assert got[100] == pytest.approx(0.0008, abs=1e-4), got
    assert got[100] < got[50] < got[26] < got[10], (
        "DSR must be monotonically decreasing in N, or the wasted-iteration "
        "argument in the artifact does not hold")


def test_crashed_experiment_still_costs_a_trial():
    """The ordering that makes an unrunnable pool member expensive: the trial
    counter increments BEFORE the try/except that wraps the run.

    Structural (line ordering), because the alternative is executing a full
    optimizer iteration. Kept honest by asserting the anchors are unique, so
    this cannot pass on an arbitrary match.
    """
    src = (REPO / "backend" / "backtest" / "quant_optimizer.py").read_text(
        encoding="utf-8").splitlines()
    inc = [i + 1 for i, l in enumerate(src) if "self.num_trials += 1" in l]
    run = [i + 1 for i, l in enumerate(src) if "result = self.engine.run_backtest(" in l]
    exc = [i + 1 for i, l in enumerate(src) if "QuantOptimizer: experiment" in l]
    assert inc and run and exc, (inc, run, exc)
    experiment_inc = min(inc)
    experiment_run = max(run)
    assert experiment_inc < experiment_run, (
        f"num_trials increments at {experiment_inc} but the run is at "
        f"{experiment_run}; if the order ever reverses, a crashed experiment "
        "stops costing a trial and the artifact's argument must be revisited")
    assert experiment_run < min(exc), "the crash handler must follow the run"


# ---------------------------------------------------------------------------
# The window-aware pool -- derived from 82.21, not hardcoded
# ---------------------------------------------------------------------------


def test_unrunnable_members_are_excluded_by_a_DERIVED_rule():
    """`qarp` cannot be labelled before the fundamentals coverage start, and
    82.21 makes the engine RAISE there. The exclusion is derived from the same
    predicate the engine uses -- a hardcoded {"qarp"} would go stale the moment
    another fundamentals-dependent strategy is registered."""
    from backend.backtest.fundamentals_coverage import (
        FUNDAMENTALS_COVERAGE_START,
        label_fundamentals_dependent_strategies,
    )

    dependent = label_fundamentals_dependent_strategies()
    assert dependent, "fixture precondition: something is fundamentals-dependent"

    uncovered = selectable_strategies_for_window("2018-01-01")
    covered = selectable_strategies_for_window("2025-01-01")

    assert covered == AVAILABLE_STRATEGIES, (
        "a covered window must not lose any member")
    assert set(AVAILABLE_STRATEGIES) - set(uncovered) == (
        dependent & set(AVAILABLE_STRATEGIES)), (
        "exactly the fundamentals-dependent members must drop out")
    assert uncovered, "the pool must not empty out"
    assert len(uncovered) < len(covered), (
        "if these are equal the exclusion did nothing and the guard is vacuous")

    # The boundary is the MEASURED coverage start, not a typed date.
    #
    # phase-82.51 MOVED that boundary. `FUNDAMENTALS_COVERAGE_START` is the
    # earliest `report_date` in the table, but the publication-lag embargo means
    # no row is VISIBLE until 60 days later -- so at the raw start the engine now
    # raises `REFUSED`, and the optimizer must agree with it. The intent of this
    # assertion is unchanged (a DERIVED boundary, never a typed date); only which
    # derived value is the boundary has changed.
    from backend.backtest.fundamentals_coverage import effective_coverage_start

    assert selectable_strategies_for_window(effective_coverage_start()) == covered

    # And the raw start is now INSIDE the embargo gap, so the dependent members
    # must drop out there. Pinning both sides keeps this from silently reverting
    # to the pre-82.51 boundary.
    at_raw_start = selectable_strategies_for_window(FUNDAMENTALS_COVERAGE_START)
    assert set(AVAILABLE_STRATEGIES) - set(at_raw_start) == (
        dependent & set(AVAILABLE_STRATEGIES)), (
        "at the RAW coverage start nothing is visible yet (82.51 embargo), so "
        "the fundamentals-dependent members must be excluded there too")


def test_the_window_exclusion_is_derived_not_a_literal():
    """A mutation run caught this too: replacing the derivation with a
    hardcoded {"qarp"} survived every assertion, because qarp IS today's
    answer. No test over live data can tell a derivation from a literal that
    happens to be correct -- so inject a SYNTHETIC dependent set and require the
    filter to follow it.
    """
    other = "mean_reversion"
    assert other in AVAILABLE_STRATEGIES, "fixture precondition"
    got = selectable_strategies_for_window(
        "2018-01-01", dependent_fn=lambda: {other}
    )
    assert other not in got, (
        f"the filter ignored the injected dependent set -- it is keyed on a "
        f"literal, not on the predicate: {got}")
    assert "qarp" in got, (
        "and it must NOT still be excluding qarp, which the injected set does "
        "not name -- that would prove a hardcoded exclusion")
