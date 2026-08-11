"""phase-86.32 -- the cumulative attempt budget, and the things it must never do.

The budget's job is to make the evaluate loop STOP. The risk it introduces is the
opposite of the one it fixes: a ceiling that quietly lets work through would be far
worse than a loop that runs long. So the tests below spend most of their effort on
the direction of failure, not on the happy path.

Every assertion here is paired in `scripts/qa/mutation_matrix_86_32.py`; a cell
that reverts the cumulative counter to a consecutive one, or that makes a dropped
spawn invisible, or that lets exhaustion pass, must turn a NAMED test red.
"""

from __future__ import annotations

import itertools

from scripts.harness.attempt_budget import (
    DEFAULT_MAX_ATTEMPTS,
    FIXTURE_86_28,
    BudgetState,
    Disposition,
    Outcome,
    legacy_consecutive_fails,
    replay_86_28,
)


# ── the core fix: cumulative, never reset ────────────────────────────────


def test_no_outcome_ever_decrements_the_attempt_count():
    """The defect was a counter that a verdict could reset. Prove none can.

    Exhaustive over every ordering of length 3 drawn from all four outcomes: 64
    sequences, not a hand-picked few. A single reset anywhere would show up as a
    non-monotonic count.
    """
    for seq in itertools.product(list(Outcome), repeat=3):
        st = BudgetState(step_id="t", max_attempts=99)
        seen = []
        for o in seq:
            st.record(o)
            seen.append(st.attempts_used)
        assert seen == [1, 2, 3], f"attempt count was not monotonic for {seq}: {seen}"


def test_conditional_does_not_reset_the_budget():
    """CONDITIONAL is the reset that actually bites in production (:1177)."""
    st = BudgetState(step_id="t", max_attempts=99)
    for o in (Outcome.FAIL, Outcome.CONDITIONAL, Outcome.FAIL, Outcome.CONDITIONAL):
        st.record(o)
    assert st.attempts_used == 4
    # and the legacy rule on the same sequence:
    assert legacy_consecutive_fails(
        [Outcome.FAIL, Outcome.CONDITIONAL, Outcome.FAIL, Outcome.CONDITIONAL]
    ) == 0, "the legacy counter should end at 0 here -- that IS the defect"


# ── criterion 2: dropped spawns are attempts ─────────────────────────────


def test_dropped_spawns_count_against_the_budget():
    st = BudgetState(step_id="t", max_attempts=3)
    st.record(Outcome.NO_VERDICT)
    st.record(Outcome.NO_VERDICT)
    assert st.disposition() is Disposition.CONTINUE
    st.record(Outcome.NO_VERDICT)
    assert st.exhausted, "three dropped spawns must exhaust a 3-attempt budget"
    assert st.disposition() is Disposition.ESCALATE


def test_the_gap_between_attempts_and_verdicts_is_visible():
    """A verdict-keyed counter cannot see drops. Assert the budget reports both."""
    st = BudgetState(step_id="t", max_attempts=99)
    st.record(Outcome.CONDITIONAL)
    st.record(Outcome.NO_VERDICT)
    st.record(Outcome.NO_VERDICT)
    assert st.attempts_used == 3
    assert st.verdicts_seen == 1
    assert st.dropped == 2
    assert st.attempts_used > st.verdicts_seen, (
        "if these are ever equal the budget has become verdict-keyed, which is "
        "exactly the blindness this step exists to remove"
    )


def test_token_ceiling_binds_independently_of_attempt_count():
    st = BudgetState(step_id="t", max_attempts=99, max_tokens=1000)
    st.record(Outcome.CONDITIONAL, tokens=999)
    assert not st.exhausted
    st.record(Outcome.NO_VERDICT, tokens=1)
    assert st.exhausted, "a dropped spawn's tokens must count toward the ceiling"


# ── criterion 3: exhaustion escalates and CANNOT auto-pass ───────────────


def test_exhaustion_cannot_auto_pass():
    """The load-bearing safety property. Exhaustive, not illustrative.

    Over every sequence of non-PASS outcomes up to the budget length, the
    disposition must never be CLOSED_PASS. If a ceiling can ever manufacture a
    pass, the ceiling is more dangerous than the unbounded loop it replaces.
    """
    non_pass = [Outcome.CONDITIONAL, Outcome.FAIL, Outcome.NO_VERDICT]
    checked = 0
    for n in range(1, DEFAULT_MAX_ATTEMPTS + 2):
        for seq in itertools.product(non_pass, repeat=n):
            st = BudgetState(step_id="t")
            for o in seq:
                st.record(o)
            checked += 1
            assert st.disposition() is not Disposition.CLOSED_PASS, (
                f"exhaustion produced a PASS for {seq} -- forbidden"
            )
    assert checked > 300, f"vacuity guard: only {checked} sequences examined"


def test_escalation_summary_is_written_and_says_it_is_not_a_pass():
    st = BudgetState(step_id="86.28")
    for _ in range(DEFAULT_MAX_ATTEMPTS):
        st.record(Outcome.CONDITIONAL)
    s = st.escalation_summary()
    assert s, "exhaustion must produce a written summary (criterion 3)"
    assert "NOT A PASS" in s and "NOT A FAIL" in s
    assert "OPERATOR DECISION REQUIRED" in s
    assert "attempts used" in s and "tokens used" in s


def test_no_summary_when_not_exhausted():
    """Positive control for the assertion above: the summary must be conditional.

    Without this, `escalation_summary()` could return its text unconditionally and
    every assertion above would still pass.
    """
    st = BudgetState(step_id="t")
    st.record(Outcome.CONDITIONAL)
    assert st.escalation_summary() == ""


# ── criterion 4: PRODUCT vs EVIDENCE, without lowering anything ──────────


def test_a_fail_stays_a_fail_under_every_flag_combination():
    """Criterion 4's explicit regression demand.

    The 2026-08-10 FAIL was for a fabricated transcript -- an EVIDENCE defect. The
    new PRODUCT/EVIDENCE split must not become a door that lets such a step close
    by declaring the product fine. Nothing reachable from a non-PASS history may
    return a close.
    """
    for product, evidence in itertools.product([True, False], repeat=2):
        st = BudgetState(step_id="86.28", max_attempts=99)
        st.record(Outcome.FAIL, note="fabricated transcript (2026-08-10)")
        got = st.close_kind(product_verified=product, evidence_complete=evidence)
        assert got != "CLOSED_COMPLETE", (
            f"a FAIL closed as complete with product={product} evidence={evidence}"
        )
        assert got != "CLOSED_PRODUCT_RESIDUALS_QUEUED", (
            f"a FAIL took the residuals door with product={product} evidence={evidence}"
        )
        assert got == Disposition.CONTINUE.value


def test_residuals_door_requires_an_actual_pass():
    st = BudgetState(step_id="t", max_attempts=99)
    st.record(Outcome.PASS)
    assert st.close_kind(True, False) == "CLOSED_PRODUCT_RESIDUALS_QUEUED"
    assert st.close_kind(True, True) == "CLOSED_COMPLETE"
    # product NOT verified, even with a PASS, must not close
    assert st.close_kind(False, True) == "ESCALATE"


def test_pass_on_the_final_permitted_attempt_still_closes_green():
    """A step that earns a PASS at the wire is closed, not escalated for lateness."""
    st = BudgetState(step_id="t")
    for _ in range(DEFAULT_MAX_ATTEMPTS - 1):
        st.record(Outcome.CONDITIONAL)
    st.record(Outcome.PASS)
    assert st.exhausted, "precondition: the budget IS exhausted here"
    assert st.disposition() is Disposition.CLOSED_PASS, (
        "PASS must be checked before exhaustion, or real work gets thrown away"
    )


# ── criterion 5: the 86.28 replay ────────────────────────────────────────


def test_86_28_fixture_shape_matches_the_recorded_history():
    """Precondition: if the fixture drifts, the replay proves nothing."""
    assert len(FIXTURE_86_28) == 8
    outcomes = [o for _, o in FIXTURE_86_28]
    assert sum(1 for o in outcomes if o is Outcome.NO_VERDICT) == 3, (
        "the recorded history has three rail failures; the fixture must match"
    )
    assert sum(1 for o in outcomes if o is Outcome.CONDITIONAL) == 4
    assert sum(1 for o in outcomes if o is Outcome.FAIL) == 1
    assert len({r for r, _ in FIXTURE_86_28}) == 8, "run ids must be distinct"


def test_86_28_replay_terminates_where_the_legacy_rule_never_would():
    r = replay_86_28()
    assert r["new_rule_terminates_at_attempt"] == 5
    assert r["new_rule_disposition"] == "ESCALATE"
    # The legacy counter ends at zero: the CONDITIONAL at attempt 7 wipes the FAIL
    # at attempt 6. This is the defect, asserted rather than described.
    assert r["legacy_consecutive_fails_final"] == 0
    assert r["legacy_would_have_terminated"] is False
    assert r["attempts_invisible_to_legacy_counter"] == 3
