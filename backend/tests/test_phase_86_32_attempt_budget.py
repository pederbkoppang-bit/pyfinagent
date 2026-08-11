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
import pathlib
import re

from scripts.harness.attempt_budget import (
    DEFAULT_MAX_ATTEMPTS,
    FIXTURE_86_28,
    FIXTURE_86_28_ROWS,
    NOT_QA_ATTEMPTS_86_28,
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


def _parse_ledger_from_record() -> list[tuple[str, Outcome]]:
    """Re-derive the 86.28 attempt series FROM THE RECORD ON DISK.

    This is the whole point of the rewrite. The previous version of this guard
    asserted `len == 8`, `3 NO_VERDICT`, `4 CONDITIONAL`, `1 FAIL` and `8 distinct
    ids` -- every one a property of the fixture CONSTANT. It never opened a file,
    so it passed identically against the correct sequence and against a fixture
    with 3 non-attempts, 2 inverted outcomes and 2 omissions. The cycle-1 Q/A ran
    its exact body against both and got PASS/PASS.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    ledger = (root / "handoff" / "current" / "evaluator_critique_86.28.md").read_text(
        errors="replace"
    )
    rows: list[tuple[str, Outcome]] = []
    # | # | cycle | `wf_...` | VERDICT | headline |
    row_re = re.compile(
        r"^\|\s*[\d-]+\s*\|\s*(\d+)\s*\|\s*`(wf_[0-9a-f]{8}-[0-9a-z]{3})`\s*\|\s*"
        r"\**([A-Z]+)\**",
        re.M,
    )
    for m in row_re.finditer(ledger):
        raw = m.group(3).upper()
        rows.append((m.group(2), Outcome.NO_VERDICT if raw == "DROPPED" else Outcome(raw)))
    return rows


def test_fixture_matches_the_recorded_ledger():
    """The guard that the cycle-1 FAIL was about. It READS the record.

    Compared by SYMMETRIC DIFFERENCE over (run_id, outcome) pairs, never by count:
    two sequences with equal cardinality can cover different members, which is
    exactly how the original fixture passed its own check.
    """
    recorded = _parse_ledger_from_record()
    assert len(recorded) >= 7, (
        f"parsed only {len(recorded)} ledger rows -- the parser is broken, and a "
        "broken parser must fail loudly rather than silently agreeing"
    )
    fixture_prefix = FIXTURE_86_28[: len(recorded)]
    missing = set(recorded) - set(fixture_prefix)
    spurious = set(fixture_prefix) - set(recorded)
    assert not missing and not spurious, (
        f"fixture does not match the ledger.\n  missing from fixture: {sorted(missing)}"
        f"\n  spurious in fixture: {sorted(spurious)}"
    )
    assert fixture_prefix == recorded, (
        f"same members, WRONG ORDER.\n  ledger : {recorded}\n  fixture: {fixture_prefix}"
    )


def test_the_eighth_attempt_is_documented_where_the_fixture_says_it_is():
    """The ledger table omits the cycle-7 drop; it lives in live_check §9.

    Asserted against the file rather than trusted, because the fixture's own
    provenance string is just as capable of being wrong as its outcome was.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    lc = (root / "handoff" / "current" / "live_check_86.28.md").read_text(errors="replace")
    cycle, run_id, outcome, source = FIXTURE_86_28_ROWS[-1]
    assert cycle == 7 and outcome is Outcome.NO_VERDICT
    assert run_id in lc, f"{run_id} is not in live_check_86.28.md -- provenance is false"
    assert "dropped without a verdict" in lc, "the drop is not recorded as a drop"
    assert "live_check_86.28.md" in source


def test_no_non_attempt_run_id_leaked_into_the_fixture():
    """The original defect, pinned so it cannot recur.

    Three research-gate/audit runs were mistaken for Q/A attempts by a positional
    parse of a file containing two populations of run id.
    """
    ids = {r for r, _ in FIXTURE_86_28}
    leaked = ids & NOT_QA_ATTEMPTS_86_28
    assert not leaked, f"non-attempt run ids in the fixture: {sorted(leaked)}"
    assert len(ids) == len(FIXTURE_86_28), "run ids must be distinct"


def test_86_28_replay_terminates_where_the_legacy_rule_never_would():
    r = replay_86_28()
    assert r["new_rule_terminates_at_attempt"] == 5
    assert r["new_rule_disposition"] == "ESCALATE"
    # The legacy counter ends at zero: the CONDITIONAL at attempt 7 wipes the FAIL
    # at attempt 6. This is the defect, asserted rather than described.
    assert r["legacy_consecutive_fails_final"] == 0
    assert r["legacy_would_have_terminated"] is False
    assert r["attempts_invisible_to_legacy_counter"] == 3
