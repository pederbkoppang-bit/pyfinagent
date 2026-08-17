"""phase-86.32 -- a CUMULATIVE per-step attempt budget that no verdict resets.

THE DEFECT THIS REPLACES
------------------------
`run_harness.py` tracks `consecutive_fails`. PASS zeroes it (`:1162`) and so does
CONDITIONAL (`:1177`, under the comment "does not count as a FAIL") -- and
CONDITIONAL is the most common non-terminal verdict this harness produces. So the
sequence `CONDITIONAL, FAIL, CONDITIONAL, ...` tops out at 1 and
`MAX_CONSECUTIVE_FAIL` is unreachable. The counter is also process-local, so it
cannot see a loop spread across sessions -- which is how the Layer-3 per-step loop
actually runs.

Reset-on-success is a HEALTH-CHECK idiom (is the dependency up right now?).
Applying it to WORK ACCOUNTING (how much have we spent on this item?) is the root
cause. (phase-86.71 research-gate correction, 2026-08-17: an earlier revision of
this paragraph claimed "none of them reset on one success" about the breaker
literature -- FALSE for Fowler's canonical circuit breaker, whose half-open
state DOES reset the failure count on a successful probe. The true and narrower
claim: circuit breakers answer a LIVENESS question, where reset-on-success is
correct; WORK-ACCOUNTING bounds -- Google SRE's retry budget as a ratio of
total requests, GEP-3388 retry budgets -- are cumulative over a window, and
that is the family this module belongs to. The 5-attempt / 1.2M-token ceilings
are sourced INTERNALLY from this repo's measured distribution below, and must
never be attributed to Anthropic or any external source.)

WHAT THIS ADDS -- and what it deliberately does NOT touch
--------------------------------------------------------
It adds a ceiling. It changes NO verdict semantics: PASS, CONDITIONAL and FAIL mean
exactly what they meant, and nothing here can turn a FAIL into anything else. A
budget can only stop the loop EARLIER; it can never let a step through that the
Q/A refused. That direction is the whole safety argument and it is asserted by
`test_exhaustion_cannot_auto_pass` and by mutation cell M4.

THE ATTEMPT/OUTCOME DISTINCTION IS THE POINT (criterion 2)
----------------------------------------------------------
A dropped spawn costs full tokens and returns NO verdict. Any counter keyed on the
VERDICT is therefore blind to it -- which is exactly how 86.28 ran 8 attempts while
its counter saw 5. Measured on 2026-08-11, between 8.6% (513 runs, all-time) and
29.2% (24 runs, that day) of Workflow runs drop. Those two figures come from
different windows AND different sources and are NOT reconciled; both are recorded
rather than one being quoted. Either way the blind fraction is large enough that a
verdict-keyed ceiling is not a ceiling.

So: **the budget increments on ATTEMPT, not on OUTCOME.**
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum

# Cumulative attempts per step before the loop must stop and escalate.
# Rationale, measured over 513 runs / 164 steps (phase-86.32 research gate):
# runs-per-step {1:27, 2:48, 3:38, 4:28, 5:13, 6:5, 7:2, 8:1, 9:2}. A ceiling of 5
# leaves 93.9% of historically-completed steps untouched (27+48+38+28+13 = 154 of
# 164) while catching the 10 long tails that hold a disproportionate share of the
# spend. It is a CEILING, not a target.
DEFAULT_MAX_ATTEMPTS = 5

# Token ceiling per step. p50 is 419,739 and the observed max is 1,832,223 (step
# 75.5, 8 runs). 1.2M sits above the p50 by ~3x and below the worst case, so it
# binds only on genuine runaways.
DEFAULT_MAX_TOKENS = 1_200_000


class Outcome(str, Enum):
    """What an attempt produced. NO_VERDICT is a first-class outcome, not an error.

    A dropped or errored spawn is NOT a verdict and must never be treated as one.
    It is still an ATTEMPT and still costs tokens, which is the whole reason this
    enum exists rather than a nullable verdict string.
    """

    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    FAIL = "FAIL"
    NO_VERDICT = "NO_VERDICT"


class Disposition(str, Enum):
    CONTINUE = "CONTINUE"          # budget remains; the loop may run again
    CLOSED_PASS = "CLOSED_PASS"    # a Q/A returned PASS -- the ONLY way to close green
    ESCALATE = "ESCALATE"          # budget exhausted -> operator decides. NEVER auto-pass.


@dataclass
class Attempt:
    n: int
    outcome: Outcome
    tokens: int = 0
    run_id: str = ""
    note: str = ""


@dataclass
class BudgetState:
    """Cumulative, monotonic. NOTHING in this class ever decrements a counter."""

    step_id: str
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_tokens: int = DEFAULT_MAX_TOKENS
    attempts: list[Attempt] = field(default_factory=list)

    # ── monotonic accessors ──────────────────────────────────────────────
    @property
    def attempts_used(self) -> int:
        """EVERY attempt, including those that returned no verdict."""
        return len(self.attempts)

    @property
    def verdicts_seen(self) -> int:
        """Attempts that actually produced a verdict -- always <= attempts_used.

        The gap between these two numbers is what the old counter could not see.
        """
        return sum(1 for a in self.attempts if a.outcome is not Outcome.NO_VERDICT)

    @property
    def dropped(self) -> int:
        return sum(1 for a in self.attempts if a.outcome is Outcome.NO_VERDICT)

    @property
    def tokens_used(self) -> int:
        return sum(a.tokens for a in self.attempts)

    @property
    def exhausted(self) -> bool:
        return (self.attempts_used >= self.max_attempts
                or self.tokens_used >= self.max_tokens)

    def record(self, outcome: Outcome, tokens: int = 0, run_id: str = "",
               note: str = "") -> Attempt:
        a = Attempt(n=self.attempts_used + 1, outcome=outcome, tokens=tokens,
                    run_id=run_id, note=note)
        self.attempts.append(a)
        return a

    # ── the decision ─────────────────────────────────────────────────────
    def disposition(self) -> Disposition:
        """The ONLY place the loop's fate is decided.

        Order matters and is deliberate: PASS is checked FIRST, so a step that
        earns a PASS on its final permitted attempt closes green rather than being
        escalated for running to the wire. Exhaustion is checked second. There is
        no third branch that could turn exhaustion into a pass -- see
        `test_exhaustion_cannot_auto_pass`.
        """
        if any(a.outcome is Outcome.PASS for a in self.attempts):
            return Disposition.CLOSED_PASS
        if self.exhausted:
            return Disposition.ESCALATE
        return Disposition.CONTINUE

    # ── criterion 4: PRODUCT vs EVIDENCE, separably ──────────────────────
    def close_kind(self, product_verified: bool, evidence_complete: bool) -> str:
        """Separate 'the code is right' from 'the paperwork is complete'.

        This exists so a step whose CODE is verified but whose INSTRUMENTATION has
        residuals can close with those residuals queued as their own steps, instead
        of spinning the loop on paperwork.

        IT CANNOT LOWER ANY THRESHOLD. It is reachable only from CLOSED_PASS -- a
        state that requires an actual Q/A PASS. On any other disposition it returns
        the disposition itself. So a FAIL stays a FAIL no matter what flags are
        passed, which is what criterion 4's regression demand ("the 2026-08-10 FAIL
        for a fabricated transcript would STILL be a FAIL") turns on.
        """
        d = self.disposition()
        if d is not Disposition.CLOSED_PASS:
            return d.value
        if product_verified and evidence_complete:
            return "CLOSED_COMPLETE"
        if product_verified and not evidence_complete:
            return "CLOSED_PRODUCT_RESIDUALS_QUEUED"
        return "ESCALATE"

    def escalation_summary(self) -> str:
        """Criterion 3: a WRITTEN summary for the operator on exhaustion.

        Deliberately states what is NOT known. An escalation that reads like a
        success is worse than none.
        """
        if self.disposition() is not Disposition.ESCALATE:
            return ""
        mix: dict[str, int] = {}
        for a in self.attempts:
            mix[a.outcome.value] = mix.get(a.outcome.value, 0) + 1
        lines = [
            f"# BUDGET EXHAUSTED -- step {self.step_id} -- OPERATOR DECISION REQUIRED",
            "",
            f"- attempts used : {self.attempts_used} / {self.max_attempts}",
            f"- tokens used   : {self.tokens_used:,} / {self.max_tokens:,}",
            f"- verdicts seen : {self.verdicts_seen}"
            f"  (so {self.dropped} attempt(s) produced NO verdict and cost tokens anyway)",
            f"- outcome mix   : {mix}",
            "",
            "## THIS IS NOT A PASS AND NOT A FAIL",
            "",
            "The step is NOT verified. No verdict is implied by exhaustion, and none",
            "may be inferred from it. The loop stopped because it hit a cost ceiling,",
            "which says nothing about whether the work is correct.",
            "",
            "## What the operator must decide",
            "",
            "1. RAISE the budget and continue, if the remaining work is bounded and known.",
            "2. PARK the step with a written disposition (this project's existing vocabulary).",
            "3. SPLIT it: close the verified part, queue the residuals as their own steps.",
            "",
            "## Per-attempt record",
            "",
        ]
        for a in self.attempts:
            lines.append(
                f"- attempt {a.n}: {a.outcome.value}"
                + (f"  run={a.run_id}" if a.run_id else "")
                + (f"  tokens={a.tokens:,}" if a.tokens else "")
                + (f"  -- {a.note}" if a.note else "")
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        d = asdict(self)
        d["attempts"] = [{**asdict(a), "outcome": a.outcome.value} for a in self.attempts]
        d["derived"] = {
            "attempts_used": self.attempts_used,
            "verdicts_seen": self.verdicts_seen,
            "dropped": self.dropped,
            "tokens_used": self.tokens_used,
            "exhausted": self.exhausted,
            "disposition": self.disposition().value,
        }
        return json.dumps(d, indent=2)


# ── criterion 5: the 86.28 regression fixture ────────────────────────────
#
# REBUILT FROM THE RECORD after the cycle-1 Q/A FAILED this step (2026-08-11).
#
# THE ORIGINAL FIXTURE WAS WRONG AND THE FAILURE MODE IS WORTH KEEPING.
# It was built by parsing `evaluator_critique_86.28_history.md` in DOCUMENT ORDER,
# scraping `wf_*` ids and verdict headings and pairing them positionally. That file
# contains TWO DIFFERENT POPULATIONS of run id -- Q/A attempts, and the 86.28
# author's own live `research-gate.js` evidence runs -- and a positional parse
# cannot tell them apart. Result: 3 of 8 rows were not attempts at all (one of
# them, wf_23d9ed4b-22c, SUCCEEDED: agentCount 0 / totalTokens 0 / durationMs 5,
# recorded as a drop), 2 outcomes were inverted, and 2 real attempts were missing.
#
# The justification offered for trusting it -- "the 3 no-verdict attempts
# corroborate that step's claim of three rail failures" -- was pure CARDINALITY
# AGREEMENT over a different member set. Symmetric difference: 3 spurious, 2
# omitted, out of 8.
#
# Each row now carries its SOURCE, and `test_fixture_matches_the_recorded_ledger`
# re-derives the sequence from those files rather than asserting properties of this
# constant. A fixture that cannot be checked against the record is not a fixture.
#
# NOTE THE LEDGER ITSELF IS INCOMPLETE, which is darkly apt for this step: the
# `## Verdict ledger` table records 7 attempts and OMITS the cycle-7 drop, which
# lives only in `live_check_86.28.md` §9. A drop went unrecorded in the very ledger
# whose subject is drops going unrecorded.
#
# (cycle, run_id, outcome, source)
FIXTURE_86_28_ROWS: list[tuple[int, str, Outcome, str]] = [
    (1, "wf_10c6cbd2-cad", Outcome.CONDITIONAL, "evaluator_critique_86.28.md::ledger"),
    (2, "wf_d0934c91-70b", Outcome.CONDITIONAL, "evaluator_critique_86.28.md::ledger"),
    (3, "wf_01c83c86-09d", Outcome.NO_VERDICT,  "evaluator_critique_86.28.md::ledger"),
    (3, "wf_e262facc-cdc", Outcome.FAIL,        "evaluator_critique_86.28.md::ledger"),
    (4, "wf_5a217e41-9b9", Outcome.CONDITIONAL, "evaluator_critique_86.28.md::ledger"),
    (5, "wf_344395f1-4ac", Outcome.CONDITIONAL, "evaluator_critique_86.28.md::ledger"),
    (6, "wf_9c55b720-ef3", Outcome.NO_VERDICT,  "evaluator_critique_86.28.md::ledger"),
    (7, "wf_e03ec2d0-c07", Outcome.NO_VERDICT,  "live_check_86.28.md::section-9"),
]

FIXTURE_86_28: list[tuple[str, Outcome]] = [
    (run_id, outcome) for _cycle, run_id, outcome, _src in FIXTURE_86_28_ROWS
]

# Run ids that appear in the 86.28 artifacts but are NOT Q/A attempts. Named so a
# future parse cannot silently re-absorb them: this is the exact set the original
# fixture mistook for attempts, plus the audit and gate runs.
NOT_QA_ATTEMPTS_86_28: set[str] = {
    "wf_23d9ed4b-22c",  # live research-gate.js evidence run (SUCCEEDED)
    "wf_4da39b31-695",  # live research-gate.js evidence run
    "wf_60de95f7-5dc",  # the step's own research GATE run
    "wf_d61fef3b-25c",  # audit run, report-only
}


def legacy_consecutive_fails(seq: list[Outcome]) -> int:
    """The CURRENT rule, reimplemented exactly, so the replay compares like with like.

    Mirrors run_harness.py:1160-1177 -- PASS resets, FAIL increments, and anything
    else (CONDITIONAL) resets. NO_VERDICT never reaches that code at all, since a
    dropped spawn produces no `grades["verdict"]` to branch on.
    """
    c = 0
    for o in seq:
        if o is Outcome.PASS:
            c = 0
        elif o is Outcome.FAIL:
            c += 1
        elif o is Outcome.CONDITIONAL:
            c = 0
        # NO_VERDICT: unreachable in the legacy path -- no verdict to branch on.
    return c


def replay_86_28(max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> dict:
    """Replay the fixture against BOTH rules and report where each terminates."""
    st = BudgetState(step_id="86.28", max_attempts=max_attempts)
    terminated_at = None
    for run_id, outcome in FIXTURE_86_28:
        st.record(outcome, run_id=run_id)
        if terminated_at is None and st.disposition() is not Disposition.CONTINUE:
            terminated_at = st.attempts_used
    seq = [o for _, o in FIXTURE_86_28]
    return {
        "attempts_in_fixture": len(FIXTURE_86_28),
        "new_rule_terminates_at_attempt": terminated_at,
        "new_rule_disposition": st.disposition().value,
        "legacy_consecutive_fails_final": legacy_consecutive_fails(seq),
        "legacy_would_have_terminated": legacy_consecutive_fails(seq) >= 3,
        "verdicts_seen": st.verdicts_seen,
        "dropped": st.dropped,
        "attempts_invisible_to_legacy_counter": st.dropped,
    }


if __name__ == "__main__":
    r = replay_86_28()
    print(json.dumps(r, indent=2))
    print()
    st = BudgetState(step_id="86.28")
    for run_id, outcome in FIXTURE_86_28[:DEFAULT_MAX_ATTEMPTS]:
        st.record(outcome, run_id=run_id)
    print(st.escalation_summary())
