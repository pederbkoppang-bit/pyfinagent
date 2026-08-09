# Contract -- phase-86.21

**Step:** 86.21 (P2) -- the 3rd-CONDITIONAL counter is structurally blind to any
step still in flight, and fails open silently.
**Date:** 2026-08-10. **Cycle:** 199.
**Research gate:** PASSED -- `handoff/current/research_brief_86.21.md`
(run `wf_f916b683-d59`: 7 sources read in full >= floor 5, 26 URLs >= floor 10,
recency scan performed, 8 internal files inspected; 24,895 chars independently
re-read on disk, all 7 claimed URLs verified present).

---

## 1. Research-gate summary

**The premise is confirmed and already git-confessed.** Commit `a1b92d14`
backfilled five 36.17 cycles in ONE commit, its own message stating that
`harness_log.md` had zero rows across five Q/A cycles and that "the counter was
blind to the entire step".

**A FRESHER reproduction happened tonight and supersedes the step text's
example.** The step text builds its reproduction on 36.17; that is no longer
reproducible, because 36.17 has since closed and now carries six rows. Across
**phase-86.20's three Q/A cycles** the prescribed grep returned **0 every
time**, and the cycle-2 and cycle-3 Q/As each said so in their own verdicts --
each having been hand-fed its verdict history by Main, i.e. **by the party the
rule constrains.** The executor must build the reproduction on a genuinely
in-flight step, not on 36.17.

**THREE findings the research produced that the step text does not mention, and
the first one changes what "fixing this" even means:**

1. **The rule exists in THREE files with TWO DIFFERENT PREDICATES.**
   `CLAUDE.md:358-359` says *"3+ consecutive CONDITIONAL verdicts without an
   intervening PASS or FAIL"* -- consecutive, with reset. `qa.md:512-515` says
   *"already 2+ `result=CONDITIONAL` entries for this step-id"* -- a cumulative
   grep, while calling it "consecutive". **They give OPPOSITE answers on 36.17
   cycle 194.** Fixing the SOURCE alone leaves the escalation ambiguous, so this
   step must also reconcile the predicate or it fixes nothing decidable.
2. **The prescribed source is ~48% unparseable.** Re-derived independently:
   **574 of 1189** cycle headers carry no `phase=` at all (48.3%). And there are
   **12+ distinct `result=` tokens** -- including `PASS_WITH_FINDINGS` and
   `PASS_AFTER_RETRY` -- so any reset predicate written as `== "PASS"` never
   fires on them, silently extending a "consecutive" run across a real pass.
3. **`evaluator_critique_<id>.md` is NOT a drop-in substitute.** 17+ filename
   shapes exist, and one-file-per-cycle is not an invariant: 36.17 ran six
   cycles and left ONE file. Measured separately tonight: a parser keyed on
   `## Cycle N verdict` returns 3 for 86.20 and 2 for 86.17 but **0 for 36.17**,
   whose file uses heading depth 1 -- a SILENT ZERO over five real verdicts.
   That is the defect recurring inside its own proposed fix.

**Who can even write a ledger:** the Q/A has no `Write` tool and the Workflow
runtime has no filesystem access. **Main -- or a hook -- is the only possible
writer**, which is the independence problem in its sharpest form and is why
criterion 4 demands an explicit answer rather than a preference.

**Literature:** SLSA L2+ (the trusted control plane writes provenance, never the
worker); Kubernetes `.status.failed` + `podFailurePolicy` (the controller owns
the count, and what-counts is an explicit policy); RFC 9413 §5.1 (fail loud
rather than tolerate); Fowler on event sourcing (post-hoc reconstruction
dissolves the log's value -- which is exactly what `a1b92d14` did).

## 2. Hypothesis

The counter fails open because its only source is written at step CLOSE. Giving
it a source that accumulates per CYCLE, a parser that is tolerant of the format
drift already present in the corpus, and a return that DISTINGUISHES "zero
verdicts" from "could not parse", makes the rule decidable while it is still
in flight -- and stating ONE predicate makes the answer unambiguous.

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. "The blindness is REPRODUCED FIRST and recorded verbatim: for a step with N>1 recorded verdicts but status still pending, show the log-grep the rule prescribes returning zero rows"
2. "A counting source is chosen that does NOT require writing harness_log rows mid-step, and the reason is stated; LOG-is-last ordering is preserved"
3. "The counter returns the CORRECT count for a step mid-flight, proven against 36.17's real five-verdict history (CONDITIONAL, FAIL, FAIL, CONDITIONAL, CONDITIONAL) -- which also exercises the reset-on-FAIL path"
4. "The independence question is answered explicitly: state whether a Main-supplied count is advisory or authoritative, given Main is the audited party"
5. "Fail-safe direction is asserted and TESTED: if the counting source is missing or unreadable, say plainly whether it fails open or closed and why that is right here"
6. "MUTATION-TEST the counter: corrupt or empty the source and assert it NOTICES rather than silently reporting zero -- silently reporting zero is the defect being fixed"

**Verification command (immutable):**
`bash -c 'grep -c "^## Cycle" handoff/harness_log.md && ls handoff/current/evaluator_critique_*.md | head -3'`

**live_check (immutable):** "Verbatim counter output against 36.17's real verdict history showing 5 verdicts and the correct consecutive-CONDITIONAL count, side by side with the verbatim zero-row output of the log-grep the rule currently prescribes."

## 4. Design decisions this step must make explicitly

- **Source.** A per-step append-only verdict ledger, written when Main
  transcribes each verdict -- i.e. per CYCLE, not at close. It must NOT be
  `harness_log.md`: LOG-is-last is deliberate, and the step text forbids
  writing log rows mid-step.
- **Predicate.** ONE, stated: consecutive-with-reset per `CLAUDE.md`, with the
  reset firing on any terminal token that is not a CONDITIONAL (so
  `PASS_WITH_FINDINGS` resets, which a `== "PASS"` test would miss).
  `qa.md`'s cumulative wording is corrected to match, or the ambiguity survives.
- **Fail direction.** Criterion 5 requires this to be asserted AND TESTED.
  A missing ledger for a step that has never been graded is legitimately zero;
  an UNPARSEABLE ledger is not. The two must return different things, and the
  unparseable case must be loud -- RFC 9413 §5.1.
- **Independence.** Criterion 4 requires an explicit answer. Main writes the
  ledger, so a Main-supplied count cannot be authoritative on its own; what
  makes it auditable is that the ledger is append-only and git-committed, so a
  retro-edit is visible in history rather than invisible in prose.

## 5. Plan

1. **[done]** Research gate PASSED.
2. **[this file]** Contract, BEFORE any code.
3. **Reproduce FIRST** (criterion 1) on a genuinely in-flight step -- NOT 36.17.
4. Build the counter as a re-runnable `scripts/qa/` checker.
5. Prove the correct count against 36.17's real five-verdict history including
   the reset-on-FAIL path (criterion 3).
6. Answer independence (criterion 4) and fail-direction (criterion 5) in the
   artifacts, with the fail direction TESTED.
7. Mutation-test (criterion 6): corrupt/empty the source and assert it NOTICES
   rather than silently reporting zero.
8. Q/A; transcribe verbatim; log; flip.

## 6. Traps (measured)

- **Do not build the reproduction on 36.17** -- it has closed and now carries
  six rows. Use a genuinely in-flight step.
- **A parser keyed on one heading depth returns a SILENT ZERO.** Measured on
  36.17's critique file.
- **`== "PASS"` is not a reset test** -- 12+ result tokens exist.
- **Do not write harness_log rows mid-step.** LOG-is-last is deliberate.
- **Do not claim the counter is independent** if Main writes its input. Say what
  it actually is.

## 7. References

- `handoff/current/research_brief_86.21.md` (7 read in full, 26 URLs).
- RFC 9413 -- https://www.rfc-editor.org/rfc/rfc9413.html
- SLSA v1.0 threats -- https://slsa.dev/spec/v1.0/threats
- Kubernetes Job semantics -- https://kubernetes.io/docs/concepts/workloads/controllers/job/
- Fowler, Event Sourcing -- https://martinfowler.com/eaaDev/EventSourcing.html
- Fail-open vs fail-closed -- https://authzed.com/blog/fail-open
- Internal: `CLAUDE.md:358`, `.claude/agents/qa.md:512`, `handoff/harness_log.md`.
