# Day report — overnight drain, 2026-08-16 20:47 → 2026-08-17 00:53 CEST

## The three answers you asked for, first

1. **Did the preflight pass?** **YES.** All four gates green, `/api/health` 200,
   no merge conflicts, and the known-red `verify_workflow_args_boundary.mjs`
   confirmed at exactly the documented `84 passed / 3 failed`.
2. **Did the circuit breaker trip?** **YES**, at 00:50, after 86.97 and 86.94
   both parked at the 3-attempt cap. Step work stopped; the remaining time went
   to the §4 measurement, and **86.98 was not implemented**.
3. **How many steps reached PASS?** **One.** 86.92.

Measured, not estimated: **11 workflow runs, 2,441,303 tokens** of the 3,000,000
ceiling (558,697 unused), over **4.11 h** since the preflight stamp. No metered
spend — everything ran on the Max rail.

---

## What closed

### 86.92 — a gate that had been DEAD for 6d12h — **PASS**, pushed

`verify_workflow_args_boundary.mjs` is the only checker driving the args boundary
of **both** Layer-3 workflow scripts, and it had been failing for a reason nobody
acted on. `84 passed / 3 failed` → **`96 passed / 0 failed`**.

**The filed cause was wrong, and proving that was the step.** The masterplan, the
step title and the night goal all blamed a stale on-disk brief. `enforceGate` is
pure — it never opens that file. Control: pointing `brief_path` at a *nonexistent*
file produces byte-identical violations. The real stale fixture was the checker's
own hand-written `verification` literal, supplying 4 of the 9 fields the schema
requires. Bisected in real worktrees: green at `089726f9` (08-10 08:27), red at
`cad38647` (**phase-86.6**, 08:51) — not phase-86.37 as filed.

**Worse than a red gate:** the rot had made mutation cell `[4]
drop-blind-violation` **non-discriminating** — false with the guard present *and*
absent. It had silently stopped being a mutation test.

Side effect worth having: **86.23 is unblocked** (its immutable command went
exit 1 → 0).

---

## What parked, and why it is not a shrug

| step | verdicts | where it stands |
|---|---|---|
| **86.97** | C, C, **FAIL** | Criteria 1,2,3,4,6,7 MET at cycle 3. Failed on criterion 5 and one claim that did not reproduce. |
| **86.94** | F, F, **CONDITIONAL** | **All 7 criteria MET** on their literal wording at cycle 3; capped on evidence integrity. Guard ships green at 45/0. |

Both carry a named, actionable diagnosis in `.claude/masterplan.json` notes and a
row in `handoff/current/night_diagnostics.md`. Neither is a mystery.

**Shipped and green even from the parked steps:**

- `verify_decision_log_86_97.py` (NEW, 35 assertions) — kills a mutant that was
  **invisible** to the 86.91 checker: deleting the production call left the
  extracted source *byte-identical*, so no assertion in that file could ever have
  caught it.
- `verify_no_sliding_windows_86_94.py` (NEW, 45 assertions).
- `replay_changelog_rule_86_68.py` — **a real production fix**: phase-86.91's
  "both ends pinned" was still **TZ-local** (707 under Oslo/UTC/NY, **787** under
  Seoul — an 80-commit spread decided by `$TZ`). `CORPUS_SINCE` now ends in `Z`;
  the published figures are unchanged and are now regenerable off this laptop.

**Filed rather than left as prose**, each verified to reproduce from disk:
**86.101**, **86.102**, **86.103**.

---

## The honest read on why the night went this way

The breaker's premise is that two parks mean *the harness* is the blocker. **The
data says otherwise, and that is the most useful thing I can hand you.**

In **7 of 7** capping cycles the finding was a **real defect in my own work**, and
I reproduced every one myself before accepting it — twice my re-measurement showed
the evaluator had *understated* the problem. There was no rubber-stamp
CONDITIONAL on met criteria, which is what the day session had seen.

The recurring shape is one class, and it is embarrassing precisely because it is
consistent:

> **I write a guard, and the guard carries the very defect it was built to catch.**

- 86.92: a control for *"prove the stripper is live"* that could not fail.
- 86.97: a buildability oracle (`bash -n`) blind to the only failures its mutants
  could have, because both mutants live inside a quoted heredoc.
- 86.94: a fail-**open** `continue` inside the module whose thesis is fail-closed;
  a scan that matched its own documentation; a checker that flagged itself the
  moment it was committed; a criterion-4 predicate satisfiable by vocabulary; and
  a *correction that accompanied instead of replacing* — inside the step whose
  criterion 5 is exactly that rule.

That is an author problem, not an evaluator problem. The rails worked: they
stopped two steps that would otherwise have eaten the night, each with a
diagnosis you can act on in one short cycle.

---

## The 86.98 input you asked for (§4)

`handoff/current/verdict_population_86_98_input.md`. **86.98 was not
implemented — this is measurement only.**

Across **2034 run records / 377 recovered verdicts / 44 sessions**, the day
session's "8 of 15 (53%)" **generalises**: **106 of 186 CONDITIONALs (57.0%)**
assert every immutable criterion was MET.

The split that actually decides the policy:

```
of those 106 --  EMPTY violated_criteria :  0
                 cite a NUMBERED criterion: 13
                 cite ONLY quality tags   : 93     (illusory-guard 20, Contradiction 19, ...)
```

**Zero capped on nothing.** So 50% of all CONDITIONALs met every criterion and
capped on findings no criterion names — but every one named a real finding. That
makes 86.98 a **policy** question (should a criteria-complete step close over
quality tags?), not a bug. Three options are laid out there; none is chosen.

The classifier was built from 231 harvested `MET`-sentences rather than my own
phrasings, excludes partials, and is positive-controlled: 87.0% of PASS verdicts
classify as all-met, which is what a working detector must do.

---

## Claims I could not verify, stated plainly

- **Three corrections to 86.94 were made AFTER its cycle-3 verdict and were
  therefore never re-graded**: the stale "37 files" (instrument says 282), the
  55-vs-49 prose/instrument split, and the retraction of my "found a live site"
  claim. I re-measured that last one myself — reverting only the `WINDOW_RE`
  widening leaves the enumeration byte-identical, so it found **zero**. All three
  are disclosed in the masterplan notes.
- **86.94's remaining three findings are unfixed and unverified**:
  `quoted_as_evidence` is only isinstance-checked (a *wrong* bool stays green);
  the `<unparsed>` fail-closed branch has no mutation cell; the argv cells may be
  credited to the wrong leg.
- **86.97's two blockers are unfixed**: the `:214` `_flip_magnitude()` mutant
  survives and writes a spurious `bump=minor reason=unrecorded`, and
  `live_check_86.91.md:104` still carries an unbounded criterion-4 claim.
- **The immutable commands for 86.94 and 86.97 cannot fail on the defects those
  steps address.** Both are parse/green-checks that would have stayed green
  throughout. Disclosed in each contract; the real evidence is in the live_checks.
- `evaluator_critique_86.91.md:161/:263` still carry the un-TZ-qualified figure.
  **Deliberate**: editing an evaluator's returned verdict would falsify the
  record. The cycle-3 evaluator accepted that reasoning.

---

## Repo state

`origin == HEAD` confirmed after every push. No step was flipped without a PASS.
No gate was loosened to get green. `qa.md`, `qa-verdict.js` and
`research-gate.js` were never touched (rail R5), verified by `git diff --stat`.
Every commit used explicit pathspecs — a peer session's uncommitted
`backend/api/sovereign_api.py` and `frontend/src/*` edits (dated 08-14) were
never swept in, which the cycle-1 86.94 evaluator independently confirmed.
