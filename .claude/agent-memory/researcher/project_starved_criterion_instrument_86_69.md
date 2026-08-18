---
name: starved-criterion-instrument-86-69
description: 86.69 cycle 2 -- a starved criterion needs a DENOMINATOR not a rerun; the honest-absence branch fires only when BOTH paths fail so arming produces LITE rows not NULL rows; the unstarving ledger is UNTRACKED
metadata:
  type: project
---

Step **86.69** (empty `0.0`/`HOLD` rows) was PARKED CONDITIONAL after cycle 1 on
"starved" criteria C4/C5. Cycle 2 re-verified it. Three things worth carrying.

**1. A starved criterion is usually a MISSING DENOMINATOR, not a missing rerun.**
The park's reasoning was that the guard's population was empty so "no further
evaluation can manufacture that". True about the OUTCOME, but `0/6` was reported
with no statement of how many times the guarded condition arose -- so a genuinely
clean window and a starved window are indistinguishable. The fix is to report the
count of the guard's TRIGGER beside the outcome. Then "0 occurrences" is a
measurement, not an absence of one. This generalises to every guard-arming step.

**Why:** the same step had already been burned by a bare share -- post-arm `0/6`
looked like recovery until pre-arm 2026-08-10 and 2026-08-14 were found to be
`0/6` too.

**How to apply:** when a contract arms a guard, make one criterion report the
guard's denominator from an existing population. For this guard the trigger is
`_parse_json_with_fallback` returning `None` for `agent="Synthesis-Final"`, whose
legacy warning `"{agent} returned invalid JSON"` has been in `backend.log` all
along -- a denominator is obtainable with no new code.

**2. The honest-absence branch is the TAIL, not the main line.** In
`autonomous_loop.py` the `_degraded`/NULL dict is returned only when BOTH the full
AND lite paths fail. The 211-row defect is full-fails-lite-succeeds, so arming the
integrity flag converts the defect into a **LITE row, not a NULL row**. A criterion
written as "show NULL rows appear" starves a second time. Verify which branch a
fix actually reaches before writing the criterion. Related: [[dead-sell-rule-86-58]].

**3. The instrument that would unstarve it was UNTRACKED, not merely uncommitted.**
`backend/agents/parse_failure_ledger.py` existed in the working tree with wiring in
`orchestrator.py`, but `git status` showed `??` and `git show HEAD:...orchestrator.py
| grep -c record_parse_failure` returned **0**. One notch worse than the usual
"committed is not in force". **Check `git show HEAD:<file>` before depending on a
symbol you just read from the working tree** -- reading it proves it exists on
disk, not that it is in the repo or in the running process.

Cross-refs: `handoff/current/escalation_86.69_starved_measurement.md`,
`handoff/current/research_brief_86.69.md` §CYCLE-2 C/F.
