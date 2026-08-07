---
name: decision-input-integrity-61-2
description: Step 61.2 triage -- the step was ALREADY BUILT and dark, has 6 immutable criteria (not zero), 72.0.1's premise is dead, and a later step's live-state gate turned its tests red
metadata:
  type: project
---

Triage of masterplan step 61.2 (decision-input integrity), 2026-08-07.

**Before triaging any "old P0" step, check whether it was already built.**
61.2 read like an un-started 2-month-old defect list. It was fully
implemented on 2026-07-08 in commit `6186784c` (20 source files + a 459-line
test module) and is `status: pending` only because Q/A returned CONDITIONAL
on *live* evidence that could not exist yet. `git log --grep="<step-id>"`
answers this in one call and reframes the whole contract from "rescope or
drop" to "what stands between the dark build and a close".

**Why:** the drain-goal framing ("2 months old, rescope-or-drop where
stale") primes you to look for staleness, not for completion. Two of the six
sub-items were already LIVE in production, and the settings.py field
descriptions literally cite "phase-61.2 (criterion 2)" -- the evidence was
one grep away.

**How to apply:** on any step whose name reads as a defect list, run
`git log --oneline --all --grep="<step-id>"` and
`grep -rn "phase-<step-id>" backend/` BEFORE the liveness audit. Field/flag
descriptions in `backend/config/settings.py` are the highest-signal place a
prior build leaves its fingerprints.

**A step's `verification.success_criteria` can be non-empty even when the
caller says it is empty.** Always dump the masterplan entry yourself. 61.2
carries six immutable criteria, one per sub-item. Inventing "proposed
criteria" for it would have been an amendment of immutable criteria -- a
protocol breach dressed as diligence. Related: [[feedback_immutable_criteria_must_be_green_able]].

**Flag-dark builds need a THREE-way liveness verdict, not two.** Each
sub-item is one of: ungated-and-live, built-but-flag-dark (defect still
firing in prod), or already-fixed-and-live-proven. Only BQ separates the
last two. Here: `null_name` went 5/day -> exactly 0/day from 2026-07-09
onward, proving an ungated fix deployed and held for a month.

**A later step can silently turn an older step's tests red.** phase-36.13
(`3227347a`, 2026-07-26) added a kill-switch gate to `execute_buy` that
falls back to the module singleton, which replays the REAL on-disk audit
log. Any test calling `execute_buy` without injecting `_injected_ks_state`
now passes or fails depending on whether the operator's book happens to be
paused. Two 61.2 tests written 2026-07-08 flipped red this way.
**Why it matters:** the failure looks like a 61.2 regression and is not.
**How to apply:** when an old step's tests fail, `git log -S` the assertion's
*production* gate, not the test -- the breaking change is usually in a
newer, unrelated step. Related: [[feedback_a_green_suite_can_be_blind]].

**Queued "sharper" steps go stale too -- verify their premise before
deferring work to them.** 72.0.1 exists to fix "meta_scorer.py:220-225
constructs ClaudeClient with anthropic_api_key directly". phase-78.1 already
rewired that call through `make_client`; there is no ClaudeClient
construction left in the file. Deferring 61.2's meta-scorer leg to 72.0.1
would have deferred it to a step whose premise no longer holds.

**A masterplan `audit_note` is evidence, not truth.** The phase-76 note on
61.2 asserted the 0.00/HOLD rows would survive the fix because the manual
`save_report` sites lack a guard. Measured: all 142 rows carry
`recommendation='HOLD'` upper-case, which ONLY `autonomous_loop.py:1942`
emits (the manual sites emit `'N/A'`). The manual-path gap is real but
LATENT -- it is not the writer of the observed population. Related:
[[feedback_measure_dont_assert_claims]].
