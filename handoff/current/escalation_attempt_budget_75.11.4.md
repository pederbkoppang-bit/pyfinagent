# BUDGET EXHAUSTED -- step 75.11.4 -- OPERATOR DECISION REQUIRED

- attempts used : 5 / 5
- tokens used   : 0 / 1,200,000
- verdicts seen : 3  (so 2 attempt(s) produced NO verdict and cost tokens anyway)
- outcome mix   : {'FAIL': 2, 'NO_VERDICT': 2, 'CONDITIONAL': 1}

## THIS IS NOT A PASS AND NOT A FAIL

The step is NOT verified. No verdict is implied by exhaustion, and none
may be inferred from it. The loop stopped because it hit a cost ceiling,
which says nothing about whether the work is correct.

## What the operator must decide

1. RAISE the budget and continue, if the remaining work is bounded and known.
2. PARK the step with a written disposition (this project's existing vocabulary).
3. SPLIT it: close the verified part, queue the residuals as their own steps.

## Per-attempt record

- attempt 1: FAIL
- attempt 2: FAIL
- attempt 3: NO_VERDICT
- attempt 4: CONDITIONAL
- attempt 5: NO_VERDICT
## How to proceed (operator)

A further attempt requires an AUDITED extension row:

    python3 scripts/harness/attempt_gate.py --operator-extend 75.11.4 --by 1 --reason "<why another attempt is warranted>"

The denial itself is NOT a verdict: the step remains exactly as the
last Q/A left it.

*(written 2026-08-17T20:04:01Z by attempt_gate.py at the deny)*

---

# Main's addendum (2026-08-17T20:05Z) -- the substantive state

Written by Main after the deny. **Nothing here is a verdict and nothing here
flips the step.** It exists so the operator's choice above is an informed one.

## The last COMPLETED evaluation said all 13 criteria are MET

Attempt 4 (`wf_51313030-ddd`, CONDITIONAL) opens with the evaluator's own
words:

> "All 13 immutable criteria are MET and I made at least one guard FAIL myself
> for every one of them ... THE SHIPPED PRODUCT IS CORRECT; every finding is an
> evidence or adjacent-coverage defect."

It independently reproduced the mutation matrix cell-for-cell against a green
null-mutant control, matched all three sha256 baselines, and killed three
further mutants of its own that this step's harness cannot even reach.

## Every finding it raised has since been CLOSED (attempt 5 was denied, so
## these are unevaluated by a Q/A -- that is exactly what the operator is
## being asked to weigh)

| # | finding | severity | disposition |
|---|---|---|---|
| D1 | `_move` mkdir'd BEFORE the dry-run return, so a bare run wrote to disk | **PRODUCT** | FIXED; the 3 empty dirs it left (`phase-80.5/-81.1/-82.23`) removed; guarded by `test_c6_a_dry_run_creates_no_directories`; mutant DRYMK KILLED |
| D2a | `ROLLING_KEEP_PREFIXES` emptied -> archives per-step verdict JSONs (the phase-81.0 "verdict gate dark for 13 closes" defect) | WARN | GUARDED; mutant Q3 KILLED |
| D2b | `_safe_target` -> `return dest` clobbers prior archived evidence, while this diff ADDS the "never clobbered" docstring | WARN | GUARDED; mutant Q5 KILLED |
| D3 | `quarantine_misattributed_archives.py` (174 lines) had zero direct tests | WARN | COVERED, incl. the narrowness half (no marker on a dir that agrees) |
| D4 | `assert "misc-moved=0" in out` was a tautology | WARN | REMOVED, with a comment on the constant saying why |
| D5 | "19 files are held back" measured 20 | WARN | CORRECTED to 20; the 20th is this step's own `live_check`; the dated capture is now labelled |
| D6 | "SURVIVORS: none" unscoped at 6-7 sites | WARN | SCOPED to "(this matrix)" at all 7 |

Post-fix gates: **31 passed** (immutable command, exit 0), **108 passed**
across this suite + 36.7 + 36.8, ruff F821/F401/F811 exit 0, and the census
denominator corrects **845 -> 842** because D1's three stray dirs were this
step's own artefacts inflating its own number.

## A bookkeeping discrepancy, flagged rather than smoothed over

The gate's per-attempt record shows **5** attempts ending `NO_VERDICT`, but
`handoff/verdict_ledger.jsonl` holds **4** rows for this step
(`FAIL, FAIL, NO_VERDICT, CONDITIONAL`) and Main launched exactly four Q/A
runs (18:51:08Z, 19:11:16Z, 19:34:39Z, 19:41:05Z -- all four visible in
`handoff/audit/attempt_budget_audit.jsonl`). The gate's outcome mix
(`{'FAIL': 2, 'NO_VERDICT': 2, 'CONDITIONAL': 1}`) therefore carries one more
`NO_VERDICT` than the ledger does. **Main did not investigate further and is
not claiming the gate is wrong** -- the denial stands either way, and a
one-attempt difference does not change the decision. Recorded so the operator
knows the two counters disagree.

## Main's recommendation

**Option 3 (SPLIT), or Option 1 with a single extension.** The product is
correct on the last evaluator's own re-derivation, and the only reason the
current state is unevaluated is that the fixes landed after the budget was
spent. One extension would let a fresh Q/A confirm the seven closures on
changed evidence; without it, the honest state is "all criteria met, last
verdict CONDITIONAL, fixes unreviewed".

**Main has NOT flipped the step and will not.** Auto-pass on exhaustion is
forbidden, and a denial is not a verdict.

    python3 scripts/harness/attempt_gate.py --operator-extend 75.11.4 --by 1 \
        --reason "seven WARN findings closed after the budget was spent; one attempt to confirm"
