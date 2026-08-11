# live_check -- step 86.32

Required by `verification.live_check`: the 86.28 eight-attempt replay against
the new rule showing where it terminates; the demonstration that a
fabricated-transcript FAIL is still a FAIL; and the check proving exhaustion
cannot auto-pass.

Captured 2026-08-11 by Main (`pyfinagent-06`). All output verbatim from
`scripts/harness/attempt_budget.py`.

---

## 1. The 86.28 eight-attempt replay -- where it terminates

```json
{
  "attempts_in_fixture": 8,
  "new_rule_terminates_at_attempt": 5,
  "new_rule_disposition": "ESCALATE",
  "legacy_consecutive_fails_final": 0,
  "legacy_would_have_terminated": false,
  "verdicts_seen": 5,
  "dropped": 3,
  "attempts_invisible_to_legacy_counter": 3
}
```

### The escalation it produces

```
# BUDGET EXHAUSTED -- step 86.28 -- OPERATOR DECISION REQUIRED

- attempts used : 5 / 5
- tokens used   : 0 / 1,200,000
- verdicts seen : 3  (so 2 attempt(s) produced NO verdict and cost tokens anyway)
- outcome mix   : {'CONDITIONAL': 3, 'NO_VERDICT': 2}

## THIS IS NOT A PASS AND NOT A FAIL

The step is NOT verified. No verdict is implied by exhaustion, and none
may be inferred from it. The loop stopped because it hit a cost ceiling,
which says nothing about whether the work is correct.

## What the operator must decide

1. RAISE the budget and continue, if the remaining work is bounded and known.
2. PARK the step with a written disposition (this project's existing vocabulary).
3. SPLIT it: close the verified part, queue the residuals as their own steps.

## Per-attempt record

- attempt 1: CONDITIONAL  run=wf_10c6cbd2-cad
- attempt 2: NO_VERDICT  run=wf_23d9ed4b-22c
- attempt 3: CONDITIONAL  run=wf_d0934c91-70b
- attempt 4: NO_VERDICT  run=wf_4da39b31-695
- attempt 5: CONDITIONAL  run=wf_e262facc-cdc
```

## 2. A fabricated-transcript FAIL is STILL a FAIL

```
  close_kind(product_verified=True , evidence_complete=True ) -> CONTINUE
  close_kind(product_verified=True , evidence_complete=False) -> CONTINUE
  close_kind(product_verified=False, evidence_complete=True ) -> CONTINUE
  close_kind(product_verified=False, evidence_complete=False) -> CONTINUE

  No combination yields CLOSED_COMPLETE or CLOSED_PRODUCT_RESIDUALS_QUEUED.
  The residuals door is reachable ONLY from an actual Q/A PASS.
```

## 3. Exhaustion cannot auto-pass

```
  exhaustive sweep over every non-PASS sequence, length 1..6
    sequences examined : 1092
    yielding CLOSED_PASS: 0
  -> auto-pass on exhaustion is unreachable (1092 sequences, 0 passes)
```

## 4. Scope note

The budget is **not yet wired into `run_harness.py`**. This capture proves the
MECHANISM and its guards; it is not evidence that any production loop is
currently bounded. The reset at `run_harness.py:1177` is documented, not edited.
