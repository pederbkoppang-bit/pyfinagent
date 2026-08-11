# live_check -- step 86.32

Required: the 86.28 eight-attempt replay showing where it terminates; the
demonstration that a fabricated-transcript FAIL is still a FAIL; and the check
proving exhaustion cannot auto-pass.

**Regenerated 2026-08-11 after the cycle-1 Q/A FAIL.** The previous revision's
per-attempt table misattributed 3 of 5 printed rows.

---

## 1. The 86.28 eight-attempt replay -- where it terminates

The fixture is REBUILT FROM THE RECORD after the cycle-1 Q/A FAILED this
step for using a sequence that was not the 86.28 series. Each row carries its
source; `test_fixture_matches_the_recorded_ledger` re-derives it from those
files rather than asserting properties of the constant.

| # | cycle | run | outcome | source |
|---|---|---|---|---|
| 1 | 1 | `wf_10c6cbd2-cad` | CONDITIONAL | evaluator_critique_86.28.md::ledger |
| 2 | 2 | `wf_d0934c91-70b` | CONDITIONAL | evaluator_critique_86.28.md::ledger |
| 3 | 3 | `wf_01c83c86-09d` | **NO VERDICT** | evaluator_critique_86.28.md::ledger |
| 4 | 3 | `wf_e262facc-cdc` | FAIL | evaluator_critique_86.28.md::ledger |
| 5 | 4 | `wf_5a217e41-9b9` | CONDITIONAL | evaluator_critique_86.28.md::ledger |
| 6 | 5 | `wf_344395f1-4ac` | CONDITIONAL | evaluator_critique_86.28.md::ledger |
| 7 | 6 | `wf_9c55b720-ef3` | **NO VERDICT** | evaluator_critique_86.28.md::ledger |
| 8 | 7 | `wf_e03ec2d0-c07` | **NO VERDICT** | live_check_86.28.md::section-9 |

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
- verdicts seen : 4  (so 1 attempt(s) produced NO verdict and cost tokens anyway)
- outcome mix   : {'CONDITIONAL': 3, 'NO_VERDICT': 1, 'FAIL': 1}

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
- attempt 2: CONDITIONAL  run=wf_d0934c91-70b
- attempt 3: NO_VERDICT  run=wf_01c83c86-09d
- attempt 4: FAIL  run=wf_e262facc-cdc
- attempt 5: CONDITIONAL  run=wf_5a217e41-9b9
```

## 2. A fabricated-transcript FAIL is STILL a FAIL
```
  close_kind(product=True , evidence=True ) -> CONTINUE
  close_kind(product=True , evidence=False) -> CONTINUE
  close_kind(product=False, evidence=True ) -> CONTINUE
  close_kind(product=False, evidence=False) -> CONTINUE

  No combination closes. The residuals door needs an actual Q/A PASS.
```

## 3. Exhaustion cannot auto-pass
```
  exhaustive sweep, lengths 1..6
    sequences examined  : 1092
    yielding CLOSED_PASS: 0
```

## 4. Scope

The budget is **not wired into `run_harness.py`**. This proves the MECHANISM;
it is not evidence that any production loop is bounded. `run_harness.py:1177`
is documented, not edited.
