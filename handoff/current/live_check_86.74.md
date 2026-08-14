# live_check -- step 86.74

Required shape: "a driven 0%-verdict producing no order with the flag OFF, the
post-fix persisted-verdict share against the 0-of-129 baseline, and the
paper_trades sweep result with its enumeration rule."

Three items. **Two are satisfied by measurement; the third is NOT, and is
recorded as unsatisfied rather than substituted with a proxy.**

---

## 1. A driven 0%-verdict produces NO order with the flag OFF -- SATISFIED

Driven through the real `decide_trades` (not a stub), with
`paper_risk_judge_shape_fix_enabled=False` -- the shipped production state.

```
shape_fix=OFF (production)   BUY orders = 0   BLOCKED (correct)
shape_fix=ON                 BUY orders = 0   BLOCKED (correct)

For reference, the defect executed: BUY 4.8064 x DELL @ 497.72 = 2392.26
                       which is    : 10.00% of NAV 23920.63
```

The pre-fix code produced a **$2,392.26 BUY = exactly 10.00% of NAV** on this
input. Both flag states now produce **no order**.

## 2. The paper_trades sweep with its enumeration rule -- SATISFIED (as PARTIAL)

**Enumeration rule.** Population = every `paper_trades` row with
`UPPER(action)='BUY'`, all time = **34** (`COUNT(*)=66`, `COUNTIF(BUY)=34`,
`COUNT(DISTINCT trade_id)=66` -- taken from the table, not from a join). Joined to
`analysis_results` on `ticker` AND `ABS(TIMESTAMP_DIFF(analysis_date,
TIMESTAMP(analysis_id), SECOND)) < 2`. Verdict read from
`$.final_synthesis.risk_assessment.judge`. Flagged when a completed verdict was
`REJECT` or `pct = 0` and a BUY nevertheless executed.

```
INVERSION confirmed                 :  1   DELL 2026-08-13  notional 2392.26  REJECT/0.0
verdict permitted the buy           :  0
NO joinable verdict -> UNDETERMINED : 33   (2026-04-26 .. 2026-07-31)
POSITIVE CONTROL -- DELL detected   :  True
```

**The 33 are UNDETERMINED, not a measured zero.** Criterion 7 asks that a zero be
reported as a measured zero with a positive control; here the answer is not zero
and not clean -- it is **1 confirmed plus 33 unresolved**, and I am not claiming
DELL was the only occurrence. The positive control passes (the query provably
detects the known case), so the **1** is trustworthy; the **33** are a coverage
limit of the join, not evidence of absence.

## 3. Post-fix persisted-verdict share vs the 0-of-129 baseline -- **NOT SATISFIED**

**Baseline, reproduced exactly** (`total_rows=129, decision=0, risk_level=0,
pct=0`, 2026-07-20..2026-08-13) with the query in `experiment_results_86.74.md` §C4.

**The post-fix share cannot be measured yet.** It requires an autonomous cycle to
run with the new code, and:

- backend restarts are **batched to session end** (standing operator instruction);
- the running process (`pid 27945`, started 2026-08-14 13:30:35 CEST) still holds
  the **pre-fix** module -- committed is not in force;
- fabricating or estimating a share would be inventing a measurement.

The write is proven at the **unit seam** (`TestVerdictIsPersistedPerTicker`, and
mutation **M3** deleting the write turns it red), **not in BigQuery**. This item
is therefore **open**, and the step should not be read as having demonstrated it.

---

## Flag state at capture (measured in-process, not from the file)

```
paper_risk_judge_reject_binding      True     <- backend/.env:84
paper_risk_judge_parse_fail_reject   False
paper_risk_judge_shape_fix_enabled   False    <- the defect; still OFF, unchanged
paper_atomic_swap_enabled            False
```

No flag was promoted and no `.env` was written by this step.
