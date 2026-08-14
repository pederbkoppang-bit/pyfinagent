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

## 2. The paper_trades sweep -- PARTIAL (a "RESOLVED" claim here was WRONG; see 2c)

**Enumeration rule.** Population = every `paper_trades` row with
`UPPER(action)='BUY'`, all time = **34** (`COUNT(*)=66`, `COUNTIF(BUY)=34`,
`COUNT(DISTINCT trade_id)=66`, taken from the table). Joined to
`analysis_results` on `ticker` AND
`ABS(TIMESTAMP_DIFF(analysis_date, TIMESTAMP(analysis_id), SECOND)) < 2`.
Verdict read **nested-first then flat** (the lite path is flat).

```
INVERSION -- a verdict of REJECT or 0% yet a BUY executed :  1   <- DELL, and only DELL
verdict PERMITTED the buy                                 :  0
UNDETERMINED                                              : 33
POSITIVE CONTROL -- DELL detected                         : True
```

### 2a. The criterion's question, answered as far as the data allows

> *"report how many positions were sized at the 10%-NAV default while a completed
> risk verdict existed"*

**One position is CONFIRMED: DELL, 2026-08-13, $2,392.26 = 10.00% of NAV against
REJECT/0%.** For the other 33 the question **cannot be answered from persisted
data** -- and that is reported as a limit, not as a zero.

### 2b. What DID improve: the 33 now have a cause decomposition

Previously one undifferentiated bucket. Now:

| n | shape | why unrecoverable |
|---:|---|---|
| 19 | an `analysis_results` row EXISTS and joins | its `full_report_json` has **no `final_synthesis` subtree at all** -- the report is truncated |
| 14 | no row within 2s | nearest row per ticker is **15-20 DAYS** away; `analysis_results` holds **zero rows 2026-04-20..2026-05-15** while the table dates to 2025-11-23. Cause named in code: phase-24.2 F-2, *"full pipeline previously evaporated without persistence"*, closed by phase-25.A2 |

Both are persistence gaps of different shapes. **Neither is recoverable by widening
the join.**

### 2c. AN OVERCLAIM I MADE AND THEN REFUTED MYSELF -- recorded, not quietly deleted

An earlier revision of this section claimed **"C7 RESOLVED"**, on the reasoning that
the 19 were a *measured not-an-inversion* because *"the `risk_assessment` key is
absent entirely, so no verdict existed and the 10% default was legitimate"*.

**That reasoning is unsound, and the test that kills it is one query:**

```
final_synthesis PRESENT but risk_assessment absent :  0
final_synthesis ALSO absent (report truncated)     : 19
```

`final_synthesis` is absent in **all 19**. So the pipeline did **not** "reach
synthesis and attach no risk assessment" -- **the persisted report is truncated.**
A verdict may well have existed and simply never been written. "Key absent" only
supports *"not persisted"*; I read it as *"never existed"*, which is a strictly
stronger claim the data does not carry.

**The 19 therefore revert to UNDETERMINED**, and C7 stays **PARTIAL** exactly as
the Q/A's CONDITIONAL had it. I caught this by running the attack I had asked the
evaluator to run, *after* committing and pushing the wrong claim -- so the
correction is recorded here rather than the claim being silently removed.

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
