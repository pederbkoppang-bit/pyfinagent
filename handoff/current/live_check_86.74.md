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

## 2. The paper_trades sweep -- RESOLVED (supersedes the earlier PARTIAL)

**Enumeration rule.** Population = every `paper_trades` row with
`UPPER(action)='BUY'`, all time = **34** (`COUNT(*)=66`, `COUNTIF(BUY)=34`,
`COUNT(DISTINCT trade_id)=66`, taken from the table). Joined to
`analysis_results` on `ticker` AND
`ABS(TIMESTAMP_DIFF(analysis_date, TIMESTAMP(analysis_id), SECOND)) < 2`.
Verdict read **nested-first then flat** (`$.final_synthesis.risk_assessment.judge`
then `$.final_synthesis.risk_assessment`) -- the lite path is flat, and an earlier
version of this sweep read **nested only**, which is why it under-reported.

```
INVERSION -- a verdict of REJECT or 0% yet a BUY executed :  1
verdict PERMITTED the buy                                 :  0
joined, but the row carries NO risk verdict at all        : 19
NO joinable analysis row (permanently unattributable)     : 14
                                                     sum  : 34
POSITIVE CONTROL -- DELL detected                         : True
```

### The criterion's actual question, answered

> *"report how many positions were sized at the 10%-NAV default while a completed
> risk verdict existed"*

**Exactly ONE: DELL, 2026-08-13, $2,392.26 = 10.00% of NAV against REJECT/0%.**

### The 19 are a MEASURED not-an-inversion, not a gap

For all 19, the **`risk_assessment` key is ABSENT ENTIRELY** from the persisted
report (verified by `JSON_VALUE(...,'$.final_synthesis.risk_assessment') IS NULL`
returning true for 19 of 19). **No verdict existed**, so the 10% default was
legitimately applied -- the inversion is not merely unobserved, it is impossible
for these rows. 4 carry `_path='lite'` (the lite path skips risk assessment by
design, `orchestrator.py:1736`); 15 predate the `_path` provenance stamp.

### The 14 are PERMANENTLY unattributable, with a measured cause

All 14 fall in **2026-04-26 .. 2026-05-01**, and the nearest analysis row for each
ticker is **15-20 DAYS away** -- so this is not a join-tolerance problem and no
widening can rescue them:

```
analysis_results rows, 2026-04-20 .. 2026-05-20 :  none until 2026-05-16
earliest analysis_results row overall            :  2025-11-23
```

The table existed and was being written months earlier, so the gap is specific:
**full-path runs were not persisted at all** in that window -- documented in the
code as phase-24.2 F-2, *"full pipeline previously evaporated without
persistence"*, and closed by phase-25.A2's `_persist_analysis`
(`autonomous_loop.py:3382`). The analyses these 14 BUYs acted on were **never
written**, so no join can ever recover them.

**Determined: 20 of 34. Permanently unrecoverable: 14 of 34, with a stated cause.
Inversions: exactly 1.** This supersedes the earlier "33 UNDETERMINED", which
under-reported because it read only the nested verdict shape and did not
decompose the join failures.

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
