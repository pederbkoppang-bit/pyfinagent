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

## 3. Post-fix persisted-verdict share vs the 0-of-129 baseline -- **SATISFIED 2026-08-14**

*(This section previously read "NOT SATISFIED -- cannot be measured yet". That was
true when written and is now false; it is REPLACED rather than annotated, because a
correction that merely accompanies the old text leaves two live claims in one file.)*

**Baseline, reproduced exactly** (`total_rows=129, decision=0, risk_level=0,
pct=0`, 2026-07-20..2026-08-13) with the query in `experiment_results_86.74.md` §C4.

**The blocker cleared without a manual cycle and without a restart.** The claim in
the old text -- that the running process held the pre-fix module -- was true of
`pid 27945`, but that process is gone. Measured, not inferred:

| fact | value | source |
|---|---|---|
| C4 fix committed | **2026-08-14T14:36:20Z** | `git log -1 --format=%cI -S risk_judge_decision -- backend/services/autonomous_loop.py` (9d14291e) |
| running backend started | **2026-08-14T15:52:08Z** (pid 85562) | `ps -o pid,lstart -p 85562` |
| scheduled cycle | started **18:00:00Z**, completed **19:33:13Z**, 0 trades | `handoff/cycle_history.jsonl` (cycle `68925781`) |

The process started **76 minutes after** the fix landed, so the 18:00Z cycle
executed post-fix code. **No cycle was triggered manually** -- this is the ordinary
scheduled run.

**THE MEASUREMENT, with the row count beside the share:**

```
BASELINE 2026-07-20..2026-08-13 : total_rows=129  decision=0  risk_level=0  pct=0   ->   0 of 129 (0%)
POST-FIX 2026-08-14             : total_rows=  6  decision=6  risk_level=6  pct=6   ->   6 of 6 (100%)
```

Per-ticker, all six from inside the cycle window:

| ticker | `risk_judge_decision` | `pct` | `analysis_date` |
|---|---|---:|---|
| PANW | REJECT | 0 | 18:35:23Z |
| WDAY | REJECT | 0 | 18:36:27Z |
| HPE | REJECT | 0 | 18:37:54Z |
| STX | APPROVE_REDUCED | 2 | 19:02:32Z |
| MRVL | REJECT | 0 | 19:04:26Z |
| NTAP | APPROVE_REDUCED | 2 | 19:32:26Z |

**Why this is not a vacuous green.** The column is not populated with one constant:
two distinct decisions (`REJECT`, `APPROVE_REDUCED`) and two distinct pcts (0, 2)
appear, so the write is carrying real per-ticker verdict content rather than a
literal. The 0%/REJECT rows are also the shape that the C7 inversion check reads,
so the same write feeds the gate this step exists to protect.

**WHAT IS NOT CLAIMED.** n = **6 rows, one cycle**. This demonstrates the write
path now populates all three columns on the autonomous rail; it is **not** a
stability claim over time, and a single cycle cannot be one. The unit-seam proof
(`TestVerdictIsPersistedPerTicker`, mutation **M3** turning it red) still carries
the regression guard; BigQuery now corroborates it end-to-end.

---

## Flag state at capture (measured in-process, not from the file)

```
paper_risk_judge_reject_binding      True     <- backend/.env:84
paper_risk_judge_parse_fail_reject   False
paper_risk_judge_shape_fix_enabled   False    <- the defect; still OFF, unchanged
paper_atomic_swap_enabled            False
```

No flag was promoted and no `.env` was written by this step.

### 2d. The retraction in 2c was RIGHT — but I reached it with a BROKEN PROBE

The evaluator flagged that 2c's decisive number is a **zero with no positive
control**, and that criterion 7's own standard forbids exactly that. Adding the
control **found an error in my own instrument**:

```
JSON_VALUE(full_report_json,'$.final_synthesis') IS NULL   -> TRUE for 567 of 567 rows
```

**`JSON_VALUE` extracts SCALARS ONLY and returns NULL for an object**, so it
reported "final_synthesis absent" for **every row in the table** — including
DELL's 2026-08-13 row, from which I had *successfully read*
`$.final_synthesis.risk_assessment.judge.decision = 'REJECT'`. Both cannot be true;
the probe was answering a question about types, not about content.

**Positive control, on the artifact rather than invented:**

```
DELL 2026-08-13, judge_decision = 'REJECT'
  JSON_VALUE says final_synthesis absent : True    <- FALSE POSITIVE
  JSON_QUERY says final_synthesis absent : False   <- correct
```

**Re-measured with `JSON_QUERY`, the answer is UNCHANGED:** `final_synthesis` is
absent in **19 of 19**. So **2c's conclusion stands** — the reports are truncated,
and "no verdict existed" remains unsupportable. **C7 stays PARTIAL.**

But I got the right answer for the wrong reason, and that is worth stating: a probe
that returns `TRUE` for every row cannot distinguish anything. Had the 19 in fact
been synthesis-present, my broken probe would have hidden it and I would have
"confirmed" the retraction just as confidently. **The control is what turned a
lucky answer into a measured one.**

### 2e. NEW DEFECT, surfaced by that control — truncation is STILL FIRING

Correctly measured, the truncated-report shape is real, was never historical, and
is **accumulating**:

| month | rows | truncated | % |
|---|---:|---:|---:|
| 2025-11 .. 2026-03 | 54 | 0 | 0.0% |
| 2026-05 | 174 | 58 | **33.3%** |
| 2026-06 | 134 | 68 | **50.7%** |
| 2026-07 | 137 | 12 | 8.8% |
| 2026-08 | 68 | 6 | **8.8% — still firing** |

**An `analysis_results` row persisted with no `final_synthesis` subtree at all** is
a persistence defect in its own right. It is *why* C7 is permanently unclosable by
measurement, and because it is still firing, **C7's undetermined set grows**. Queued
as its own step (`queued_defects_from_86.74.md` D5) rather than fixed here.

### 2g. The 19/14/0 split INDEPENDENTLY RE-DERIVED (2026-08-14, later session)

The split above was, on its own admission, single-authored. It has now been
re-derived from the stated enumeration rule by a **query written from scratch**,
not by re-running the original SQL, with the instrument controlled first.

**Instrument control, run BEFORE the classification** (the failure mode 2d records):

```
DELL 2026-08-13   judge.decision = 'REJECT'
  JSON_VALUE(full_report_json,'$.final_synthesis') IS NULL  -> TRUE    <- FALSE POSITIVE
  JSON_QUERY(full_report_json,'$.final_synthesis') IS NULL  -> FALSE   <- correct
```

`JSON_QUERY` is therefore the only instrument used below.

**Result -- every bucket reproduces exactly:**

| bucket | n |
|---|---:|
| `INVERSION` (REJECT or 0% yet a BUY executed) | **1** (DELL, and only DELL) |
| `PERMITTED` (verdict allowed the buy) | **0** (bucket empty) |
| `UNDET_truncated_no_final_synthesis` | **19** |
| `UNDET_no_row_within_2s` | **14** |
| `UNDET_fs_present_but_no_risk_assessment` | **0** (bucket empty) |

**Completeness check, which is what makes the zeros meaningful:**
1 + 0 + 19 + 14 + 0 = **34** = the full `UPPER(action)='BUY'` population. Because the
buckets sum to the population, no row is unclassified **and no BUY fanned out to two
`analysis_results` rows** -- a join fan-out would have pushed the total above 34.
The two empty buckets are therefore measured zeros, not absent categories.

**WHAT THIS DOES AND DOES NOT SETTLE.** It settles that the numbers are
reproducible from the enumeration rule by an independently written query, which is
what "the split is mine alone" was flagging. It does **not** make C7 closable: the
33 remain unrecoverable for the two persistence reasons already recorded, and 2e's
truncation defect is still firing (D5). **C7 stays PARTIAL.** Independent
*third-party* confirmation is a Q/A's job, not a second derivation by the same
author -- that limit is stated here rather than papered over.

### 2h. A SECOND VERDICT SOURCE cuts undetermined from 33 to 14 (cycle-6 Q/A, WARN)

**The "33 UNDETERMINED" figure was a property of my enumeration rule, not of the
data.** Every version of this section enumerated verdict-existence from ONE source —
the `analysis_results.full_report_json` blob. `paper_trades` carries its own
per-trade column, **`risk_judge_decision`**, written on the BUY row itself. Measured:

```
paper_trades BUY rows                         : 34
... with risk_judge_decision populated        : 19   (15 APPROVE_REDUCED, 3 REJECT, 1 APPROVE_HEDGED)
```

**Cross-tabulated against my own buckets, the mapping is exact:**

| my bucket | `paper_trades` verdict | n |
|---|---|---:|
| `AR_verdict_known` (DELL, the inversion) | absent | 1 |
| `UNDET_no_row_within_2s` | absent | **14** |
| `UNDET_truncated_no_final_synthesis` | **PRESENT** | **19** |

**All 19 rows I called undetermined have a persisted verdict after all.** The
truncation defect (2e/D5) destroyed the *blob* copy; the per-trade column survived.
So the honest count is **14 undetermined, not 33** — and 2c's reasoning is
vindicated in the strongest possible way: "key absent" really did mean *not
persisted here*, never *never existed*, and the verdict was recoverable from
somewhere else entirely.

**Does this create new inversions? No — inversion stays 1.** C7 asks for positions
*sized at the 10%-NAV default while a completed verdict existed*. The three REJECTs
are the only inversion candidates on the verdict leg:

| ticker | date | notional |
|---|---|---:|
| HPE | 2026-06-02 | $245.04 |
| DELL | 2026-06-03 | $246.67 |
| 066570.KS | 2026-06-09 | $238.40 |

All three are **~$240 against a ~$24k book (~1% of NAV)** — an order of magnitude
below the 10% default, which on this book is $2,392.26 (the DELL inversion's exact
size). These are the known phase-57.1 away-week trio, sized at a reduced pct.
**RESIDUAL, stated rather than papered over:** I anchored "~$24k" on the
2026-08-13 NAV because `paper_portfolio` holds current state only and the June
snapshot table lives in a different BigQuery *location*, so a single-query join was
not available. The conclusion is robust — NAV would have to have been ~$2,400 for
$240 to be the 10% default, against a $25,000 starting capital — but the ratio is
not precise, and a per-date NAV join is queued rather than claimed.

**Method lesson.** Two independent operationalizations of "a verdict existed"
disagreed by 19 rows, and I only ever ran one. Where a second source exists, the
symmetric difference IS the measurement; a single-source enumeration reports the
rule as if it were the territory.

### 2f. The in-system precedent for 2c's reasoning

2c argues *"key absent supports not-persisted, never never-existed"* on principle.
**This step already proved it empirically** — criterion **C6** exists because
`signal_attribution.py` dropped the RiskJudge row entirely when `pct` was `None`, so
**a real DELL verdict left no trace in a persisted artifact**. Same system, same
verdict, same week. Anyone tempted to re-run my original reasoning should be pointed
at C6, not at an abstract argument.
