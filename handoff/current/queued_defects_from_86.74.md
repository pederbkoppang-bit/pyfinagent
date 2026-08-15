# Defects discovered during 86.74 -- to be QUEUED as their own steps

Held in the scratchpad, NOT the repo, because the tree is frozen during EVALUATE.
Each was found while working 86.74 and deliberately NOT fixed inline
(`feedback_queue_discovered_defects_in_masterplan`). Written for an executor with
no context from this session.

---

## D1 (P1) -- two swap-path tests are RED at HEAD and nobody noticed

**Measured, not inferred.** Both fail with `portfolio_manager.py` reverted to
`HEAD`, i.e. before any 86.74 change:

```
backend/tests/test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
    AssertionError: Expected 2 swap SELLs, got 1
backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
```

Method: `git show HEAD:backend/services/portfolio_manager.py > <path>`, run, then
restore from a scratchpad copy and verify sha256 byte-identical.

**Why it matters beyond the two tests.** These cover the SWAP path, which is one
of the four sizing seams 86.74 touched, and the swap path can open positions.
A red test on a money path that nobody is watching is the
operations-that-cannot-fail-loudly class again.

**Do NOT assume 86.74 caused these.** It did not; that was measured. Start by
bisecting to find which commit turned them red, then decide whether the test or
the code is wrong -- `test_swap_framework_fills_zero_buy_gap` expects 2 swap
SELLs and gets 1, which could legitimately be either.

**Non-scope:** do not delete or `xfail` either test to get green.

---

## D2 (P1) -- the historical inversion sweep is **14**/34 UNDETERMINED (was 33; corrected 2026-08-14)

86.74 criterion 7 asked how many prior positions were opened under the
falsy-zero/nesting inversion. Answer: **1 confirmed (DELL 2026-08-13), 14 of 34
BUYs undetermined** -- NOT a clean bill, but far better than first reported.

**THE "33" WAS A PROPERTY OF THE ENUMERATION RULE, NOT OF THE DATA.** Every earlier
version read verdict-existence from ONE source, the `analysis_results` JSON blob.
`paper_trades.risk_judge_decision` is a **second, per-trade** verdict column written
on the BUY row itself; it is populated on **19 of 34** BUY rows (15
`APPROVE_REDUCED`, 3 `REJECT`, 1 `APPROVE_HEDGED`) and cross-tabulates **exactly**
onto the 19 rows the blob-based rule called "truncated / undetermined". D5's
truncation destroyed the blob copy of the verdict; the per-trade column survived it.
Only the 14 with no `analysis_results` row within 2s are absent in both sources.

**Inversion count is unchanged at 1.** The 3 `REJECT` BUYs (HPE 2026-06-02 $245.04,
DELL 2026-06-03 $246.67, 066570.KS 2026-06-09 $238.40) are ~1% of a ~$24k book,
against the $2,392.26 a 10%-NAV default produces there -- reduced sizing, not the
default. **Residual for the executor:** that ratio is anchored on the 2026-08-13
NAV, because `paper_portfolio` holds current state only and the historical snapshot
table (`pyfinagent_pms`, US) is in a different BigQuery *location* from
`financial_reports` (us-central1), so a per-date notional/NAV join needs two queries.
Doing that join properly is the remaining D2 work.

**Enumeration rule used** (reuse it): population = `paper_trades` rows with
`UPPER(action)='BUY'`, all time = 34 (`COUNT(*)=66`, `COUNTIF(BUY)=34`,
`COUNT(DISTINCT trade_id)=66`). Join to `analysis_results` on `ticker` AND
`ABS(TIMESTAMP_DIFF(analysis_date, TIMESTAMP(analysis_id), SECOND)) < 2`; verdict
from `$.final_synthesis.risk_assessment.judge`.

**Why 14 are absent from BOTH sources** is the actual open question. `analysis_results`
holds 567 rows (2025-11-23..2026-08-13), 372 with a nested judge verdict, so the data
broadly exists -- the JOIN is the limit, not the corpus.

**The 14 truly-undetermined BUYs, re-measured 2026-08-15** (`ar_verdict IS NULL AND
no analysis_results row within 2s`, and `paper_trades.risk_judge_decision` empty on
all 14):

| date | n | tickers |
|---|---:|---|
| 2026-04-26 | 9 | CIEN, DELL, GLW, INTC, LITE, ON, SNDK, TER, WDC |
| 2026-04-27 | 1 | COHR |
| 2026-04-28 | 3 | GEV, KEYS, MU |
| 2026-05-01 | 1 | FIX |
| **total** | **14** | |

All four dates fall inside the `2026-04-20..2026-05-15` window where
`analysis_results` holds **zero** rows -- so these are a persistence gap, not a join
defect, and widening the join tolerance cannot recover them.

*(An earlier revision of this paragraph asked "why 33 don't join" and listed
`04-26 (9), 04-27 (1), 04-28 (3), 05-01 (1), 05-29 (1), 06-01..06-10 (12), 07-09 (2),
07-20 (1), 07-31 (1)` under the heading "Undetermined BUY dates". That list was wrong
three ways: it was labelled undetermined when its last five entries are the
RECOVERABLE truncated bucket, it summed to **31** rather than the 33 it claimed, and
its recoverable tail listed 17 of the actual 19. Corrected against a re-run of the
enumeration rule rather than by editing the number.)*

**Positive control that must keep passing:** DELL 2026-08-13 must be detected by
whatever join you build. A sweep that cannot find the known case cannot report a
zero.

**Deliverable:** either resolve the remaining **14** to a verdict each, or state
precisely why a given BUY is permanently unattributable, and report the final count
with the population rule beside it. Report zero, if it is zero, as a MEASURED zero.
The 19 truncated rows are already resolved -- do not re-open them; read
`paper_trades.risk_judge_decision` (15 `APPROVE_REDUCED`, 3 `REJECT`,
1 `APPROVE_HEDGED`).

---

## D3 (P1) -- ~~verify the persisted-verdict fix in BigQuery after the restart~~ **DONE 2026-08-14**

**CLOSED.** The restart landed in the prior session (`d6a1500a`, 15:52:58Z), the
scheduled cycle `68925781` ran 18:00:00Z->19:33:13Z on the resulting process
(pid 85562), and the baseline/post-fix comparison below was executed: **129 rows
0/0/0 -> 6 rows 6/6/6 = 6 of 6 (100%)**. Full detail in `live_check_86.74.md` §3
and `experiment_results_86.74.md` §C4. No manual cycle was run.

*(The paragraph that stood here said the fix was "NOT in BQ -- the running process
still holds pre-fix code and restarts are batched to session end". Both clauses
were false by the time a reader would act on them, and the risk was that a reader
would trigger the restart or the cycle that the batching policy exists to prevent.)*

**Baseline to compare against** (reproduce it first, it must still be 0/129):

```sql
SELECT COUNT(*) total,
       COUNTIF(risk_judge_decision IS NOT NULL AND risk_judge_decision != '') dec_pop,
       COUNTIF(risk_level IS NOT NULL AND risk_level != '') lvl_pop,
       COUNTIF(recommended_position_pct IS NOT NULL) pct_pop
FROM `sunny-might-477607-p8.financial_reports.analysis_results`
WHERE DATE(analysis_date) BETWEEN '2026-07-20' AND '2026-08-13'
-- expected: 129 / 0 / 0 / 0
```

After the next autonomous cycle on restarted code, the same query over the NEW
window must show a non-zero populated share. **State the row count of the new
window beside the share** -- a 100% share over 1 row is not evidence.

**Trap:** `committed is NOT in force`. Confirm the running pid started AFTER the
86.74 commit (`ps -o pid,lstart -p <pid>` vs `git log -1 --format=%cd 9d14291e`)
before reading any post-fix number.

---

## D4 (P2) -- `_extract_position_pct`'s legacy shim is a remaining trap

86.74 fixed the falsy-zero inside it, but the shim still returns `Optional[float]`
and therefore still collapses `UNPARSEABLE` and `ABSENT` to the same `None`.
Every sizing caller was moved to `_resolve_position_pct` / `_sizing_pct`, so no
live money path depends on the shim -- but it remains importable and three tests
in `test_dod4_tier1_coverage_investment.py` still call it.

**Deliverable:** either delete it and update those three tests, or document at the
definition that it must never be used for sizing. Low urgency, real trap.

---

## D5 (P1) -- `analysis_results` rows persisted with NO `final_synthesis` subtree, and it is STILL FIRING

### D5 archaeology, 2026-08-14 -- the break is 06-11, and it is NOT a commit that removed a key

The prior goal recorded *"ONE change ~06-11/06-14 fits; COMMIT NOT IDENTIFIED"*.
Four findings, all measured, that reframe what is being looked for.

**1. The break is 2026-06-11, not 06-15.** Daily (not monthly -- monthly hides it):

| date | rows | missing `final_synthesis` | % |
|---|---:|---:|---:|
| 2026-06-05 | 9 | 9 | 100.0 |
| 2026-06-08 | 5 | 5 | 100.0 |
| 2026-06-09 | 5 | 5 | 100.0 |
| 2026-06-10 | 5 | 5 | 100.0 |
| **2026-06-11** | **8** | **1** | **12.5** |
| 2026-06-12 | 5 | 2 | 40.0 |
| 2026-06-15 | 7 | 0 | 0.0 |
| 2026-06-17 | 5 | 1 | 20.0 |
| 2026-06-22 | 7 | 0 | 0.0 |

**2. It is NOT a step function at row level -- so "a commit flipped it" is the wrong
shape.** Within the autonomous cycles: 06-10 (18:03-18:38Z) is 5/5 missing; 06-11
(18:35-19:09Z) is 5/5 present; **06-12 is MIXED** -- 18:03:27Z and 18:04:14Z missing,
then 18:35:23Z / 18:36:22Z / 18:39:05Z present. A per-deploy cause cannot produce a
split inside one cycle. The mechanism is **per-analysis**: an analysis that does not
reach synthesis persists a report with no such key.

**3. NO commit in the window touches `final_synthesis` by name.** Checked all five
candidates that touch the pipeline (`78b264bf` 03:41Z, `fa62b5fe` 11:06Z,
`7f0de140` 11:51Z, `6a4fc351` 13:41Z, `b0fe1983` 14:30Z): added/removed
`final_synthesis` lines under `backend/` = **0 / 0 on every one**. So the cause is
indirect and no textual search for the key will find it.

**4. The four phase-60 commits are COLLINEAR and timing cannot separate them.** All
four land 11:06Z-14:30Z, i.e. before the 06-11 18:35Z cycle that flipped. The
best-fitting candidate by NAME is `fa62b5fe` *"phase-60.1: deep-pipeline restoration
+ honest-degradation alarm"* (+161 lines in `orchestrator.py`), and a restored deep
pipeline reaching synthesis where a degraded one did not fits every row above -- but
that is a hypothesis consistent with the data, **not an attribution**, and the two
06-11 morning rows (10:17Z, 10:44Z) already carry synthesis BEFORE `fa62b5fe`
landed, so even the timing is not clean (those are off-cycle, likely API-triggered).

**5. THE DISCRIMINATING TELEMETRY DOES NOT EXIST FOR THE BREAK WINDOW.** The obvious
test -- correlate missing-synthesis against the per-cycle degradation rate -- cannot
be run: `handoff/cycle_history.jsonl` carries `degradation` as `None` for every June
cycle. The instrument was itself added by the phase-60.1 "honest-degradation alarm",
so **it postdates the thing it would have measured.**

**Conclusion for whoever picks this up:** stop looking for the commit that removed a
key; there isn't one. The question is *"what made analyses stop reaching synthesis
so often"*, the answer is a RATE not a flip, and the June telemetry to attribute it
is absent. The tractable path is forward-looking: D5 is **still firing** (8.8% in
2026-08), so instrument the CURRENT truncation with the degradation fields that now
exist, rather than trying to attribute a June regime change from data that cannot
support it.

**Surfaced 2026-08-14** while adding a positive control the Q/A asked for. Named by
the Q/A as absent from the masterplan: it searched the pending steps for
`final_synthesis`/truncation and got 23 hits, **none of them this** — the nearest
are **61.2** (persisting a synthetic `0.00`/`HOLD` on synthesis FAILURE) and
**75.5.8** (truncation-blind LLM-JSON parsers). Both adjacent; neither the same
defect.

**MEASURED (use `JSON_QUERY`, NOT `JSON_VALUE` — see the trap below):**

```sql
SELECT DATE_TRUNC(DATE(analysis_date),MONTH) month, COUNT(*) rows_total,
  COUNTIF(JSON_QUERY(full_report_json,'$.final_synthesis') IS NULL) truncated
FROM `sunny-might-477607-p8.financial_reports.analysis_results`
GROUP BY month ORDER BY month
```

| month | rows | truncated | % |
|---|---:|---:|---:|
| 2025-11 .. 2026-03 | 54 | 0 | 0.0% |
| 2026-05 | 174 | 58 | 33.3% |
| 2026-06 | 134 | 68 | **50.7%** |
| 2026-07 | 137 | 12 | 8.8% |
| 2026-08 | 68 | 6 | **8.8% — STILL FIRING** |

**Why it matters beyond tidiness.** A row with no `final_synthesis` carries no
verdict, no recommendation and no rationale, so **no audit of a trade against its
risk verdict is possible for that row**. It is exactly what makes 86.74's criterion
7 permanently unclosable by measurement — and because it is still firing, **the
undetermined set GROWS**. That converts C7 from "unrecoverable backwards" to
"accumulating".

**Likely related to 86.69** (81% of analyses persist as an empty HOLD scored 0.0,
dated break 06-12/06-15). The 50.7% June peak here overlaps that window. **Do not
assume they are the same defect** — establish it or refute it.

### THE TRAP, pinned so the next reader does not repeat it

`JSON_VALUE(full_report_json,'$.final_synthesis')` **extracts scalars only and
returns NULL for an object**, so it reports "absent" for **every row in the table**
(567 of 567), including rows whose nested judge decision reads back fine. I used it,
got a 100%-truncation result that contradicted a value I had already read from the
same row, and only caught it because the Q/A insisted the decisive zero needed a
positive control.

**Positive control to reuse:** DELL `2026-08-13` has `judge.decision='REJECT'`, so
its `final_synthesis` **must** be present. Any probe that calls that row truncated
is broken.

**Deliverable:** determine whether the write is still producing truncated rows
today (8.8% suggests yes), find the producer, and either fix it or record why a
truncated row is unavoidable. Report the rate with the month breakdown and the
population rule beside it.
