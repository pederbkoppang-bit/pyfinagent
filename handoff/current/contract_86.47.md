# Contract -- step 86.47

**Step:** 86.47 -- why did the book stop trading? The drought's cause is open
and unowned. **P2, money-adjacent.**

## Research-gate summary (what the gate CHANGED about the plan)

Gate **PASSED** (`wf_e21bd34e-bcb`; 14 sources read in full, 48 URLs against 50
distinct in the brief, audit-class dry after 14 rounds; brief
`research_brief_86.47.md`, 50,399 chars).

**The step exists because two prior steps were filed on drought theories their
own gates refuted, both by reasoning from a degradation they had just measured
to the drought. The gate refuted THREE of this step's own premises, and the
step text itself demanded exactly that.**

**1. The step's headline fact is STALE.** It says *"the last trade in
`financial_reports.paper_trades` was 2026-07-31 (NTAP)"*. Re-derived: the last
trade is **2026-08-13, a DELL BUY**. Everything downstream of "7 weekdays of
silence" is therefore built on a wrong endpoint.

**2. Two table locations in the step text are wrong.** `analysis_results` (580
rows, 91 columns) and `signals_log` live in **`financial_reports`**, not
`pyfinagent_data`. A census pointed at the stated dataset returns
`Table not found`.

**3. `risk_judge_decision` is populated on 18 of 580 analysis rows (3.1%)** --
zero in May, zero in July -- and `risk_intervention_log` has **0 rows**. So
criterion 3's warning is not hypothetical: a funnel keyed on that column
measures its own blindness. *(Note the two tables disagree and both are true:
in `paper_trades` the column is populated on 19 of 34 BUYs and 0 of 32 SELLs.
Any figure must name its table.)*

**4. The lite/full split IS derivable**, contrary to the step's doubt --
but only from **2026-06-11**, when `$._path` first appears. All-time coverage
is **288/580 = 49.7%**; from that epoch it is 100%. Criterion 4 is satisfiable
for the post-break window, and the pre-break baseline cell is path-UNKNOWN.

**5. The gate corrected my own denominator.** I had computed the silence as
3 weekdays; it is **2 analysis days / 13 analyses** -- 08-16 and 08-17 are a
weekend, and the 08-18 cycle had not run at measurement time (cycle hour 10 ET).
The opportunity unit is an ANALYSIS, not a weekday, and `paper_analyze_top_n=5`
caps analyses per cycle.

**6. A confound that forbids attributing the recovery.** FOUR changes land
together on **2026-08-14**: sonnet-4-6 -> sonnet-5, risk columns becoming
populated, zero-scores ceasing, and the recommendation default moving to Hold.
They are **not identifiable** from observational data.

## Hypothesis

The drought as framed does not exist. What exists is a real, deterministic,
UPSTREAM defect that suppressed BUY *supply* from 2026-06-11 to 2026-08-13 --
`final_synthesis.error = "Failed to parse final report."` -- and no gate is
implicated, because a gate cannot refuse a recommendation that was never
produced. The defect appears to have stopped on 2026-08-14, but four changes
landed together and 13 analyses is far too small to confirm recovery.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json`)

1. The step FIRST establishes whether the zero-trade run is anomalous, by re-deriving the trade base rate from BigQuery itself rather than inheriting the numbers in this step text, and STATES THE NORMALISATION RULE beside every rate (weekday vs calendar day, and the window's endpoints). If the run is within normal variance, saying so is a PASS outcome and no cause need be found.
2. A per-recommendation funnel census is produced over a stated window: how many analyses produced a BUY-class recommendation, how many reached the risk gate, how many were refused, and with what stated reason. Counts must be accompanied by the query and the window; a count without its predicate is a rejected outcome.
3. Before any funnel number keyed on risk_judge_decision is reported, the step PROVES that column is populated for the rows it counts -- 86.25 measured 32 of 32 SELL rows with an EMPTY value, so a census over that column can silently measure its own blindness. If the column is empty for BUYs too, that is the finding and the funnel must be derived another way or reported as underivable.
4. The lite-path confound is addressed explicitly: BUY recommendations originating on the 2-call lite wrapper are distinguished from those from the full 28-agent pipeline, because the two known BUY refusals of 2026-08-10 both came from the degraded path and a census that merges them cannot tell a gate problem from a degradation problem.
5. No gate, threshold, or risk parameter is loosened, and no flag is promoted. If the step concludes a gate is mis-calibrated, it says so with OUTCOME evidence (what the refused trades would have returned) and files the change as its own operator-gated step rather than making it.
6. Any causal claim survives an explicit base-rate check stated in the artifact: given the measured trade rate, how likely is the observed silence under the null of a healthy funnel? A cause asserted without that check is a rejected outcome, because two prior steps were filed on drought theories that their own research gates refuted.

**Immutable verification command:**
`bash -c 'source .venv/bin/activate && python -c "import ast;ast.parse(open(\"backend/services/autonomous_loop.py\").read());print(\"parsed\")"'`

**Immutable live_check:** `live_check_86.47.md` with the re-derived base rate
and its normalisation rule; the funnel census with its query and window; the
proof that `risk_judge_decision` is populated for the counted rows (or the
finding that it is not); and the base-rate check applied to whatever cause is
proposed.

## Plan

**This step ships a MEASUREMENT, not a behaviour change.** Its deliverable is a
re-runnable census script plus the artifact. No production file is modified --
which is the correct shape for a step whose first duty is to establish whether
a defect exists at all.

**P1 -- criterion 1, and it resolves NEGATIVE.** Re-derive the trade history
and the base rate from BigQuery, stating the normalisation rule and both
endpoints. Report the corrected last-trade date. Compute the silence in
**analyses**, not weekdays, and report P(zero) under the post-break BUY rate.
State the power bound: how many analyses would be needed for a zero-run to
clear p<0.05. **"The run is within normal variance" is the step's own stated
PASS outcome and is the expected result here.**

**P2 -- criterion 2, the funnel census, which is where the real signal is.**
The trade count is not decisive; the per-recommendation census is. Count
analyses by recommendation class over a stated window, with the query printed,
split by era. Report the dated break.

**P3 -- criterion 3, BEFORE any risk-gate number.** Prove the population state
of `risk_judge_decision` in **both** tables and report the coverage fraction.
Where it is too sparse to support a funnel, say the funnel is **underivable
from that column** and derive the refusal signal another way
(`final_synthesis.risk_assessment`) or report it as underivable. Naming the
table beside every figure is mandatory -- the two disagree.

**P4 -- criterion 4, the lite/full split, which IS derivable.** Partition every
count by `JSON_VALUE(full_report_json,'$._path')`. This is load-bearing rather
than cosmetic: the post-break BUY rate is **36.8% on lite-with-successful-
synthesis** against **0.0% on full-with-failed-synthesis**, so a merged census
would average a working path with a broken one and see neither.

**P5 -- criterion 6, the base-rate check, applied to BOTH claims and reported
with opposite outcomes.** The trade-count silence does NOT clear a 5% bar, so
it is not evidence of breakage. The synthesis-failure -> zero-BUY link clears
it by ~62 orders of magnitude (10^-61.7 over the full 236-row failed cell at the pre-break rate p=0.452). **Reporting that the same step's two candidate
claims fall on opposite sides of the bar is the point**, not an inconsistency.

**P6 -- what this step must NOT conclude.** Four changes landed together on
2026-08-14; the improvement is **not attributable** to any one of them. Two
analysis days is not evidence the fix worked, and not evidence it failed. The
artifact states the power requirement AS THE CENSUS COMPUTES IT rather than
declaring recovery. No power figure is restated in this contract: a number
restated in prose is a number that can go stale, and an earlier revision of
this very sentence carried a `~97` that the census had superseded.

## Scope honesty -- what this step does NOT do

- **It loosens no gate and promotes no flag** (criterion 5). It also does not
  claim any gate is mis-calibrated: the gate found that counting REJECTs
  cannot show miscalibration (r = -0.032), and criterion 5 requires OUTCOME
  evidence, which this step does not have.
- **It does not fix the synthesis parse failure.** That is step **86.108**'s
  subject -- the same `Synthesis-Final returned invalid JSON` population it
  made countable. This step measures the CONSEQUENCE and hands off; it must not
  claim 86.108's fix.
- **It does not re-open 86.38 or 86.41**, and does not claim the empty-HOLD
  persistence question owned by **86.69**.
- **It does not attribute the 2026-08-14 improvement**, which is unidentifiable.
- **No production file is modified**, so nothing here needs a restart.

## References

`research_brief_86.47.md` (the three refuted premises, the analysis-day
denominator, the four-way 08-14 confound, the power bound, the reject-inference
argument); `contract_86.108.md` (the synthesis parse-failure population);
`financial_reports.analysis_results` / `.paper_trades`.
