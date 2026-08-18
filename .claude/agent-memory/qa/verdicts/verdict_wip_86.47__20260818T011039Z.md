STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.47
WRITTEN: 2026-08-18T01:10:39Z

# Q/A write-first record -- step 86.47 (drought / funnel census)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status scope, lint, runtime smoke, re-run census
C. Claim auditing (4b) + guard vacuity (4c) + LLM judgment vs the 6 immutable criteria

## Findings log (appended as established)
- (start) qa.md read in full at 01:10Z.
- qa_wip 86.47: attempt_number=1, prior_attempts=0, source_present=true, status ok.
  verdict_history --evidence-only: status=no_rows_for_step, verdicts=(none).
  prior_attempts(0) == ledger rows(0) -> no staleness signal. First attempt.
- IMMUTABLE CMD: `ast.parse(backend/services/autonomous_loop.py)` -> "parsed", EXIT=0. REPRODUCED.
- `python scripts/qa/drought_census_86_47.py` -> CENSUS_EXIT=0. REPRODUCED.
- Git scope: only NEW file for this step = scripts/qa/drought_census_86_47.py (untracked).
  Working tree ALSO has modified production files (backend/services/autonomous_loop.py 21:42,
  backend/api/sovereign_api.py 15:54, 10 frontend files) -- all mtimes 2026-08-17 15:54-22:19,
  i.e. BEFORE the 86.47 work window (research_brief 03:04, contract 03:08, census 03:09).
  Pre-existing from other session work; NOT attributable to 86.47. But live_check §9 asserts
  "No production file was modified at all" without qualification -> see finding F3.

### F1 [MATERIAL] "_path present on 100% of rows" is FALSE -- measured 49.7%
Claim appears in THREE artifacts: contract §4, live_check §2, experiment_results premise 3.
Re-derived by me directly against BigQuery (financial_reports.analysis_results, 580 rows):
  total=580 have_path=288 pct=49.7%  [2025-11-23 .. 2026-08-17]
  first date with _path = 2026-06-11 ; every row before that is NULL
The step's OWN funnel table refutes it: 17 + 221 = 238 rows print as `(unmarked)`, which is
the COALESCE default for a NULL _path. So the artifact contains a 100%-coverage claim and, two
sections later, a table showing 45.2% of counted rows unmarked.
Materiality: the 221-row `A_pre / ok / (unmarked)` cell supplies 100 of the 111 BUYs AND the
pre-break rate p_pre = 100/221 = 0.452 that drives the criterion-6 headline 10^-61.7. That
baseline is drawn ENTIRELY from path-unknown rows, and no artifact discloses it.

### F2 [MATERIAL] normalisation rule mixes populations -- 3 of 26 "trade days" are weekends
census prints: "trade-DAYS per WEEKDAY over [2026-04-26 .. 2026-08-13], both endpoints
inclusive and both are trade days.  26 trade-days / 79 weekdays = 0.3291 per weekday"
Measured from the script's own TRADE_DAYS list:
  WEEKEND trade days = [('2026-04-26','Sun'), ('2026-05-16','Sat'), ('2026-05-17','Sun')]
  window left endpoint 2026-04-26 = SUNDAY
So the numerator counts 3 days that are not members of the denominator's population, and the
stated endpoint is a non-weekday. Weekday-only numerator would be 23/79 = 0.2911.
Criterion 1 demands the normalisation rule be STATED beside every rate; it is stated and it is
wrong-as-applied. Not load-bearing for the P-value (which uses the analysis unit), but it is
the rate criterion 1 names.

### Reproduced EXACTLY against BigQuery (independent re-derivation, not inherited)
I have live ADC BigQuery access from the same .venv the census runs in, so every recorded
figure was re-derived by me, not read.
- FUNNEL: all 8 cells reproduce byte-for-byte (TOTAL 526 analyses / 111 BUYs).
- RJD coverage: paper_trades BUY 19 POPULATED/15 EMPTY (=19/34, 55.9%); SELL 0/32 (0.0%);
  analysis_results POPULATED 18 / EMPTY 511 / NULL 51 (=18/580, 3.1%). All reproduce.
- 26 distinct trade days, identical list. Last trade = 2026-08-13T19:31:19Z DELL BUY -> the
  step's correction of the stale "2026-07-31 NTAP" premise is CORRECT.
- Table locations: financial_reports.analysis_results 580 rows, .signals_log 119 rows;
  pyfinagent_data.analysis_results -> 404 Not found. Step's premise-2 correction CORRECT.
- risk_intervention_log: 0 rows (in pyfinagent_data; not present in financial_reports).
- DAILY_TAIL: all 11 rows reproduce exactly.
- SYNTH_ERROR: 'Failed to parse final report.' path=full n=219 2026-06-11..2026-08-13. Exact.
- 76.7% zero-score post-break (211/275) and avg 205,850 bytes both reproduce exactly;
  zero&failed = 211 (all zero-score post-break rows are synth-failed).
- Criterion-4 premise check: 2026-08-10's two BUYs (HPE, CRWD) are BOTH path=lite. Confirms
  the criterion's own premise and the step's path split addresses it.
Deterministic gates: immutable cmd EXIT=0; census EXIT=0; --sql EXIT=0 (5 query blocks);
ruff F821,F401,F811 over DERIVED scope (git diff + git ls-files --others, 3 files, non-empty
guard satisfied) -> "All checks passed!" exit 0; `import backend.services.autonomous_loop`
OK; GET :8000/api/health -> 200 {"status":"ok"}.
Harness compliance: brief 03:04:55 < contract 03:08:53 < census 03:09:03 < live_check
03:09:43 < experiment_results 03:10:06. Gate envelope: brief_status COMPLETE,
external_sources_read_in_full 14 (>=5), urls_collected 48 (>=10), recency_scan_performed
true, coverage present, gate_passed true. All 6 criteria present VERBATIM in the contract
(exact string match against masterplan.json). masterplan 86.47 status=pending, no
`phase=86.47 result=` cycle header in harness_log -> log-last OK.

### F4 [BLOCKING] the refusal funnel is DERIVABLE; the step reports it UNDERIVABLE
The artifacts report the refusal signal "underivable" and never count refusals.
Measured by me, `financial_reports.analysis_results`, JSON_QUERY (NOT JSON_VALUE -- `judge`
is an OBJECT and JSON_VALUE returns NULL for objects, which is almost certainly how the
author's probe read it as absent; I hit the same trap and corrected it):
  final_synthesis.risk_assessment.judge present: 382/526 since 2026-05-01 (72.6%)
    A_pre 126/251 ; B_post 256/275 (93.1%)
  shape: {"decision": REJECT|APPROVE_REDUCED|APPROVE_HEDGED,
          "reasoning": "<full stated reason>", "recommended_position_pct": N}
  B_post decisions: APPROVE_REDUCED 206, REJECT 42, APPROVE_HEDGED 8, absent 19
THE SILENCE WINDOW (2026-08-14 + 2026-08-17), all 13 analyses, all path=full:
  REJECT pct=0 : PANW, WDAY, HPE, MRVL (08-14); 009150.KS, HPE, NTAP, DELL (08-17)  = 8
  APPROVE_REDUCED pct=2/2/2/3 : STX, NTAP, SNDK, MU ; APPROVE_HEDGED pct=5 : MRVL   = 5
Stated reasons are explicit, e.g. "REJECT new capital in DELL ... the book at 100.0%
Technology across only 2 positions ... against a 60.0% threshold -- a ~40-point breach".
I independently confirmed the premise: financial_reports.paper_positions holds exactly TWO
positions, NTAP and DELL (both Technology).
BINDINGNESS: settings.py:342 default reject_binding=False, BUT settings.py:348 records the
phase-86.74 correction -- nested-first resolution is now UNCONDITIONAL and "decide_trades
driven with a nested REJECT/0% returns no order with BOTH shape_fix and reject_binding OFF
-- verified by executed test". portfolio_manager.py:330-341 reads risk_assessment["judge"]
unconditionally and treats an explicit 0% as no-buy. All 13 window rows are path=full.
=> Criterion 2's (b) reached the gate, (c) refused, (d) with what stated reason are ALL
answerable and ALL absent. And the headline "NO GATE IS AT FAULT -- a gate cannot refuse a
recommendation that was never produced" is asserted over a window in which synthesis was
healthy (0 failures on both days, per the step's own table) and a gate refused 8 of 13 at
0% with a stated portfolio-construction reason.
The step's OWN research brief named the route: line 592 "use `paper_trades.risk_judge_decision`
AND THE JSON BLOB as the [alternative]". It was not executed.
NOT claimed by me: that the gate is mis-calibrated. That needs outcome evidence (criterion 5)
and the step is RIGHT to refuse it. The finding is that the funnel is derivable and uncounted.

### F5 [WARN] criterion 6's null is contaminated by the effect under test
Criterion 6 requires the check "under the null of a HEALTHY funnel". The step's null is
p = 8/275 = 0.0291, and 211 of those 275 rows ARE the failed-synthesis cell the same artifact
identifies as broken and which yields 0 BUYs by construction. Sensitivity (my computation):
  8/275  step's choice (77% broken denominator)          P(0 in 13)=0.681  need=102
  8/64   post-break, ok-synthesis only (healthy)         P(0 in 13)=0.176  need=23
  1/45   post-break, ok-synth, FULL path (path-matched)  P(0 in 13)=0.747  need=134
  100/221 pre-break healthy                              P(0 in 13)=0.0004 need=5
CONCLUSION IS ROBUST under 3 of 4 nulls -- I am not overturning "not anomalous". The defect is
that one null is reported with no sensitivity and no disclosure that 77% of its denominator is
the broken cell. Also: contract line 118 says "~97 analyses (~16 days)" while the shipped
census computes 102 -- a stale carry-over from the brief's 8/262 rate; and "~16 trading days"
is 97/6, not 102/6 = 17.

### F6 [WARN] mis-specified statistic in the shipped deliverable (mutation-proven)
census:220-221 `lp = fn*log(1-p_pre)/log(10)` computes P(**ZERO**) but the label interpolates
`fb`. Mutant M1 (B_post FAILED full BUYs 0 -> 99) prints, byte-identical to control:
  control: "P(0 BUYs in 236 failed analyses | p=0.452) = 10^-61.7"
  M1     : "P(99 BUYs in 236 failed analyses | p=0.452) = 10^-61.7"
i.e. the printed probability does not depend on the count it claims to be the probability of.
Correct only at fb=0. M1 also prints the self-contradicting pair "B_post FAILED full 211 99
46.9%" + "SYNTHESIS FAILED, every era and path: 99 BUYs in 236 analyses" + "by 62 orders of
magnitude", exit 0, no detection.

### F7 [WARN] guard vacuity -- the two listed rerunnable_checks cannot fail
The evidence lists `python scripts/qa/drought_census_86_47.py` and `--sql` as rerunnable
checks and quotes CENSUS_EXIT=0 as verification. The script has ZERO assertions; every figure
is a hardcoded constant.
  M1 (falsify headline funnel cell 0 BUYs -> 99)      -> SURVIVED, exit 0
  M2 (falsify criterion-3 proof 18/580 -> 580/580)    -> SURVIVED, exit 0, and STILL prints
      "=> A funnel keyed on risk_judge_decision would measure its own blindness" at 100.0%
      coverage -- criterion 3's conclusion is a string literal independent of the measurement.
qa.md 4c shapes #1 and #4. Named fix: assert the invariants (failed-cell BUYs == 0; coverage
below a stated bound before printing the blindness conclusion), or make the script query BQ.
The docstring's stated reason for hardcoding -- "rather than holding BigQuery credentials" --
does not hold here: I ran every one of its queries live from the SAME .venv via ADC.

### F8 [NOTE] live_check §8 daily table drops an in-range row
The header says "Every number below comes from a query printed by ... --sql". Q_DAILY
(>= 2026-08-05) returns 11 rows; §8 prints 7, omitting 08-05/06/07 (before the shown range,
fine) AND **2026-08-11 (6 analyses, 1 synth_failed, 1 BUY)**, which is INSIDE the displayed
08-08..08-17 range and is present in the script's own DAILY_TAIL. Reproduced by me. Changes
no conclusion; it is an edited capture in a block labelled as query output (qa.md 4b).

### F9 [NOTE] scope-claim wording
live_check §9 / experiment_results: "No production file was modified at all". True of THIS
step's authorship (verified: its only new file is the untracked scripts/qa/drought_census_86_47.py;
the modified backend/frontend files carry mtimes 2026-08-17 15:54-22:19, hours before the
86.47 work window, and their content is the /reports summary fix + sovereign 1y window).
False of the working tree as an unqualified statement. Scope the claim.
Also live_check §4 quotes "risk_intervention_log has 0 rows" without naming its dataset, in a
section whose own stated rule is that every figure names its table (it is pyfinagent_data).

## Criterion-by-criterion
C1 MET-with-defect (F2): base rate genuinely re-derived and reproduces; rule stated but
   mixes populations (3 weekend days in a per-weekday numerator; Sunday left endpoint).
C2 NOT MET (F4): 1 of 4 required quantities delivered; the other 3 are derivable at 93.1%
   post-break coverage and 13/13 in the silence window, with decisions AND stated reasons.
C3 MET: proof produced first, both tables named, all three figures reproduce exactly.
   (Its "underivable" conclusion is a hardcoded print -- see F7 -- and it is over-extended
   into C2's territory, but C3 as written is satisfied.)
C4 MET-with-defect (F1): split applied everywhere and load-bearing; premise confirmed
   (08-10 BUYs both lite); but "100% of rows" is 49.7%, and the path-blind 221-row cell
   supplies the p_pre baseline undisclosed.
C5 MET: nothing loosened, nothing promoted, no miscalibration claim. (F9 wording only.)
C6 NOT MET as written (F5): null is not a healthy-funnel null; no sensitivity stated;
   plus the mis-specified statistic F6.

## Verdict reasoning
Worst-of-lenses: reproduce lens = CONDITIONAL (everything measured reproduces exactly; two
claims do not); scope-honesty lens = CONDITIONAL (excellent 86.108/86.69 boundaries and an
admirable refusal to attribute the 08-14 recovery, undercut by three overclaims);
correctness lens = FAIL (C2 materially unaddressed on 3 of 4 quantities, and the step's
answer to its own title question is contradicted by 8 binding-shape REJECTs at 0% in the
exact window). min = FAIL.

COMPLETED: 2026-08-18T01:22:30Z
