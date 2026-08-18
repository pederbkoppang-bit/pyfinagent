# experiment_results -- step 86.47

**GENERATE complete for all six criteria.** Verbatim evidence in
`handoff/current/live_check_86.47.md`; this file is the build record.

## What was built

| File | Purpose |
|---|---|
| `scripts/qa/drought_census_86_47.py` | The step's whole deliverable: a re-runnable census that asks the questions in the order that prevents the failure mode this step exists for. `--sql` prints every BigQuery query; `--verify` runs **48 assertions**, counted from the checks that actually ran rather than a literal and exits non-zero if any recorded figure stops holding. |

**No production file was modified.** That is the correct shape for a step whose
first duty is to establish whether a defect exists at all — and it is why
nothing here needs a restart.

## Verbatim verification output

```
$ bash -c 'source .venv/bin/activate && python -c "import ast;ast.parse(open(\"backend/services/autonomous_loop.py\").read());print(\"parsed\")"'
parsed
EXIT=0

$ python scripts/qa/drought_census_86_47.py
EXIT=0

$ python scripts/qa/drought_census_86_47.py --verify
OK: all 48 invariants hold
VERIFY_EXIT=0

$ uvx ruff check --select F821,F401,F811 --no-cache scripts/qa/drought_census_86_47.py
All checks passed!    RUFF_EXIT=0
```

## The result, in two phases

**PHASE 1 (2026-06-11 .. 2026-08-13)** — BUY *supply* was suppressed by
`final_synthesis.error = "Failed to parse final report."`: **0 BUYs in 236
failed analyses**, deterministic across every era and path. No gate involved,
because a gate cannot refuse what never arrives. This is step 86.108's
population; 86.47 hands off.

**PHASE 2 (from 2026-08-14)** — synthesis is HEALTHY (0 failures both days),
and still 0 BUY-class recommendations in 13 analyses — **while the risk judge
refused 8 of 13 at 0%** on sector concentration (the book is DELL + NTAP, both
Technology = 100% against a 60% cap, confirmed against `paper_positions`).

**THE GATE IS NOT EXONERATED** — it is active and refusing. But the
counterfactual "a ground that would bind any BUY that did arrive" is
**WITHDRAWN**: the one BUY that did arrive was refused at 0% and a trade
executed anyway. The no-BUY-arrived reading holds only inside the 2-day window.

## Three premises of the step text, refuted by measurement

1. **"The last trade was 2026-07-31 (NTAP)."** It is **2026-08-13, a DELL BUY**.
   Every "days of silence" figure downstream of that endpoint is wrong.
2. **`analysis_results` / `signals_log` are in `pyfinagent_data`.** They are in
   **`financial_reports`**; the stated dataset returns `Table not found`.
3. **"The lite/full split may be underivable."** It IS derivable — but only
   from **2026-06-11**, when `$._path` first appears. All-time coverage is
   **288/580 = 49.7%**; from that epoch, 100%. (Cycle 1 asserted a flat "100%"
   here and in two other artifacts. It was wrong, and the consequence matters:
   the pre-break baseline cell is path-UNKNOWN.)

## Corrections to my own work, caught BEFORE the verdict

- **I measured the silence in WEEKDAYS (3) and got P=37.3%.** The opportunity
  unit is an **ANALYSIS**; the correct figure is 13 analyses and P=0.681. The
  research gate caught it. Measuring in the wrong unit overstated the evidence
  in the direction of "something is broken" — the exact bias that sank the two
  prior steps.
- **The census printed a hardcoded "~48 orders of magnitude" beside a computed
  `10^-61.7`.** That whole statistic was **deleted** in cycle 2, not merely
  derived: the cycle-1 Q/A showed its label interpolated the BUY count while the
  formula computed P(zero), so a mutant's output was byte-identical to the
  control. Neither figure appears in the shipped census.

## Criterion-by-criterion

| # | Result |
|---|---|
| 1 | Base rate **23 weekday trade-days / 79 weekdays = 0.2911**, normalisation rule stated and the 3 weekend trade-days named and EXCLUDED. Last trade re-derived as 2026-08-13 (the step text's 2026-07-31 is stale). Silence = 2 analysis days / 13 analyses. |
| 2 | **Two populations, because they answer different halves.** (a) The BUY×GATE crossing over the post-break era — **8 BUY-class recommendations, 1 reached the recorded gate, 1 refused at 0%**; the other 7 are lite-path with NO `risk_assessment` at all. (b) The window's judge verdicts — 13/13 present, 8 REJECT at 0%, 5 approvals at 2-5% — with the stated reason and the book's composition confirmed against `paper_positions`. Every query printed by `--sql`. |
| 3 | **Proved BEFORE any funnel keyed on that column**: `risk_judge_decision` is 3.1% populated in `analysis_results`, 55.9% for BUYs and 0% for SELLs in `paper_trades`, and `risk_intervention_log` (dataset `pyfinagent_data`) has 0 rows. That column is reported unusable; the refusal funnel is derived from `final_synthesis.risk_assessment.judge` via **JSON_QUERY**. |
| 4 | Every count split by `$._path`, **with its epoch stated**: 49.7% all-time, 100% from 2026-06-11. Load-bearing: 36.8% BUY on lite-with-ok-synthesis vs 0.0% on full-with-failed, and the healthy baseline cell is path-UNKNOWN. |
| 5 | Nothing loosened, nothing promoted, no `.env`, no production file authored. **No miscalibration claim** — that needs OUTCOME evidence this step does not have; the 8 REJECTs are reported as what the gate SAID. |
| 6 | **Four-null sensitivity table** including the healthy-funnel null. The conclusion flips on the null: supply has NOT returned to the pre-break rate (P=0.0004), while 13 analyses cannot distinguish "the cap binds" from "the rate is low" (102 needed). |

## Scope honesty

- **86.108 owns the synthesis parse failure**; this step measures the
  consequence and hands off. It does not claim 86.108's fix.
- **86.69 owns the empty-HOLD persistence question**; the 76.7% zero-score
  figure corroborates it independently but does not re-litigate it.
- **86.38 and 86.41 are not re-opened.**
- **The 2026-08-14 recovery is deliberately UNATTRIBUTED.** Four changes landed
  together and are not identifiable; 13 analyses cannot confirm recovery either
  way. The artifact states the power requirement instead of declaring victory.
- **This step proposes no code change.** The census computes the power bounds;
  they are not restated here, because a number restated in prose is a number
  that can go stale.

## Cycle 2 -- response to the FAIL (`wf_acfe2459-948`)

**The FAIL was correct and the finding was real.** Five defects, all closed.

1. **Criterion 2 was materially unaddressed, and my "underivable" was an
   unverified negative.** `JSON_VALUE` returns NULL for a JSON **object**, so my
   probe on the judge read 0/526 and looked like absence. `JSON_QUERY` reads
   382/526, 256/275 post-break, **13/13 in the silence window**. The refusal
   funnel is now produced in full: 8 REJECT at 0%, 5 approvals at 2-5%, with the
   stated reason and the book's actual composition confirmed against
   `paper_positions`. *(This trap is in this project's own memory. I cited it
   earlier in the same session and then walked into it.)*
2. **Criterion 6's null was 77% the broken population.** Replaced with a
   four-row sensitivity table including the healthy-funnel null. The conclusion
   flips on the null, and both halves are now stated: supply has NOT returned to
   the pre-break rate (P=0.0004), while 13 analyses cannot distinguish "the
   sector cap binds" from "the rate is low" (102 needed).
3. **Criterion 1's normalisation mixed populations** — 26 trade-days (3 of them
   weekend) over 79 weekdays. Corrected to 23/79 = 0.2911, with the three
   weekend days named.
4. **"`_path` present on 100% of rows" was false** in three artifacts: 49.7%
   all-time, 100% only from 2026-06-11. The consequence is now stated — the
   healthy baseline cell is path-UNKNOWN, so the healthy BUY rate cannot be
   attributed to a path.
5. **The census had zero assertions**, so two Q/A mutants survived at exit 0.
   It now carries **13 invariants** (`--verify`), including the central claim,
   and the mis-specified statistic whose output was byte-identical under mutation
   is deleted.

Also fixed: the dropped 2026-08-11 daily row and the un-named dataset on
`risk_intervention_log`.

**Cycle 2 shipped this section claiming the contract's stale "~97" was fixed
while it was still there** — I edited a different occurrence in the same file
and did not re-grep. The cycle-3 Q/A caught it. It is now removed, and the
contract restates no power figure at all.

## Cycle 3 -- response to the CONDITIONAL (`wf_775cfbb1-5ee`)

All 6 criteria MET, and the evaluator re-derived **every** figure from BigQuery
with 100% reproduction — including a JSON_VALUE control returning 0/526 that
independently confirms the mechanism behind cycle 1's error. Two WARN findings,
both closed.

1. **A past-tense remediation claim of mine was FALSE.** Cycle 2 wrote *"Also
   fixed: the contract's '~97' against the census's 102"* while
   `contract_86.47.md:119` still carried it — I edited a different occurrence in
   the same file and never re-grepped. Now removed, and the contract restates no
   power figure at all. The related "~48 orders of magnitude ... now derived"
   line is corrected too: that statistic was **deleted**, not derived.
2. **9 of 15 mutants survived cycle 2's guards.** The gap was the CONCLUSIONS,
   not the assertions — they were prose printed regardless of the numbers above
   them, `n_an` was never cross-checked against `len(WINDOW)`, and
   `N_INVARIANTS` was a literal that could not see a deleted guard. Fixed at the
   class level: derived count, conditional conclusions, whole-tuple daily
   pinning, a partial-coverage assertion on the BUY column and a disagreement
   assertion across the two tables. Re-probed: **8 of 8 previously-surviving
   mutants now KILLED**, with a whitespace null control correctly inert.

Its three NOTE items are accepted and not re-litigated: the `reasoning` text has
no printed query (I wrote that query ad hoc and the evaluator verified the claim
holds 8/8); the research brief's own "100%" lines are a dated gate artifact and
are annotated rather than rewritten; and criterion 1's answer is given in
analysis units rather than trade units, which the criterion permits.

## Cycle 4 -- response to the CONDITIONAL (`wf_89107a13-3d6`)

Five of six criteria MET on 100%-reproduced evidence. Criterion 2 was NOT met
and the diagnosis was exact.

**1. I measured the BUY→gate stage over a window with ZERO BUY-class
recommendations.** The stage the criterion names had no members, so "REACHED
THE GATE 13/13" described HOLD rows, not BUYs. Corrected: the crossing is now
measured over the post-break era, where **8 BUY-class recommendations exist —
7 lite-path with no `risk_assessment` at all (including the 2026-08-10
CRWD+HPE pair the criteria cite by name), and 1 full-path (2026-08-13 DELL)
that the judge REJECTed at 0%.** That is a sharper form of criterion 4's point
than the supply funnel gives: the lite path does not record a risk verdict at
all.

**2. My counterfactual is WITHDRAWN.** "A ground that would bind any BUY that
did arrive" is falsified by the single observed instance — the one BUY that
arrived was refused at 0% and a DELL BUY of 4.806437 shares executed 53 minutes
later. A window-scoped negative was stated as the general answer, which is the
same error class the FAIL was for, one level up.

**3. A MONEY-PATH OBSERVATION, handed to 86.74 and NOT diagnosed here.** Four
BUY trades executed while a REJECT verdict was on record; **three carry
`REJECT` in the trade row itself** (2026-06-02 HPE, 2026-06-03 DELL, 2026-06-09
066570.KS, all `reason='swap_buy'`). This is step 86.74's subject — a
falsy-zero check inversion — already filed and parked on an operator decision.
Recorded so 86.74 has the measurement; criterion 5 forbids this step touching a
gate, and no mechanism is asserted.

**4. Four mutants still survived on bounds-only guards.** `jp > 0` accepted
382→5 and 256→5 while the artifact printed "It IS derivable"; the two FUNNEL
cells carrying criterion 4's load-bearing 36.8%-vs-2.2% contrast were unguarded
entirely. Now: judge coverage must be a **majority**, the lite/full rates are
bounded, and the **contrast itself** is asserted. 42 invariants.

## Cycle 5 -- response to the FAIL (`wf_9d469015-800`)

**The FAIL was correct and it caught a real analytical error, not bookkeeping.**

**1. I collapsed two INDEPENDENT gates into one.** Cycle 4 wrote that the one
BUY which reached the gate was refused "on the same sector-concentration ground
as §4". The judge's recorded `reasoning` names a different driver:
`projected_dd_over_cap`, **projected_dd 22.5% vs a 10% cap**, and files
concentration explicitly under *"CORROBORATING DOWNSIDE (independent of the
gate)"*. Corrected — and the corrected version is a **better** answer: two
independent gates bind, and the drawdown cap is the more general of the two
because at ~0.5× annualized vol it *"trips for ANY realized vol above ~20%"*.
Manufacturing a single cause is the exact failure this step exists to prevent,
and I committed it in the section added to fix the previous criterion-2 miss.

**2. The claim had no predicate.** `--sql | grep -c reasoning` was 0 — none of
the printed queries selected the field the claim was about. `Q_BUY_CROSSING`
now selects `reasoning`, so the stated-reason element reproduces.

**3. A five-file protocol breach of mine.** Cycles 2 and 3 returned verdicts
that I recorded in the ledger but never transcribed into
`evaluator_critique_86.47.md`. The standing rule is ledger row **and** verbatim
transcription in the same turn; I did the first and not the second, twice. All
four verdicts are now transcribed and the gap is disclosed at the top of that
file.

**4. Guard and citation fixes.** The criterion-6 post-break-null sentence was
unconditional prose that survived a FUNNEL cell crossing 0.05 — now conditional.
`live_check` cited an `if _p0 < 0.05 ...` line that **did not exist** in the
deliverable; corrected, and its cycle-3 matrix is now labelled as predating the
cycle-4/5 guard families rather than implying it covered them. Added guards:
the FUNNEL↔crossing tie, the 7-lite/1-full split by count, a path-coverage
band, and the two-gates distinction. **48 invariants.**
