# live_check -- step 86.47 (2026-08-18; exits unpiped)

**All figures AS OF 2026-08-18.** `analysis_results` grows daily, so a figure
without its as-of date is not reproducible. Every number is produced by a query
printed by `python scripts/qa/drought_census_86_47.py --sql`, and every one is
guarded by an assertion (`--verify`).

## 0. Cycle 1 FAILED, and the finding was correct

The cycle-1 verdict (`wf_acfe2459-948`) was **FAIL**, on a claim that was
genuinely wrong: this artifact said *"NO GATE IS AT FAULT -- a gate cannot
refuse a recommendation that was never produced"* about a window in which
**synthesis was healthy and the risk judge refused 8 of 13 analyses at 0%**.

The mechanism of the miss is worth stating because it is in this project's own
memory: **`JSON_VALUE` returns NULL for a JSON object.** The judge verdict is
an object, so a `JSON_VALUE` probe on it reads 0/526 and looks like absence. I
reported the refusal signal "underivable" on the strength of that probe.
`JSON_QUERY` reads 382/526. **An unverified negative is not a permitted
outcome**, and criterion 3's escape hatch is scoped to `risk_judge_decision`
alone — it never waived criterion 2.

Four further findings are addressed in §2, §4, §5 and §8.

## 1. Immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -c "import ast;ast.parse(open(\"backend/services/autonomous_loop.py\").read());print(\"parsed\")"'
parsed
EXIT=0
```

## 2. Criterion 1 -- the base rate, with a normalisation rule that holds

```
NORMALISATION: WEEKDAY trade-days / weekdays over [2026-04-26 .. 2026-08-13] inclusive.
23 weekday trade-days / 79 weekdays = 0.2911 per weekday
3 of the 26 trade-days are WEEKEND days
(2026-04-26 Sun, 2026-05-16 Sat, 2026-05-17 Sun) and are EXCLUDED.
LAST TRADE = 2026-08-13 (DELL BUY). The step says 2026-07-31 (NTAP): STALE.
THE SILENCE: 2 analysis DAYS / 13 analyses (2026-08-14, 2026-08-17).
```

**Cycle 1 divided all 26 trade-days by 79 weekdays and called it a weekday
rate** (0.3291). Three of those numerator days are weekends and the stated left
endpoint is a Sunday — a numerator must live in its denominator's population.
Corrected to 23/79 = **0.2911**.

The step's headline premise is also **stale**: the last trade is 2026-08-13, a
DELL BUY, not 2026-07-31 (NTAP). Every "days of silence" figure downstream of
that endpoint is wrong.

## 3. Criterion 3 -- proved BEFORE any funnel keyed on that column

```
financial_reports.paper_trades BUY           19/  34 =  55.9%
financial_reports.paper_trades SELL           0/  32 =   0.0%
financial_reports.analysis_results           18/ 580 =   3.1%
`risk_intervention_log` (dataset pyfinagent_data) has 0 rows.
```

Too sparse to key a funnel on. **The two tables disagree and both are true** —
one records executed trades, the other analyses — so every figure names its
table. (Cycle 1 quoted the `risk_intervention_log` figure without naming its
dataset, in a section whose own rule is that every figure names its table.)

## 4. Criterion 2 -- the refusal funnel, which cycle 1 wrongly called UNDERIVABLE

```
judge present, since 2026-05-01          382/ 526 =  72.6%
judge present, post-break (>=06-15)      256/ 275 =  93.1%
judge present, the silence window         13/  13 = 100.0%
```

**THE SILENCE WINDOW, all 13 rows, all `path=full`:**

```
date         ticker     rec    judge             pct
------------------------------------------------------
2026-08-14   HPE        HOLD   REJECT               0
2026-08-14   MRVL       HOLD   REJECT               0
2026-08-14   NTAP       HOLD   APPROVE_REDUCED      2
2026-08-14   PANW       HOLD   REJECT               0
2026-08-14   STX        HOLD   APPROVE_REDUCED      2
2026-08-14   WDAY       HOLD   REJECT               0
2026-08-17   009150.KS  HOLD   REJECT               0
2026-08-17   DELL       HOLD   REJECT               0
2026-08-17   HPE        HOLD   REJECT               0
2026-08-17   MRVL       HOLD   APPROVE_HEDGED       5
2026-08-17   MU         HOLD   APPROVE_REDUCED      3
2026-08-17   NTAP       HOLD   REJECT               0
2026-08-17   SNDK       HOLD   APPROVE_REDUCED      2

REACHED THE GATE 13/13   REFUSED 8 at 0%   APPROVED 5 at 2-5%
```

**Stated reason**, verbatim from the judge's `reasoning`: a
portfolio-construction veto on sector concentration — *"REJECT is a
portfolio-construction veto, not a bearish call"*, *"the proposed remedy is
arithmetically incapable of curing the breach it concedes"*.

**Independently confirmed against `paper_positions`:** the book holds exactly
two positions, **DELL and NTAP, both Technology** — 100% concentration against
a 60% cap. The judge's ground is factually correct.

## 5. Criterion 4 -- the path split, with its epoch

```
PATH COVERAGE 288/580 = 49.7% all-time, 100% only from 2026-06-11.
```

**Cycle 1 asserted "present on 100% of rows" in three artifacts.** It is 49.7%
all-time; the field did not exist before 2026-06-11. This matters beyond
bookkeeping: the 221-row `A_pre/ok/(unmarked)` cell supplies 100 of the 111
BUYs and the entire healthy baseline, so **the healthy BUY rate cannot be
attributed to a path.**

## 6. Criteria 2+4 -- the SUPPLY funnel, split by synthesis AND path

```
era      synthesis path            n  BUYs    BUY%
----------------------------------------------------
A_pre    FAILED   (unmarked)     17     0    0.0%
A_pre    FAILED   full            8     0    0.0%
A_pre    ok       (unmarked)    221   100   45.2%
A_pre    ok       lite            3     3  100.0%
A_pre    ok       full            2     0    0.0%
B_post   FAILED   full          211     0    0.0%
B_post   ok       full           45     1    2.2%
B_post   ok       lite           19     7   36.8%

SYNTHESIS FAILED: 0 BUYs in 236 analyses, every era and path.
```

The split is load-bearing: post-break, lite-with-ok-synthesis runs at 36.8%
while full-with-failed runs at 0.0%. A merged census averages a working path
with a broken one.

## 7. Criterion 6 -- base rate under a HEALTHY-funnel null, WITH SENSITIVITY

```
null population                                                    p   P(0 in 13)
----------------------------------------------------------------------------------
all post-break  (CONTAMINATED: 211/275 = 77% is the failed cell)  0.0291    0.6813
post-break, synthesis ok                                          0.1250    0.1762
post-break, ok + full  (matches all 13 window rows)               0.0222    0.7467
pre-break, synthesis ok  (the HEALTHY-funnel null)                0.4558    0.0004
```

**Cycle 1 reported only the first row** — a null whose denominator is 77% the
very population it had just called broken. Criterion 6 asks for the null of a
**healthy** funnel; that is the last row.

**The conclusion depends entirely on the null, and both halves matter:**
- Under the **healthy** null the silence IS surprising (P = 0.0004). Only 5
  analyses are needed to reach that bar, so 13 is already more than enough:
  **BUY supply has NOT returned to the pre-break rate.**
- At the **current** post-break rate (p = 0.0291), **102** analyses would be
  needed before a zero-run said anything — so 13 cannot distinguish "the sector
  cap now binds" from "the rate is simply low".

## 8. THE ANSWER, in two phases

```
PHASE 1  2026-06-11 .. 2026-08-13  --  'Failed to parse final report.'
         219 rows, path=full. Supply suppressed: 0 BUYs in 236.
         No gate involved: a gate cannot refuse what never arrives.
         This is step 86.108's population. 86.47 hands off.
PHASE 2  from 2026-08-14  --  synthesis HEALTHY (0 failures both days).
         Still 0 BUY-class recommendations in 13 analyses, AND the
         risk judge refused 8 of 13 at 0% on sector concentration.
```

**THE GATE IS NOT EXONERATED** — it is active and refusing. But the
counterfactual attached to that in cycles 2-3 is **WITHDRAWN**; see §8b. The
no-BUY-arrived reading holds only inside the 2-day window.

The full daily tail, **all 11 rows** the query returns (cycle 1's artifact
silently dropped 2026-08-11 from inside its own displayed range):

```
2026-08-05  6 analyses  3 synth_failed  0 buys
2026-08-06  5           2               0
2026-08-07  5           4               0
2026-08-08  6           6               0
2026-08-09 12           7               0
2026-08-10  6           0               2
2026-08-11  6           1               1
2026-08-12  6           6               0
2026-08-13  6           1               1
2026-08-14  6           0               0
2026-08-17  7           0               0
```

## 8b. Criterion 2's REAL population -- the BUY x GATE crossing

**Cycles 2-3 measured the BUY→gate stage over the 2-day silence window, which
contains ZERO BUY-class recommendations.** The stage the criterion names had no
members. Over the post-break era it does:

```
date         ticker     path   reached gate  judge     pct
----------------------------------------------------------
2026-07-09   AMD        lite   no            -           -
2026-07-09   MU         lite   no            -           -
2026-07-20   PANW       lite   no            -           -
2026-07-31   NTAP       lite   no            -           -
2026-08-10   CRWD       lite   no            -           -
2026-08-10   HPE        lite   no            -           -
2026-08-11   NTAP       lite   no            -           -
2026-08-13   DELL       full   YES           REJECT      0

BUY-class recommendations: 8   REACHED the gate: 1   REFUSED: 1 at 0% (driver: projected_dd_over_cap, NOT the sector cap)
```

**ITS STATED REASON IS A DIFFERENT GATE, and that is the finding.** Cycle 4
wrote that the DELL refusal rested "on the same sector-concentration ground as
§4". The judge's own recorded `reasoning` says otherwise:

> *"DECISION DRIVER — LIVE GATE VETO (verified, not narrative). I ran the
> composite veto chain directly (mcp pyfinagent-risk evaluate_candidate) …
> Result: **vetoed=true, reason=projected_dd_over_cap, projected_dd 22.5% vs a
> 10% cap** [INTERNAL risk-gate]. The projected-DD formula is ~0.5x annualized
> vol, so **the veto trips for ANY realized vol above ~20%**."*

and it files concentration separately, under *"CORROBORATING DOWNSIDE
(independent of the gate)"*.

So **TWO INDEPENDENT gates bind in the post-break era** — a portfolio **sector**
cap (§4) and a projected-**drawdown** cap (here) — and the drawdown one is the
more general: at ~0.5× annualized vol it trips for any name above ~20% realized
vol, which is most of a technology book. Collapsing them into one was the
single-cause manufacturing this step exists to prevent, committed by this step,
and the corrected version is a *better* answer to "why did the book stop
trading" than the one it replaces.

The query now SELECTs the `reasoning` it makes a claim about; cycle 4 asserted a
ground its own printed predicate could not produce.

The other **7 are lite-path with NO `risk_assessment` at all** — they never
reached the recorded gate. That includes the 2026-08-10 CRWD+HPE pair which
criteria 2 and 4 both cite by name, and it is a sharper form of criterion 4's
point than the supply funnel gives: the lite path does not merely score
differently, **it does not record a risk verdict at all.**

**THE WITHDRAWN CLAIM.** Cycles 2-3 wrote that the gate refuses "on a ground
that would bind any BUY that did arrive". That counterfactual is **falsified by
the single observed instance**: the one BUY that did arrive was refused at 0%
and a trade executed anyway (§8c). The no-BUY-arrived reading is true **only
inside the 2-day window**, and is contradicted one day earlier by this step's
own corrected last-trade endpoint — which sits in this step's own daily table
with `buys=1`.

## 8c. HANDED TO STEP 86.74 -- measured facts, NO mechanism asserted

```
created_at             ticker           qty reason           risk_judge_decision
--------------------------------------------------------------------------------
2026-06-02T19:18:58Z   HPE         4.402443 swap_buy         REJECT
2026-06-03T19:05:19Z   DELL        0.581563 swap_buy         REJECT
2026-06-09T18:12:39Z   066570.KS   1.468448 swap_buy         REJECT
2026-08-13T19:31:19Z   DELL        4.806437 new_buy_signal   (empty)
```

**Three carry `REJECT` in the trade row itself** — a recorded field, not an
inference. The fourth is 2026-08-13 DELL: that day's analysis was a full-path
BUY the judge **REJECTed at 0%**, and a DELL BUY of 4.806437 shares executed
**53 minutes later** with `reason='new_buy_signal'`.

**This step asserts no mechanism and changes nothing.** It is the subject of
step **86.74** — "the risk judge REJECTED DELL at 0% and the book bought it, a
falsy-zero check inversion" — which is already filed and parked on an operator
decision. Recorded here so 86.74 has the measurement. Criterion 5 forbids this
step touching a gate, and the money-path question is 86.74's to answer.

## 9. The census ASSERTS -- and cycle 2's assertions were not enough either

```
$ python scripts/qa/drought_census_86_47.py --verify
OK: all 48 invariants hold
VERIFY_EXIT=0
```

**Three rounds of this, and the lesson sharpened each time.**

*Cycle 1* had **zero** assertions; a Q/A mutated the failed-cell BUY count from
0 to 99 and it exited 0, still printing its conclusions.

*Cycle 2* added 13 — and a Q/A refuted the docstring's "every recorded figure is
now guarded" with a known-member recall test over the constants: **9 of 15
mutants survived** at exit 0 while `--verify` printed OK. The gap was not the
guards but the **conclusions**: they were prose, printed regardless of the
numbers above them. Falsifying the healthy-null cell still printed *"the silence
IS surprising ... 13 is already MORE than enough"* beside P=0.7928 and a
168-analysis requirement. And `N_INVARIANTS` was a **literal 13** against 14 real
`_check` calls, so `--verify` reported a count it had never measured and could
not notice a guard being deleted.

*Cycle 3* fixes the class, not the instances:
- the invariant count is **derived** from the checks that actually ran;
- the conclusions that depend on a computed value are **conditional** on it —
  `if n_an >= need_healthy: ... else: ...`, the sparsity ternary, and (added in
  cycle 5) the post-break-null sentence, which a Q/A showed was unconditional
  prose that survived a FUNNEL cell crossing 0.05. **A prior revision of this
  bullet cited an `if _p0 < 0.05 ...` line that was not in the deliverable** —
  a claim about the code's own mechanism citing code that did not exist;
- `n_an` is cross-checked against `len(WINDOW)` — mutating a silence-day row had
  corrupted all four criterion-6 probabilities while the header still read
  "13 rows";
- the daily table is pinned as **whole tuples**, not just dates, because the
  artifact prints every cell of it as fact;
- the `risk_judge_decision` guard now asserts BUY coverage is **partial**
  (`0 < p < total`) and that the two tables **disagree**, because a bounds-only
  check accepted 34/34 — which would have turned "the column is unusable" into
  "the column is complete" while still printing the blindness conclusion.

Re-probed after the CYCLE-3 fix, with a whitespace-only null control that
correctly stays inert. **This matrix predates the cycle-4 and cycle-5 guard
families** (judge-coverage majority, the lite/full rates and their contrast, the
five BUY-crossing guards, the two-gates guards); those carry no mutation
evidence here and a later Q/A found four survivors among them, since fixed:

```
CONTROL (as of cycle 3)          rc=0  OK: all 34 invariants hold
M7  DAILY_TAIL 08-11 counts      rc=1  KILLED
M8  SYNTH_ERROR falsified        rc=1  KILLED
M9  RJD BUY -> 34/34             rc=1  KILLED
M10 SECTOR_CAP 60 -> 5           rc=1  KILLED
M11 POSITIONS tickers            rc=1  KILLED
M12 healthy-null cell            rc=1  KILLED
M13 judge coverage post-break    rc=1  KILLED
M14 daily 08-17  7 -> 70         rc=1  KILLED
NULL (whitespace only)           rc=0  SURVIVED  (correctly inert)
```

Cycle 1 listed this script as a "re-runnable check" while it contained **zero
assertions**; the Q/A mutated the failed-cell BUY count from 0 to 99 and it
still exited 0, still printing its conclusions. The invariants now include the
central claim (`failed_synthesis_yields_zero_buys`), the sparsity bound that
licenses the blindness conclusion, the derivability of the judge, the
no-BUY-arrived reading, and the path-coverage correction — each with a message
saying what must be re-derived if it fails.

Cycle 1 also mis-specified a printed statistic: it interpolated the BUY count
into a label while the formula computed P(zero), so the mutant's output was
byte-identical to the control. That statistic was **deleted**, not repaired.

## 10. Criterion 5 -- nothing loosened, nothing promoted

No gate, threshold or risk parameter changed. No flag promoted. No `.env`
written. **The only file this step authored is `scripts/qa/drought_census_86_47.py`.**
Nothing needs a restart.

*(Scoping note: "no production file was modified" is true of this step's
authorship, not of the working tree — a peer session's `sovereign_api.py` and
`autonomous_loop.py` edits carry 2026-08-17 15:54–22:19 mtimes, hours before
this step's window, and are unrelated.)*

**No miscalibration claim is made.** The gate found that counting REJECTs
cannot show it (r = -0.032), and criterion 5 requires OUTCOME evidence — what
the refused trades would have returned — which this step does not have. The 8
REJECTs are reported as *what the gate said*, not as evidence it said it wrongly.

## 11. What this step must NOT conclude

- **The 2026-08-14 improvement is NOT attributable.** Four changes landed
  together — sonnet-4-6 → sonnet-5, risk columns populating, zero-scores
  ceasing, the recommendation default moving to Hold.
- **86.108 owns the synthesis parse failure.** This step measured the
  consequence and hands off; it does not claim 86.108's fix.
- **86.69 owns the empty-HOLD persistence question.** The 76.7% zero-score
  figure corroborates it independently without re-litigating it.
- **86.38 and 86.41 are not re-opened.**
