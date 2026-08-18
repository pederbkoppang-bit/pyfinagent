# PARK + ESCALATION -- step 86.69 -- OPERATOR DECISION REQUIRED

Written 2026-08-17T20:06Z by Main after the first evaluation returned
CONDITIONAL (`wf_69f2ae7c-21d`). **Not a pass, not a fail, and the step is not
flipped.** Attempt 1 of 5 -- budget is NOT the reason for parking.

## Why park rather than iterate

Of the four unmet criteria, **two cannot be satisfied by any amount of further
work tonight**, and iterating would spend attempts against an empty population.

## What IS met

**C1 (cause), C2 (fabrication sites), C6 (no gate loosened), C8 (mutation)** --
all MET on the evaluator's own re-derivation. On C8 it ran a WIDER matrix than
Main's: 8 cells including two fixture cells, control 7/7 green first, **all
eight killed, no survivor**, `autonomous_loop.py` sha256 unchanged after.

The in-force chain reproduced exactly: pid 41635, ELAPSED 05:55:47 read at
19:53:03Z -> start 13:57:16Z, after the 13:06:04Z env write; the loader reports
`paper_synthesis_integrity_enabled=True` and
`paper_position_recommendation_fix_enabled=False`, so the brief's sequencing
hazard is respected.

## What is NOT met, and why each is or is not fixable

### C4 + C5 -- STARVED. The armed guard was never entered.

The flag guards the **parse-failure** branch. Tonight's cycle had **zero**
parse failures: `final_synthesis.error` NULL on every row, all rows
`_path=full`, zero `Failed to parse final report`. A population that cannot
contain the condition the guard governs is not a measurement of that guard.

Worse for the comparison, the pre-arm days were already clean -- **2026-08-10
and 2026-08-14 were both 0/6 zero-score with the flag OFF** -- so the post-arm
0/N sits inside the pre-arm distribution. On the C5 half tonight is *worse*
than several pre-arm days (0 buys vs 2 on 08-10).

**These become satisfiable only on a cycle where the full path actually fails.**
No further evaluation can manufacture that.

### C4 -- the evaluator ALSO explained something Main could not

Main reported the frozen baselines as "not reproducing, cause unexplained".
The evaluator found the cause and it is **not data loss**: the two readings use
**different regime boundaries**. The audit-basis partition is
`PRE <= 06-12 / POST >= 06-15`; the published query uses
`PRE <= 06-10 / POST >= 06-11`. Cut at 06-12 the PRE side is **251 rows / 95
zero / 37.8%** -- the frozen baseline exactly. The entire 13-row delta is
`2026-06-11..06-12`, itself at 61.5%, sitting between the two regimes and
independently corroborating the corrected 06-11 break date.

**This supersedes Main's "unexplained PRE shrink in a closed historical
window".** The window was never the same window.

The evaluator also measures **n=7**, not 6: DELL landed at 19:46:09Z, after
`experiment_results` was written at 19:41:57Z.

### C3 -- FIXABLE, and the evaluator did the work

Main discharged the consumer half with three asserted bullets. The criterion
requires the consumer set to be **derived**. The evaluator derived it:
`signal_attribution.py:185` yields `"NONE"` not `"HOLD"` for a present-but-None
key (the `or`-escape fires only on an ABSENT key); NULL is in neither
`_BUY_RECS` nor `_DOWNGRADE_RECS`; `_fold_degraded_for_trading` (`:2772`,
called by the `return` at `:1254`) removes `_degraded` rows before
`decide_trades`, which also averts the `portfolio_manager.py:353/430`
`.get("final_score", 0) -> None` sort hazard.

**This is a one-edit fix** -- transcribe that derivation into
`experiment_results_86.69.md`. Deliberately NOT done tonight, because it cannot
be evaluated without also re-opening C4/C5, which are starved.

### C7 -- a genuine, already-disclosed tension

The criterion says *"NO flag is promoted and NO .env is written by this
step"*. The flag WAS promoted and `backend/.env` WAS written. Mitigating and
recorded from the start: it was numbered ASK-1, the operator answered
**"Yes -- arm it"** and **"Now"** verbatim in-session, the token
`ARM-SYNTHESIS-INTEGRITY-86.69` is in `pending_tokens.json`, and the
pre-tool-use danger hook **blocked the write twice** until the token existed.

The evaluator's reading: the criterion prescribes the numbered ask as the
*discharge*, not as a precondition for executing. **Only the operator can
resolve this** -- it is a question about their own instruction.

## Two items no artifact stated, surfaced by the evaluator

1. **Empty summaries are 100% post-arm.** All 7 post-arm rows carry
   `summary_len=0` (empty-summary among scored rows: PRE 29/151, POST 40/62,
   POST_ARM 7/7). So the *empty* half of the masterplan's row signature is
   fully present post-arm even though the *zero-score* half is gone.
   **A peer session is already fixing this**: an uncommitted edit to
   `_persist_analysis` landed at 19:42:56Z making the full path persist
   `final_synthesis.final_summary` instead of the lite-path-only
   `risk_assessment.reason`. Not this step's work and not in its commit.
2. **The tree was not frozen during EVALUATE.** That same peer edit to
   `backend/services/autonomous_loop.py` -- the criterion-3 persistence
   boundary -- landed 7 seconds after the spawn. A status flip would have
   swept an unreviewed money-path change into a phase-86.69 commit via
   `git add -A`. **Main is not flipping, and any commit uses an explicit
   pathspec.**

## What the operator must decide

1. **PARK and let C4/C5 accrue** (Main's recommendation). They satisfy
   themselves on the first cycle where synthesis actually fails; the flag is
   armed and in force, so that is now a matter of waiting.
2. **SPLIT**: close C1/C2/C6/C8 and re-file the measurement as its own step.
3. **Rule on C7** -- whether an operator-token action inside a step satisfies
   "no flag is promoted BY THIS STEP". This recurs and a ruling would settle it.
4. Authorise the one-edit C3 fix to land without re-opening the measurement.

## Cross-reference

`experiment_results_86.69.md`, `live_check_86.69.md`,
`evaluator_critique_86.69.md` (the verdict, verbatim),
`escalation_86.74_starved_criterion.md` (the sibling starved-criterion park),
and `contract_86.108.md` (owns the 2,859 agent-level parse failures, which are
a different class from this step's synthesis-level guard).

---

## OPERATOR DECISION RECORDED -- 2026-08-18

Operator granted general permission to proceed on parked decisions
("you have my promission", 2026-08-18, verbatim). Main is recording each
ruling explicitly rather than treating a blanket grant as license to pick
whichever reading is most convenient for closing the step -- a vague
authorization does not manufacture missing evidence, and is not stretched
into one here.

**Ruling on C7**: criterion 7 ("NO flag is promoted and NO .env is written
by this step") is VIOLATED AS LITERALLY WRITTEN. A flag was promoted and
`backend/.env` was written during this step's execution. The in-session
operator approval (token `ARM-SYNTHESIS-INTEGRITY-86.69`, "Yes -- arm it" /
"Now") mitigates the deviation -- it was not unauthorized, and the
resulting production state is verified safe (`paper_synthesis_integrity_
enabled=True`, `paper_position_recommendation_fix_enabled=False`, NOT the
interaction-hazard combination `portfolio_manager.py:212-220` warns
against) -- but mitigation is not satisfaction of the criterion's text.
**This is not overturned by the general permission grant.**

**Precedent for future steps** ("this recurs" per the original escalation):
a numbered ask that receives in-session operator approval must still be
RECORDED and left for the operator's own subsequent action (a separate
turn, their own hands on the keyboard/`.env`), not executed by the step
itself in the same breath -- even with real-time approval. If immediate
execution is genuinely warranted, that is itself a fresh, explicit operator
instruction to deviate from criterion 7's text for that one instance, not a
standing license.

**Consequence for the step**: C7 remains a violated criterion. This does
NOT change the outcome -- C4/C5 remain STARVED regardless of any
permission grant, because the blocking condition is an empty measurement
population (zero parse-failure events in the observation window), which no
authorization can manufacture. **Decision: PARK AS-IS (option 1, Main's
original recommendation), now operator-confirmed.** The step stays
`status: pending` in `.claude/masterplan.json`. No status flip. No file
under `backend/` is touched by this ruling.

Cycle-2 research (`wf_c1c4da62-bee`, gate PASSED, 9 sources/37 URLs)
additionally confirmed: the honest-absence branch fires only when BOTH
paths fail, so the armed guard converts the defect into a LITE row (real
score) in the common case, not a NULL row -- a future measurement expecting
NULL rows to appear will not see them; and an unwired instrument
(`backend/agents/parse_failure_ledger.py`, untracked) that could supply
C4/C5's missing denominator belongs to step 86.108, not this step, and must
not be absorbed here. See `research_brief_86.69.md` §CYCLE-2 C/F for the
full derivation.
