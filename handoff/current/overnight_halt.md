# OVERNIGHT HALT — circuit breaker R2 tripped, 2026-08-17 00:50 CEST

**This is not a preflight abort.** The preflight passed cleanly at 20:47 and the
machine has been healthy all night. The breaker tripped on the rail the operator
wrote for exactly this case:

> **R2. CIRCUIT BREAKER.** If TWO CONSECUTIVE steps park at the cap without a
> PASS, STOP ALL STEP WORK. That is evidence the HARNESS is the blocker, not the
> steps.

**86.97 parked** (C, C, FAIL) and **86.94 parked** (F, F, CONDITIONAL). Those are
consecutive. Step work is stopped; the remaining time goes to §4 measurement,
**not** to implementing 86.98.

---

## The night in one table

| step | verdicts | outcome | attempts | tokens |
|---|---|---|---|---|
| **86.92** | CONDITIONAL → **PASS** | **CLOSED**, pushed | 2 | 636,006 |
| 86.97 | CONDITIONAL, CONDITIONAL, FAIL | PARKED at cap | 3 | 891,180 |
| 86.94 | FAIL, FAIL, CONDITIONAL | PARKED at cap | 3 | 914,117 |

**One step reached PASS.** Preflight: passed. Breaker: tripped.

Token spend at the halt: **2,441,303** of the 3,000,000 ceiling (measured, not
estimated). No metered spend; everything ran on the Max rail.

---

## Is the harness the blocker? What the evidence actually says

The rail's premise is that two consecutive parks indicate the harness rather than
the steps. **Tonight's evidence does not support that reading, and saying so is
more useful than confirming the rail's hypothesis.**

The 2026-08-16 day session's finding was that *8 of 15 verdicts said every
criterion was MET and returned CONDITIONAL anyway* — a harness-side defect. That
is **not** what happened tonight:

| step | did the verdict find a REAL defect? | was it MY defect? |
|---|---|---|
| 86.92 cycle 1 | yes — my positive control could not fail (mutant M5 survived) | yes |
| 86.97 cycle 1 | yes — `buildable()` was `bash -n`, blind inside the heredoc where both mutants live | yes |
| 86.97 cycle 2 | yes — a measured figure invalidated by my own edit in the commit that stated it | yes |
| 86.97 cycle 3 | yes — a justification that did not reproduce; criterion 5 swept by my own wording | yes |
| 86.94 cycle 1 | yes — 5 surviving mutants, 3 named verbatim in the step's own audit_basis | yes |
| 86.94 cycle 2 | yes — my correction *accompanied* instead of replacing; criterion-4 judgement false | yes |
| 86.94 cycle 3 | yes — a stale figure and an effectiveness claim that **measures zero** | yes |

**Every capping finding was a real defect in my work, and I reproduced each one
myself before accepting it.** Two were caught by my own re-measurement finding
the evaluator's claim *understated* the problem. None was a rubber-stamp
CONDITIONAL on met criteria.

So the honest read is: **the evaluator is working; the author is the bottleneck.**
The recurring shape is not random — it is one class, repeated:

> **I write a guard, and the guard has the very defect it was built to catch.**

- 86.92: a control for "prove the stripper is live" that could not fail.
- 86.97: an UNSCORABLE oracle blind to the only build failures its mutants could have.
- 86.94: a fail-**open** `continue` inside the module whose thesis is fail-closed;
  a scan that matched its own documentation; a checker that flagged itself; a
  criterion-4 predicate satisfiable by vocabulary; and then a *correction* that
  accompanied instead of replacing — inside the step whose criterion 5 is exactly
  that rule.

That is worth more to the operator than a park count.

---

## What is committed and safe

Everything is pushed; `origin == HEAD` confirmed. No step was flipped without a
PASS. No gate was loosened to get green. `qa.md`, `qa-verdict.js` and
`research-gate.js` were never touched (rail R5), verified by `git diff --stat`.

**Shipped and green tonight:**

- `scripts/qa/verify_workflow_args_boundary.mjs` — **restored from 6d12h dead**
  (84/3 → 96/0). It had stopped signalling, and worse, one mutation cell had gone
  **non-discriminating**. Unblocks pending step 86.23 (exit 1 → 0).
- `scripts/qa/verify_decision_log_86_97.py` — NEW, 35 assertions. Kills the
  delete-the-call mutant that was **invisible** to the 86.91 extraction.
- `scripts/qa/verify_no_sliding_windows_86_94.py` — NEW, 45 assertions.
- `replay_changelog_rule_86_68.py` — `CORPUS_SINCE` now UTC-qualified. phase-86.91's
  pin was TZ-local (707 Oslo/UTC/NY vs **787** Seoul). Published figures unchanged;
  now regenerable off this laptop.

**Filed rather than left as prose** (each verified to reproduce from disk):
**86.101** (the `-1` sentinel's rendering), **86.102** (the control covers one of
the stripper's two operations), **86.103** (the two MUST-LOG pre-detector paths).

---

## Corrections made AFTER a verdict, and therefore not re-graded

Disclosed here because they change the tree the evaluator saw:

1. `live_check_86.94.md:274` and `experiment_results_86.94.md:99` — "37 files" → **282**.
2. The claim that the widened rule "immediately found a live site the old one
   missed" — **RETRACTED**. I re-ran the A/B myself: reverting only the
   `WINDOW_RE` widening leaves the enumeration byte-identical, so it found
   **zero**. What I mistook for a find was an `argparse` flag I then excluded.
3. The allowlist prose said 55 files where the instrument measures 49 — reconciled.

Left for the next cycle, all named by the evaluator and all small: the
`quoted_as_evidence` bool is only isinstance-checked; the `<unparsed>`
fail-closed branch has no mutation cell; the argv cells may be credited to the
wrong leg.

---

## What the operator has to decide

1. **86.94 is one short cycle from PASS.** All 7 criteria are already MET on
   their literal wording; the cap was evidence integrity, and three of the six
   findings are already fixed. It is the cheapest close on the board.
2. **86.97 is also close** — criteria 1,2,3,4,6,7 were MET at cycle 3; it failed
   on criterion 5 and on one claim that did not reproduce.
3. **The 3-attempt rail worked as designed.** It stopped two steps that would
   otherwise have eaten the night, and both stopped with a named, actionable
   diagnosis rather than a shrug.
4. **86.98's premise needs re-examination.** Tonight's data contradicts the
   day-session pattern it was filed on: 7 of 7 capping findings here were real
   author defects, not evaluator noise. The §4 population measurement below is
   the input to that decision — and per the goal, it is measurement only; 86.98
   was **not** implemented.
