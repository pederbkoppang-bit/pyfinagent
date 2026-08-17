# ESCALATION -- step 86.108 -- PARKED on the 3rd-CONDITIONAL rule

**Status: PARKED, not done, not failed.** `.claude/masterplan.json` still
carries `"status": "pending"`. No step was flipped.

**Operator decision needed. One line answers it.**

## The situation in three facts

1. **All six immutable criteria are MET.** The cycle-3 evaluator's words:
   *"All six immutable criteria have covering, independently-reproduced
   evidence and the product is sound under 28 executed mutation cells"* -- its
   own 11 cells plus my 17, which it reproduced independently.
2. **Every finding raised across all three cycles is now closed**, including
   the two that capped cycle 3. See the table below.
3. **The next Q/A verdict is FAIL by rule, whatever the evidence says.** The
   verdict sequence is `[CONDITIONAL, CONDITIONAL, CONDITIONAL]`. CLAUDE.md's
   3rd-CONDITIONAL auto-FAIL rule requires the next pass to return FAIL, not
   another CONDITIONAL. The cycle-3 return already computed
   `consecutive_conditionals: 2, would_auto_fail: true` **before** its own
   CONDITIONAL was recorded.

So a fourth spawn does not evaluate the work; it executes a rule. That is the
rule working as designed -- it exists to stop a harness logging instead of
correcting -- but here the corrections did land, and the evaluator confirmed
each one by execution. Spending attempt 4 of 5 on a predetermined FAIL, then
needing attempt 5 for the PASS, is how a step with met criteria runs out of
budget.

**I did not spawn a fourth Q/A.** Per the standing rule -- *all criteria MET +
starved => PARK + escalation file, never iterate* -- this file is the stop.

## Attempt state

| | |
|---|---|
| Verdict sequence | `CONDITIONAL, CONDITIONAL, CONDITIONAL` |
| Attempts used | 4 of 5 (`scripts/harness/attempt_gate.py --status 86.108`) |
| Ledger rows | 3, agreeing with the Q/A's own `prior_attempts` at each spawn |
| Run ids | `wf_f0fc7207-486`, `wf_a49d2d57-3e1`, `wf_95c6d117-784` |

## What each cycle actually found, and what closed it

Worth reading, because the pattern is the point: **not one finding across three
cycles was a defect in the shipped product's behaviour.** Two were real defects
in *my guards*, and the rest were claims that did not reproduce.

| Cycle | Finding | Closed by |
|---|---|---|
| 1 | **PRODUCT** -- `current_rail()` read the CC-route flag alone, but the client enters that rail on `model.startswith("claude-") AND flag`. Every Gemini-served failure was stamped `claude_code`; Gemini traffic outnumbers CC-tagged ~20x, so the misattribution was the common case. | `resolve_rail(model_name)` mirroring the real predicate; model threaded through 9 call sites; three honest `unknown`s with stated bases. |
| 1 | The rail's only guard was a **set-membership assertion**, so an inverted attribution survived an evaluator-run mutant. | Value assertions, a 5-cell truth table, the discriminating flag-only-disagreement test, cells M13/M14. |
| 1 | Ruff F401; broken `--sql` re-derivation path; a figure without its population. | All three fixed and re-verified. |
| 2 | **The cycle-1 fix relocated the defect one seam upstream** and built every guard at the old seam. A mutant hardcoding `_effective_model_name` SURVIVED 29/29. | Behavioural drivers over the real `run_debate`/`run_risk_debate`; unit tests; an AST completeness guard; cells M15-M17. |
| 2 | The ruff block used a **hand-assembled scope that omitted the one file with a finding**. | Regenerated over a scope derived from `git diff`; it now exits 1 and names the pre-existing finding. |
| 3 | **The AST guard was a BLACKLIST** -- it rejected an `ast.Constant` only, so `_client_model_name(None)` and `... or "claude-opus-4-8"` both survived. | Converted to a **whitelist** of accepted shapes; cells M18/M19 reproduce both survivors, both KILLED. Matrix **19/19**. |
| 3 | The regression figure was regenerated in one artifact and left stale in the other, **while that artifact claimed it had been regenerated**. | Both now carry the measured 560; the block was re-run, not edited. |
| 3 | "Queued as a defect" did not resolve against the masterplan. | Actually filed: **86.112**, **86.113**, **86.114**. |
| 3 | "Every one of the 37 tests drives the REAL function" was false for the AST test. | Corrected to 36 of 37, exception named. |
| 3 | `parse_llm_json` was listed as an equivalent emit site while having **zero production callers**. | Disclosed, with the coverage consequence stated: the three wired sites cover the whole measured population. |

## The decision

**Option A -- accept on the cycle-3 evaluator's finding.** It stated all six
criteria met and the product sound; the five residual findings are closed and
each closure is re-runnable (`python scripts/qa/mutation_86_108.py` → 19/19;
`pytest backend/tests/test_phase_86_108_parse_failure_ledger.py` → 37 passed).
Flip 86.108 to `done` on that basis. **This requires your authority, not mine
-- Main must never mark a step done without a PASS**, which is why the step is
parked rather than closed.

**Option B -- authorise attempt 4 knowing it returns FAIL by rule**, so the
counter resets and attempt 5 can evaluate the current evidence on its merits.
Costs both remaining attempts.

**Option C -- park indefinitely** and let the work sit uncommitted-to-done. The
code is shipped and correct either way; only the step's status is in question.

My recommendation is **A**, with **B** as the principled alternative if you want
the rule honoured literally. What I will not do is spawn a fourth Q/A and read
its forced FAIL as a judgement on the work.

## What is shipped and in force

- **Shipped and committed**: the parse-failure ledger (`model_name`, `rail`,
  `rail_basis` on every record), the four wired emit sites, the derived
  168-flag observability route, the census/era/mutation tooling, 37 tests,
  19 mutation cells.
- **NOT in force**: `GET /api/settings/flags` and
  `GET /api/observability/parse-failures` return **404** on the running backend
  (pid 41635, started 2026-08-17T13:57:16Z, before these edits), while
  `/api/observability/latency` returns 200 as a positive control. Per the
  batched-restart rule the restart is deferred to session end. **ASK-3.**
- **No flag promoted, no `.env` written.** `backend/.env` mtime is unchanged at
  2026-08-17T15:06:04 CEST.

## Standing asks from this step

- **ASK-1** -- should the five dead `_FIELD_TO_ENV` rows become writable
  (i.e. a UI write path for dark flags)? This step declined to create one.
- **ASK-2** -- superseded: now filed as step **86.114**.
- **ASK-3** -- the session-end restart that brings the two read-only routes
  into force.
