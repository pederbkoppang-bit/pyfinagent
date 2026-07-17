# Experiment results — step 71.3 (harden Q/A judgment + machine-readable verdicts)

**Phase/step:** phase-71 → 71.3 | **Date:** 2026-07-17 | **Type:** harness-protocol / qa.md rubric edit + a
fail-open verdict gate. $0, local-only, NO production/live-loop change; historical_macro FROZEN; live book untouched.

## What was changed

### `.claude/agents/qa.md` (rubric hardening — WITHIN the single Q/A role)
1. **Contract-completeness dimension** — new §4 LLM-judgment bullet: the Q/A maps EVERY immutable criterion →
   covering evidence in `experiment_results.md`; an uncovered criterion = `Missing_Assumption` that CAPS the
   verdict. Plus a "Contract completeness" row in the Quality-criteria table (lands the `completeness` grep token).
2. **Adversarial worst-of-N-LENSES leg (§4a, P0/P1 money-path only)** — the SAME single Q/A judges the claimed PASS
   from N DISTINCT lenses (correctness / does-it-reproduce / scope-honesty) and takes the WORST (`min(lens
   verdicts)`). Explicitly framed as perspective-diverse worst-of-N and **explicitly NOT** the N-IDENTICAL
   self-consistency resampling (#8a, DROPPED in phase-71.0 — correlated self-bias, arXiv:2508.06709; diverse lenses
   catch what identical resampling can't, arXiv:2505.19477). No fourth agent, no re-split.
3. **evaluator_critique.json emission note** — a new "Machine-readable verdict" section: the Q/A stays read-only;
   **Main** persists the returned verdict object to `handoff/current/evaluator_critique.json` (+ `step_id` +
   `cycle_num`; `checks_run` as an object map) alongside the `.md`, so the status-flip gate reads
   `verdict=="PASS" && ok==true` deterministically.

### `docs/runbooks/per-step-protocol.md` §4 EVALUATE
Added a "Q/A rubric hardening (phase-71.3)" block documenting the completeness dimension + the N-lens leg (distinct
lenses, NOT N-identical) + the `evaluator_critique.json` persistence. (The grep spans qa.md OR per-step-protocol.md.)

### `.claude/hooks/lib/verdict_gate.py` (NEW) + wiring (criterion 2 — "the gate reads JSON, not prose")
- A fail-open gate helper mirroring `live_check_gate.py`: reads `handoff/current/evaluator_critique.json`, returns
  `passed` iff it matches this `step_id` AND `verdict=="PASS"` AND `ok` is true; `hold` on an explicit step-matched
  non-PASS; `proceed` (FAIL-OPEN) on missing / unreadable / stale / mismatched / no-verdict. Never raises.
- Wired into `.claude/hooks/auto-commit-and-push.sh` after the two existing gates (same case pattern). On `hold` it
  logs WARN + holds the push (exit 0 — NEVER breaks the masterplan Write). Steps that predate the JSON (or emit a
  PASS) are unaffected.
- `backend/tests/test_phase_71_3_verdict_gate.py` (NEW, 9 tests): PASS→passed, CONDITIONAL/FAIL/ok-false→hold,
  missing/stale/mismatched/no-verdict/unreadable→proceed.

### evaluator_critique.json emission for 71.3 (dogfood)
Per the mechanism above, Main persists **this step's own** `evaluator_critique.json` from the 71.3 Q/A return value
(before the flip), and the flip's `verdict_gate` reads it → demonstrating the end-to-end. (At Q/A time the file is
not yet written — the Q/A verifies the MECHANISM + docs + gate + tests; criterion 2's "can read it deterministically"
is a capability satisfied by the gate + the documented emission, dogfooded at flip time.)

## Verification command output (verbatim)
```
$ bash -c 'grep -Eqi "contract completeness|completeness" .claude/agents/qa.md && grep -Eqi "worst-of-n|self-consistency|adversarial" .claude/agents/qa.md docs/runbooks/per-step-protocol.md'
VERIFICATION: PASS (exit 0)
$ bash -n .claude/hooks/auto-commit-and-push.sh                              -> OK
$ uvx ruff check --select F821,F401,F811 verdict_gate.py + test             -> All checks passed!
$ python -m pytest backend/tests/test_phase_71_3_verdict_gate.py -q          -> 9 passed
```
The qa.md "#8a negation" is present (`grep -c "NOT the N-IDENTICAL" qa.md` = 1) so the dropped idea is NOT
re-introduced. git scope: `.claude/agents/qa.md`, `docs/runbooks/per-step-protocol.md`,
`.claude/hooks/lib/verdict_gate.py` (new), `.claude/hooks/auto-commit-and-push.sh`, the new test + handoff. NO
backend/frontend production code changed.

## Criterion evidence
- **C1** — qa.md has the contract-completeness dimension + the §4a adversarial worst-of-N-LENSES leg (P0/P1),
  both WITHIN the single Q/A role (no fourth agent, no re-split), with the explicit #8a negation.
- **C2** — the machine-readable `evaluator_critique.json` schema (= the 71.1 VERDICT_SCHEMA + step_id/cycle_num);
  the fail-open `verdict_gate.py` + hook wiring make the status-flip gate read the verdict JSON deterministically
  (9 tests). Main persists 71.3's own JSON before the flip (dogfood).
- **C3** — single-Q/A-per-step + file-based handoffs PRESERVED; exactly-3-agents (the N lenses are one agent's N
  perspectives, not new agents); Q/A stays read-only (Main is the scribe → no-self-eval holds).

## Do-no-harm / scope honesty
$0; local-only; NO production/live-loop change; historical_macro FROZEN; live book untouched. The verdict gate is
FAIL-OPEN (only HOLDS on an explicit step-matched non-PASS; never breaks the masterplan Write). The dropped
N-identical #8a is explicitly negated, not re-introduced. Edits qa.md → separation-of-duties + roster note in the
harness_log (Peder review + verify_qa_roster_live.sh next session).
