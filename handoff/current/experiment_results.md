# Experiment Results — phase-72.5: Rollup + push (closes the phase-72 goal)

Date: 2026-07-18. Session: Fable 5 + ultracode, AUDIT + RESEARCH ONLY ($0 metered).

## What was built

1. **Completeness-critic research gate** (`wf_62ba4963-b33`, tier=simple, floor held: 7 sources full on postmortem/audit/DoD closure standards): **zero blocker gaps** against the operator goal's four DoD elements; two cosmetic items returned and both FIXED this step (P2 $137.32 back-annotation; P3 current→proposed header note). The critic also independently verified: immutable criteria byte-identical install→HEAD for all 15 steps (zero mutation); all 9 remediation steps pending + executor-tagged + live_check; five-file protocol complete per closed step (`handoff/archive/phase-72.{0..4}/` × 4 files + harness_log Cycles 112-116); every archived critique verbatim-transcribed with Workflow run IDs; origin/main == local at audit time; ACT-NOW ↔ P3 ↔ P4 consistency.
2. **Doc-hygiene fixes applied** (2 edits): money_diagnosis_72.md P2 resolution note; operator_decision_sheet_72.md P3 current-state header. Earlier this step (pre-contract, flagged by the Stop-gate and confirmed by the critic as resolved on-disk): the header segmentation was rewritten to carry the three verified per-window causes, and the stale "P0 in progress" marker was closed.
3. **DoD state at close**:
   - `money_diagnosis_72.md`: three sub-period sections with verified causes + closed P0-P4 (all five Q/A PASSes referenced).
   - `operator_decision_sheet_72.md`: ACT-NOW (4 items) + P1 token reconciliation (15 rows) + P3 lever ranking (7 recommend-ON in sequence, 6 evidence-based HOLDs) + P4 regime policy (two-sided evidence) + recurrence prevention.
   - Masterplan: phase-72 with 72.0-72.4 done, 9 remediation steps pending executor-tagged with immutable live_checks.
   - Pushes: 72.0 @7b2499e3, 72.1 @080f93c1, 72.2 @665d7c0e, 72.3 (7502c664 push), 72.4 (037b5580 push) all on origin/main; the 72.5 closure commit lands on the status flip after Q/A + log (the critic flagged watching auto-push.log for the known stall — Main will verify and fall back to manual push if needed).

## Verbatim verification output

```
$ bash -c 'test -f handoff/current/money_diagnosis_72.md && test -f handoff/current/operator_decision_sheet_72.md && git log origin/main --oneline -5 | grep -q "phase-72"'
72.5 VERIFICATION COMMAND EXIT: 0 (PASS)
```

## Scope honesty

This step changed only handoff documentation (2 cosmetic fixes + this rollup's own artifacts) and the masterplan status flip. No product code, no .env, no flags, throughout the entire phase — verified per-step by five independent Q/A spawns and re-verified by the 72.5 completeness critic.
