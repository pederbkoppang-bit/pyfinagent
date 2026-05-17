# Sprint Contract — phase-27.5 (E2E smoke verify on Gemini)

Generated: 2026-05-16T22:20:00+00:00
Owner: Main
Step id: 27.5
Depends on: 27.1 + 27.2 + 27.3 + 27.4 (all done)

## Research-gate summary

Combined research from 27.0 covers all four upstream fixes that this step verifies in aggregate. No new external research needed for the smoke itself.

## Hypothesis

With all four fixes applied (Anthropic strict schema, Gemini null-text safety, provider-aware lite fallback, BQ schema migration), flipping `standard_model=gemini-2.5-flash` via the Settings API and running a cycle should produce:

- Cycle status `completed`
- Lite path runs on Gemini for every analyzed ticker (full path STILL EXPECTED TO FAIL on some tickers because we have not audited the 28-skill pipeline's Gemini compatibility — but the cycle survives because the lite fallback now works)
- ≥14 of 15 analyses persisted to `financial_reports.analysis_results` (B-2 unblocked)
- Zero `'Both full and lite paths failed'` log lines (C2 unblocked)

Falsifier: if the cycle fails to complete, or analyses still fail to persist, or `'Both full and lite paths failed'` reappears, one or more of 27.1-27.4 is incomplete and we go back to per-step gating.

## Immutable success criteria (verbatim from `.claude/masterplan.json` step 27.5)

```bash
test -f handoff/current/live_check_27.5.md && \
grep -q 'cycle_id' handoff/current/live_check_27.5.md && \
grep -q 'lite_mode.*[Ff]alse' handoff/current/live_check_27.5.md && \
grep -qE 'analyses_persisted.*1[4-9]|analyses_persisted.*2[0-9]' handoff/current/live_check_27.5.md
```

Plus success_criteria from the masterplan including `zero_Full_orchestrator_failed_lines_for_the_cycle` — note Main will flag if this is NOT achieved (full pipeline Gemini-compatibility audit may need to be queued as a separate post-launch phase).

## Plan steps

1. PUT `/api/settings/models` with `gemini_model=gemini-2.5-flash`, `deep_think_model=gemini-2.5-pro` (already there from earlier in session).
2. `launchctl kickstart -k gui/$UID/com.pyfinagent.backend` to pick up new defaults.
3. Confirm `/api/health` and `/api/settings/models` reflect Gemini.
4. POST `/api/paper-trading/run-now` to trigger fresh cycle.
5. Wait for cycle to complete (poll `/api/paper-trading/status` until `loop.running=false`).
6. Capture cycle metadata (cycle_id, steps, signals_logged, trades_executed).
7. Query BQ row count delta on `analysis_results` before vs after.
8. grep `backend.log` for `'Both full and lite paths failed'` (must be 0) and `'Full orchestrator failed'` (count + first error to triage downstream).
9. Write `handoff/current/live_check_27.5.md` with verbatim cycle output + BQ delta + log grep results.
10. Q/A spawn.
11. harness_log append.
12. Flip 27.5 to done.

## Anti-patterns to avoid

- Do NOT manually mark PASS if full-path errors appear — capture them honestly and let Q/A decide.
- Do NOT cherry-pick a "good" subset of tickers — accept whatever the cycle produces.

## References

- `.claude/masterplan.json` phase-27 step 27.5 verification command (immutable)
- `handoff/current/experiment_results.md` (rolling — soon to be 27.5)
- Prior cycle evidence in `backend.log` for pre-fix baseline comparison
