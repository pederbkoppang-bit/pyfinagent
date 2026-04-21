# Sprint Contract — phase-9 / 9.7 (weekly data integrity) — REMEDIATION v1

**Step id:** 9.7 **Remediation cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Why remediation

Previous cycle inline-authored. Fresh MAS re-run.

## Research-gate summary

Fresh researcher: `handoff/current/phase-9.7-research-brief.md` — 6 sources in full, 16 URLs, three-variant queries, recency, gate_passed=true.

Validated: 20% drift threshold defensible as general default (industry range 10-40%); bespoke implementation appropriate at this scale (Soda/GE add framework overhead for no material gain).

Carry-forwards (NOT in 9.7 scope):
1. **Cross-phase to 9.9:** `scheduler.py:374` `add_job` call passes no `current_counts`/`prior_counts` args → `run()` executes with empty dicts and always reports zero drifts. Latent production bug — MUST fix in phase-9.9 scheduler wiring remediation.
2. Direction-blind drift: same threshold for drops (data loss) and growth (often normal) — drops should trigger higher-severity alert; append-only tables like `harness_learning_log` need dedicated growth-floor check
3. Single-week baseline vs rolling 4-week median (Acceldata 2025 recommendation) — more robust against backfill outlier weeks
4. Schema/null-rate/distribution/freshness-SLA checks not included — deferred to post-9.7 hardening

## Immutable criterion

`python -c "import ast; ast.parse(open('backend/slack_bot/jobs/weekly_data_integrity.py').read())" && pytest tests/slack_bot/test_weekly_data_integrity.py -q`

## Plan

1. Re-verify.
2. Capture output.
3. Spawn fresh Q/A.
4. Log and confirm.

## References

- `handoff/current/phase-9.7-research-brief.md`
- `backend/slack_bot/jobs/weekly_data_integrity.py` (57 lines)
- `tests/slack_bot/test_weekly_data_integrity.py` (3 tests)
- `backend/slack_bot/scheduler.py:374` (cross-phase carry to 9.9)
- `.claude/masterplan.json` → phase-9 / 9.7
