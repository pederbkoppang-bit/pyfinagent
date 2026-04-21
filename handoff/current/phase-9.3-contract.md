# Sprint Contract — phase-9 / 9.3 (weekly FRED refresh) — REMEDIATION v1

**Step id:** 9.3 **Remediation cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Why remediation

Previous cycle was inline-authored by Main (no real Researcher, no real Q/A). Re-running full MAS harness with fresh agents on fresh evidence.

## Research-gate summary

Fresh researcher spawn (2026-04-20) produced `handoff/current/phase-9.3-research-brief.md`:
- 6 sources read in full via WebFetch (exceeds 5 floor)
- 17 URLs collected; 11 snippet-only
- Three-variant query discipline (current-year / last-2-year / year-less canonical)
- Recency scan performed
- gate_passed = true

Validated design choices: ISO-week idempotency key is safe for DGS10/DGS2/VIXCLS/DFF (market-derived, not revised intra-week) and UNRATE/CPIAUCSL (BLS monthly-revision cadence). fredapi v0.5.2 lacks auto-throttling but FRED's 120 req/min limit is rarely hit at weekly 6-series cadence. Callable DI pattern is canonical.

Carry-forwards (deferred, not in 9.3 scope):
1. Hardcoded `_DEFAULT_SERIES` → config-injectable
2. In-memory `_GLOBAL_STORE` → BQ/Redis for prod
3. Latest-vintage only → consider ALFRED for look-ahead-safe backtesting
4. Missing T10Y2Y + BAMLH0A0HYM2 (standard US equity-regime signals in practitioner lit)
5. Consider pyfredapi v0.10.2 or FedFred v3 for auto-throttling at go-live

## Immutable criterion

`python -c "import ast; ast.parse(open('backend/slack_bot/jobs/weekly_fred_refresh.py').read())" && pytest tests/slack_bot/test_weekly_fred_refresh.py -q`
Expected: exit 0, 3/3 pass.

## Plan

1. Re-verify ast.parse + pytest 3/3 on unchanged artifact.
2. Capture verbatim output.
3. Spawn fresh Q/A.
4. Log and flip task status (masterplan already `done`).

## References

- `handoff/current/phase-9.3-research-brief.md` (6 sources in full)
- `backend/slack_bot/jobs/weekly_fred_refresh.py` (unchanged, 44 lines)
- `tests/slack_bot/test_weekly_fred_refresh.py` (unchanged, 3 tests)
- `backend/slack_bot/job_runtime.py` (IdempotencyKey.weekly + heartbeat from phase-9.1)
- `.claude/masterplan.json` → phase-9 / 9.3
