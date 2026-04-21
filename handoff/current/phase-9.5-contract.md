# Sprint Contract — phase-9 / 9.5 (hourly signal warmup) — REMEDIATION v1

**Step id:** 9.5 **Remediation cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Why remediation

Previous cycle inline-authored. Fresh MAS re-run.

## Research-gate summary

Fresh researcher: `handoff/current/phase-9.5-research-brief.md` — 5 sources in full, 11 URLs, three-variant queries, recency scan, gate_passed=true.

Validated: idempotent hourly key + injectable cache + settings-fallback watchlist is correctly structured. Dict-as-cache is acceptable for single-process APScheduler.

Carry-forwards (not in 9.5):
1. Cache entries lack TTL — a failed warmup leaves hour-old signals in dict (add `warmed_at` stamp)
2. No market-hours gating — consider optional `exchange_calendars` closed-market log (low priority; cost is minor)
3. Document Redis upgrade path for multi-worker scenarios
4. Sort watchlist by position-size desc before iterating (partial-warmup prioritization)

## Immutable criterion

`python -c "import ast; ast.parse(open('backend/slack_bot/jobs/hourly_signal_warmup.py').read())" && pytest tests/slack_bot/test_hourly_signal_warmup.py -q`

## Plan

1. Re-verify.
2. Capture output.
3. Spawn fresh Q/A.
4. Log and confirm.

## References

- `handoff/current/phase-9.5-research-brief.md`
- `backend/slack_bot/jobs/hourly_signal_warmup.py` (48 lines)
- `tests/slack_bot/test_hourly_signal_warmup.py` (3 tests)
- `.claude/masterplan.json` → phase-9 / 9.5
