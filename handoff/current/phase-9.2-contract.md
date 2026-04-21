# Sprint Contract — phase-9 / 9.2 (daily price refresh) — REMEDIATION v1

**Step id:** 9.2 **Remediation cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Why remediation

Previous cycle was inline-authored by Main (research brief + Q/A critique faked). Re-running the full MAS harness with a fresh researcher session and fresh Q/A subagent on freshly authored evidence.

## Research-gate summary

Researcher (fresh spawn 2026-04-20) produced brief `handoff/current/phase-9.2-research-brief.md`:
- 7 sources read in full via WebFetch (exceeds 5 floor)
- 17 URLs collected; 10 snippet-only
- Three-variant query discipline (current-year / last-2-year / year-less canonical)
- Recency scan performed
- gate_passed = true

Design validation: callable DI is canonical Python testability pattern; UTC-date idempotency keys are standard; fail-open / mark-on-success heartbeat is correct.

Carry-forward items flagged (NOT in scope for 9.2 remediation — these are for a later hardening phase):
1. Hardcoded 5-ticker universe → settings-driven
2. `date.today()` vs UTC mismatch
3. In-memory idempotency store → wire to BQ `job_heartbeat`
4. Missing retry/backoff in fetch path
5. Stub production `write_fn` should use MERGE not INSERT
6. yfinance ToS risk for go-live (consider Alpaca/Polygon)

## Immutable criterion

`python -c "import ast; ast.parse(open('backend/slack_bot/jobs/daily_price_refresh.py').read())" && pytest tests/slack_bot/test_daily_price_refresh.py -q`

Expected: exit 0, 3/3 tests pass.

## Plan

1. Re-verify ast.parse on the unchanged artifact.
2. Re-run pytest 3-test suite.
3. Capture verbatim output in experiment_results.md.
4. Spawn fresh Q/A.
5. Append log block; flip step status if still done (no change needed — artifacts intact).

## References

- `handoff/current/phase-9.2-research-brief.md` (researcher-authored, 7 sources in full)
- `backend/slack_bot/jobs/daily_price_refresh.py` (unchanged, 54 lines)
- `tests/slack_bot/test_daily_price_refresh.py` (unchanged, 3 tests)
- `backend/slack_bot/job_runtime.py` (IdempotencyKey + heartbeat primitives from phase-9.1)
- `.claude/masterplan.json` → phase-9 / 9.2
