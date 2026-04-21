# Sprint Contract — phase-8.5 / 8.5.2 (Budget enforcer) — REMEDIATION v1

**Step id:** 8.5.2 **Date:** 2026-04-20 **Tier:** simple

## Why remediation
Original qa_852_v1 PASS on inline-authored brief. Re-audit with researcher subagent + fresh Q/A.

## Research-gate summary
Researcher confirmed 5 sources in full (OneUptime 2026, debugg.ai 2025, thelinuxcode 2026, zetcode, PyBreaker). time.monotonic() grounded; injectable alert_fn is the canonical listener-injection pattern. Minor docstring defect (L31 says time.time()) noted but implementation correct. Brief at `handoff/current/phase-8.5.2-research-brief.md`.

## Immutable criterion
- `python scripts/harness/autoresearch_budget_test.py` exits 0.

## Plan
Re-run test. Spawn fresh Q/A. Log.

## References
- `handoff/current/phase-8.5.2-research-brief.md`
- `backend/autoresearch/budget.py`
- `scripts/harness/autoresearch_budget_test.py`
