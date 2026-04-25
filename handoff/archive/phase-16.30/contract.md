---
step: phase-16.30
title: Mini-batch hardening (#10 phosphor + #27 freshness docs + #35 fromisoformat)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.30

## Research-gate summary

`handoff/current/phase-16.30-research-brief.md`. tier=simple, 6 in-full, 16 URLs, recency scan, gate_passed=true.

## Key research findings

1. **#10 — TrendDown is NOT directly exported** from `frontend/src/lib/icons.ts`. It's re-exported under semantic names (`DebateBear` line 52, `MacroYieldSpread` line 90). Need to add identity re-export `TrendDown as TrendDown` to icons.ts, then swap the import in `RedLineMonitor.tsx:16`.

2. **#27 — Both freshness routes already work correctly**. Both `paper_trading.py:273-286` and `observability_api.py:25-41` call the same `cycle_health.compute_freshness` helper. Only gap is docstring + `.claude/rules/backend-api.md` documentation. No code logic change.

3. **#35 — fromisoformat ROOT CAUSE CONFIRMED**: `outcome_tracker.py:94` calls `datetime.fromisoformat(report["analysis_date"])` but `bigquery_client.py:268`'s `get_recent_reports` returns BQ rows where TIMESTAMP columns come back as native `datetime` objects, not strings. Fix: isinstance guard.

4. **CRITICAL: `backend/tests/test_outcome_tracker.py` does NOT exist.** The verification command's `|| true` silently swallows pytest collection errors — Q/A must explicitly verify the file exists + tests pass, NOT just rely on exit code. Main MUST create the file with a regression test.

## Hypothesis

Three small, orthogonal fixes ship in one cycle:
- 1 frontend edit (RedLineMonitor.tsx) + 1 frontend re-export addition (icons.ts)
- 1 backend bug fix (outcome_tracker.py isinstance guard)
- 1 new test file (test_outcome_tracker.py with regression test)
- 3 doc updates (paper_trading.py docstring, observability_api.py docstring, backend-api.md note)

Total: ~50-70 LOC across 6 files. Verification: vitest passes RedLineMonitor + grep returns "phosphor cleanup ok" + new pytest test passes.

## Success Criteria (verbatim, immutable)

```
cd frontend && npm run test -- --filter=RedLineMonitor && cd .. && grep -q 'phosphor-icons' frontend/src/components/RedLineMonitor.tsx && echo 'still has direct phosphor import' || echo 'phosphor cleanup ok' && python -m pytest backend/tests/test_outcome_tracker.py -q 2>&1 | tail -3 || true
```

- redline_monitor_uses_lib_icons
- freshness_docs_alias_documented
- fromisoformat_bug_fixed_or_root_caused
- no_regressions

## Plan steps

1. Add `TrendDown as TrendDown,` to `frontend/src/lib/icons.ts`
2. Edit `frontend/src/components/RedLineMonitor.tsx:16` import
3. Add isinstance guard in `backend/services/outcome_tracker.py` (line 94 area)
4. Create `backend/tests/test_outcome_tracker.py` with regression test (passes native datetime, asserts no TypeError)
5. Add canonical/alias docstrings to `paper_trading.py:273` + `observability_api.py:25`
6. Add 1-paragraph note to `.claude/rules/backend-api.md` re: dual-route pattern
7. Run: vitest + verification command + new pytest
8. Spawn Q/A

## What Q/A must audit

1. RedLineMonitor.tsx imports from `@/lib/icons`, NOT `@phosphor-icons/react`
2. `TrendDown` is exported from `icons.ts`
3. RedLineMonitor vitest still 4/4 PASS (no regression)
4. **`backend/tests/test_outcome_tracker.py` EXISTS** (Q/A must spot-check this file directly — not rely on `|| true` swallowing)
5. New pytest test PASSES (regression coverage for the BQ-datetime fromisoformat bug)
6. `evaluate_recent` now returns a non-empty list (or genuinely-empty without the spurious TypeError) when the BQ table has rows
7. Both freshness routes still serve 200; docstrings reflect canonical/alias
8. `.claude/rules/backend-api.md` mentions the dual-route pattern
9. No regression on the broader pytest suite (177/178 baseline)
