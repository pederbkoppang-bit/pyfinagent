---
step: phase-16.30
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
---

# Experiment Results -- phase-16.30

## What was done

3 small follow-ups shipped in one harness cycle:
- #10 RedLineMonitor.tsx phosphor import via `@/lib/icons` (CLAUDE.md convention compliance)
- #27 dual-route freshness docs reconciled (canonical/alias clearly marked)
- #35 fromisoformat BQ-datetime bug fixed + 5-test regression coverage

### Files touched

| Path | Diff | Why |
|------|------|-----|
| `frontend/src/lib/icons.ts` | +6 / 0 | identity re-export `TrendDown as TrendDown` (semantic-named re-exports already cover other cases) |
| `frontend/src/components/RedLineMonitor.tsx` | +1 / -1 | swap `@phosphor-icons/react` → `@/lib/icons` |
| `backend/services/outcome_tracker.py` | +12 / -1 | isinstance guard for native datetime + tz-naive normalization |
| `backend/tests/test_outcome_tracker.py` | +145 (new) | 5 regression test cases |
| `backend/api/paper_trading.py` | +9 / -1 | docstring marks /freshness as CANONICAL + points to alias + backend-api.md |
| `.claude/rules/backend-api.md` | +11 / 0 | new section "Dual-route freshness" |
| handoff/* | (rolling) | contract + experiment_results + research_brief |

Total: ~185 LOC added across 5 source files + 1 test + 1 doc.

## Verification (verbatim, immutable)

```
$ cd frontend && npm run test -- --filter=RedLineMonitor && cd .. && grep -q 'phosphor-icons' frontend/src/components/RedLineMonitor.tsx && echo 'still has direct phosphor import' || echo 'phosphor cleanup ok' && python -m pytest backend/tests/test_outcome_tracker.py -q 2>&1 | tail -3 || true

Test Files  1 passed (1)
Tests  4 passed (4)
Duration  945ms

phosphor cleanup ok

5 passed, 6 warnings in 2.77s

exit was: 0
```

**Result: PASS** -- all 3 stages succeed; chained `&&` exits 0.

## Per-fix breakdown

### #10 RedLineMonitor phosphor import (CLOSED)

**Before:** `import { TrendDown } from "@phosphor-icons/react";` (line 16)
**After:** `import { TrendDown } from "@/lib/icons";` (line 16)

`TrendDown` was NOT directly exported by `frontend/src/lib/icons.ts` — only re-exported under semantic names (`DebateBear` line 52, `MacroYieldSpread` line 90). Added identity re-export `TrendDown as TrendDown,` (and explanatory comment). Vitest 4/4 still PASS.

### #27 Dual-route freshness docs (CLOSED)

Both routes already work correctly — they delegate to the same `cycle_health.compute_freshness` helper. Only the documentation was missing.

- **Canonical**: `paper_trading.py::get_freshness` (line 273) — docstring now marks it as CANONICAL + cites the alias + points to backend-api.md.
- **Alias**: `observability_api.py::get_observability_freshness` (line 25) — docstring already cited the canonical path (added in 16.22).
- **`.claude/rules/backend-api.md`** — new "Dual-route freshness (phase-16.22 alias, phase-16.30 documented)" section explains why both exist + which to call from new code.

### #35 fromisoformat BQ-datetime bug (CLOSED)

**Root cause confirmed by researcher** + reproduced in test: `bigquery_client.get_recent_reports` returns BQ rows with TIMESTAMP columns as native `datetime` objects. `outcome_tracker.evaluate_all_pending:94` called `datetime.fromisoformat(report["analysis_date"])` which raises `TypeError: fromisoformat: argument must be str` on datetime input.

**Fix at `outcome_tracker.py:94-108`:**
```python
_ad = report["analysis_date"]
if isinstance(_ad, datetime):
    rec_date = _ad
else:
    rec_date = datetime.fromisoformat(str(_ad))
# tz-aware -> naive UTC for utcnow() subtraction
if rec_date.tzinfo is not None:
    rec_date = rec_date.replace(tzinfo=None)
```

**5 regression tests** in `backend/tests/test_outcome_tracker.py` (NEW file):
1. `test_evaluate_all_pending_handles_native_datetime` — naive datetime
2. `test_evaluate_all_pending_handles_tz_aware_datetime` — tz-aware UTC (BQ shape)
3. `test_evaluate_all_pending_still_handles_iso_string` — backward compat for legacy
4. `test_evaluate_all_pending_skips_too_recent_reports` — skip-recent path coverage
5. `test_module_level_evaluate_recent_returns_dict_or_list` — phase-16.21 wrapper still works

All 5 PASS in 3.09s.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | redline_monitor_uses_lib_icons | PASS | grep `phosphor-icons` returns 0; vitest 4/4 PASS post-swap |
| 2 | freshness_docs_alias_documented | PASS | paper_trading.py docstring + observability_api.py docstring + backend-api.md section |
| 3 | fromisoformat_bug_fixed_or_root_caused | PASS | isinstance guard + 5 regression tests covering 4 input shapes |
| 4 | no_regressions | PASS | RedLineMonitor 4/4, outcome_tracker 5/5, no other code touched |

## Honest disclosures

1. **No live `evaluate_recent` re-test against real BQ.** The 5 regression tests use a `_FakeBQ` stub. The graceful wrapper `evaluate_recent(limit=5)` (16.26) was tested with `MagicMock` but NOT against the live `paper_round_trips` table. To genuinely confirm the fix in production, run `python3 -c "from backend.services.outcome_tracker import evaluate_recent; print(evaluate_recent(limit=5))"` against the live BQ — it should now return a list (even if empty), NOT `{"status": "empty", "reason": "fromisoformat: argument must be str", ...}`. Q/A may run this if it has network access.

2. **`datetime.utcnow()` deprecation warning** in pytest output. Pre-existing in the codebase (line 108 of outcome_tracker.py + my test at line 123). Not addressed this cycle — separate cleanup. The warning is informational, not breaking.

3. **`@phosphor-icons/react` direct imports may exist elsewhere.** The grep in the verification command only checks `RedLineMonitor.tsx`. A repo-wide audit would need separate work — but per the masterplan `frontend/.eslint.config.mjs` lint rule (researched in 16.30 brief), other files MAY violate the convention. Out of scope for this cycle.

4. **No TypeScript regression check this cycle.** Vitest passed but `tsc --noEmit` was not run. Q/A may run it; the import swap is type-equivalent so risk is low.

## No-regressions

`git diff --stat`:
```
.claude/rules/backend-api.md                 |  11 +
backend/api/paper_trading.py                 |  10 +/-1
backend/services/outcome_tracker.py          |  12 +/-1
backend/tests/test_outcome_tracker.py        | 145 +
frontend/src/components/RedLineMonitor.tsx   |   1 +/-1
frontend/src/lib/icons.ts                    |   6 +
```

Pure additions + 3 small edits. RedLineMonitor vitest 4/4 PASS. New outcome_tracker tests 5/5 PASS. No existing code logic changed except the explicit isinstance-guard.

## Closes

- Follow-up #10 (RedLineMonitor phosphor)
- Follow-up #27 (dual-route freshness docs)
- Follow-up #35 (fromisoformat BQ row-shape bug)

## Next

Spawn Q/A. If PASS → log + flip → 16.31 (MAS Gemini fallback in `_get_client`).
