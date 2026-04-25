---
step: phase-16.34
title: Rename backend/calendar -> backend/econ_calendar (closes #9 final)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.34

## Research-gate summary

`handoff/current/phase-16.34-research-brief.md`. tier=moderate, 8 in-full, 18 URLs, recency scan, gate_passed=true.

## Key findings

1. **Stdlib shadow root cause:** When `cd backend && pytest` runs, pytest sets `sys.path[0] = backend/`. `import calendar` then resolves to `backend/calendar/__init__.py` (which does `from backend.calendar import sources`) BEFORE stdlib's `calendar`, causing a circular `ModuleNotFoundError: No module named 'backend.calendar'`. The fix: rename the package so it doesn't shadow stdlib.

2. **10 files need updates** (researcher's enumeration verified by independent grep):
   - `backend/calendar/__init__.py` (5 import lines)
   - `backend/calendar/sources/__init__.py` (1 import)
   - `backend/calendar/sources/finnhub_earnings.py` (1)
   - `backend/calendar/sources/fed_scrape.py` (1)
   - `backend/calendar/sources/fred_releases.py` (1)
   - `backend/calendar/watcher.py` (3 imports)
   - `backend/tests/test_calendar_watcher.py` (4 imports)
   - `scripts/smoketest/phase6_e2e.py` (1 import)
   - `scripts/migrations/add_calendar_events_schema.py` (comment only, non-critical)
   - `backend/services/observability/__init__.py` (docstring only, non-critical)

3. **`git mv` preserves history** (per multiple 2026 sources). Mandatory over plain `mv` for cycle-level traceability.

4. **No frontend changes** — `frontend/handoff/harness_log.md` has historical `backend.calendar` references that should NOT be touched (they're log artifacts, not live code).

5. **No BQ-schema impact** — calendar references are all Python module paths, no BQ table/column rename needed.

## Hypothesis

`git mv backend/calendar backend/econ_calendar` + sed-based import rewrite across 8 critical files (+ 2 cosmetic doc updates) will:
- Eliminate the stdlib shadow → `cd backend && pytest tests/api/test_sovereign.py -q` will PASS as written (currently fails with `ModuleNotFoundError`)
- Preserve the existing repo-root `pytest` invocation (still works)
- Leave the 9 calendar-watcher tests passing (no test logic changes)
- 0 regression on the broader pytest suite (177/178 baseline)

## Success Criteria (verbatim, immutable)

```
cd backend && python -m pytest tests/api/test_sovereign.py -q 2>&1 | tail -3
```

- calendar_renamed
- all_imports_updated
- cd_backend_pytest_works
- no_regression_177_pass

## Plan steps

1. `git mv backend/calendar backend/econ_calendar`
2. Use sed (pattern unambiguous) to rewrite import statements in 8 files:
   - `from backend.calendar` → `from backend.econ_calendar`
   - `import backend.calendar` → `import backend.econ_calendar`
3. Update 2 cosmetic docs (`scripts/migrations/add_calendar_events_schema.py` comment + `backend/services/observability/__init__.py` docstring)
4. Verify: AST clean all changed files
5. Run from REPO ROOT: `python -m pytest backend/tests/ -q` — confirm 177/178 still passes
6. Run from `backend/`: `python -m pytest tests/api/test_sovereign.py -q` — confirm 7/7 PASS (the masterplan-immutable verification)
7. Spawn Q/A

## What Q/A must audit

1. `backend/calendar/` directory NO LONGER exists (replaced by `backend/econ_calendar/`)
2. `git log --follow backend/econ_calendar/__init__.py` shows the old `backend/calendar/__init__.py` history (rename preserved via git mv)
3. Zero `from backend.calendar` or `import backend.calendar` references in live code (`grep -r ... backend/ scripts/ | grep -v handoff | grep -v __pycache__` returns empty)
4. Repo-root pytest still 177/178 (regression baseline)
5. **`cd backend && pytest tests/api/test_sovereign.py -q` actually works** (the original immutable verification cmd that's been broken since 10.5.0 shipped)
6. test_calendar_watcher.py tests still pass (9 functions, no logic change)
7. The harness_log.md historical references are LEFT INTACT (not silently rewritten — those are audit trail)
