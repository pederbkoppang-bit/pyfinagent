---
step: phase-16.37
title: vitest extractUrl + stdlib-shadow regression bundle (#51, #52)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/scripts/audit/lighthouse-wrapper.js (add module.exports + guard)
  - frontend/scripts/audit/lighthouse-wrapper.test.js (4 vitest cases)
  - frontend/vitest.config.ts (extend include glob)
  - tests/regression/__init__.py + test_no_calendar_shadow.py
  - 2 stale docstring fixes in backend __init__.py files
---

# Sprint Contract -- phase-16.37

## Research-gate summary

`handoff/current/phase-16.37-research-brief.md`. tier=simple, 7 in-full,
17 URLs, recency scan present, gate_passed=true. 8 internal files inspected.

## Bundled scope (2 task-list items)

| # | Task | Surface |
|---|------|---------|
| #51 | vitest unit test for `extractUrl()` argv translator | wrapper +1 export, +1 vitest config line, new test file (4 cases) |
| #52 | Stdlib-shadow regression test + cosmetic docstring fixes | new `tests/regression/` dir, 1 pytest file, 2 docstring edits |

## Hypothesis

Both follow-ups are mechanical hardening:
- #51 closes a regression-risk gap from 16.33 (lighthouse-wrapper had no
  unit coverage; an argv-parser change could silently break the
  `--url X` -> positional translation)
- #52 closes a regression-risk gap from 16.34 (the
  `backend/calendar -> backend/econ_calendar` rename has no
  CI guardrail to prevent re-shadowing)

## Concrete plan

### #51 (vitest extractUrl)

1. **Wrapper export.** Append to `frontend/scripts/audit/lighthouse-wrapper.js`:
   ```js
   if (require.main !== module) {
     module.exports = { extractUrl };
   }
   ```
   The `require.main !== module` guard prevents lighthouse from launching when the file is imported by vitest.

2. **Vitest config glob.** Edit `frontend/vitest.config.ts` line 14
   include array to add `"scripts/**/*.test.{js,ts}"`.

3. **New test file.** Create
   `frontend/scripts/audit/lighthouse-wrapper.test.js` with
   `// @vitest-environment node` (overrides global jsdom). 4 cases:
   - `extracts --url X positional form`
   - `extracts --url=X equals form`
   - `returns null when no --url arg`
   - `treats trailing --url as rest arg (loop bound check)`

### #52 (stdlib-shadow + docstrings)

1. **Docstring fixes (2 files).**
   - `backend/econ_calendar/sources/__init__.py:4-5`:
     replace "backend.calendar.get_sources()" -> "backend.econ_calendar.get_sources()" and "backend.calendar.sources" -> "backend.econ_calendar.sources".
   - `backend/services/observability/__init__.py:5`:
     replace "backend/calendar/sources/*.py" -> "backend/econ_calendar/sources/*.py".

2. **Regression test dir.** Create:
   - `tests/regression/__init__.py` (empty)
   - `tests/regression/test_no_calendar_shadow.py`

3. **Test design.** Use `subprocess.run` with `cwd=REPO_ROOT/"backend"`
   and `[sys.executable, "-c", "import calendar; print(calendar.__file__)"]`.
   Assert the output path does NOT contain "econ_calendar" AND DOES
   contain a stdlib indicator (`python` substring or `lib` substring).
   Bonus assertion: `calendar` is in `sys.stdlib_module_names`
   (Python 3.10+ check that calendar is a recognized stdlib).

   Three test cases:
   - `test_calendar_imports_stdlib_when_cwd_is_backend`
   - `test_calendar_in_stdlib_module_names`
   - `test_no_backend_calendar_directory_exists` (asserts
     `Path("backend/calendar")` does NOT exist; only
     `backend/econ_calendar/`)

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent && \
! grep -rn "backend\.calendar\|backend/calendar" backend/ docs/ scripts/ 2>/dev/null | grep -v "__pycache__" | grep -v ".pyc" | grep -v "backend/econ_calendar" && \
python -m pytest tests/regression/test_no_calendar_shadow.py -v && \
cd frontend && npx vitest run scripts/audit/lighthouse-wrapper.test.js
```

Plus:
- `wrapper_exports_extractUrl`: `require('./scripts/audit/lighthouse-wrapper').extractUrl` is a function.
- `vitest_glob_extended`: `frontend/vitest.config.ts` include array contains `scripts/**` pattern.
- `wrapper_test_passes`: 4/4 vitest cases PASS.
- `regression_test_passes`: 3/3 pytest cases PASS.
- `docstrings_clean`: zero `backend.calendar` or `backend/calendar` occurrences in source/docs/scripts (excluding archive + pyc + econ_calendar).

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. `require.main !== module` guard added (prevents lighthouse launch
   when imported by vitest).
3. `// @vitest-environment node` annotation present in wrapper test.
4. Vitest config glob extension covers `scripts/**` (no other
   accidental include changes).
5. 4 vitest cases + 3 pytest cases all PASS.
6. Stale docstring grep returns 0 hits (the verification grep above
   filters out econ_calendar -- the substring "backend/econ_calendar"
   contains "backend/" not "backend/calendar" so should not match).
7. `tests/regression/` dir created with proper `__init__.py`.
8. No mutation to lighthouse-wrapper.js logic outside the export
   addition; no mutation to other backend code outside the 2
   docstring fixes.
