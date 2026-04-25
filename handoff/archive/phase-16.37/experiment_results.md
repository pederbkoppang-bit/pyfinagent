---
step: phase-16.37
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/scripts/audit/lighthouse-wrapper.js (+ require.main guard + module.exports)
  - frontend/scripts/audit/lighthouse-wrapper.test.mjs (5 vitest cases)
  - frontend/vitest.config.ts (extended include glob)
  - tests/regression/__init__.py + test_no_calendar_shadow.py (3 pytest cases)
  - 3 cosmetic docstring fixes in backend + scripts/migrations
---

# Experiment Results -- phase-16.37

## What was done

Bundle of 2 follow-ups (#51 and #52) — both small regression-hardening
gaps from prior 16.x cycles.

### #51: vitest extractUrl test

1. **Wrapper export.** `frontend/scripts/audit/lighthouse-wrapper.js`:
   added `if (require.main !== module) { module.exports = { extractUrl }; }`
   guard so importing the file from a test does NOT spawn lighthouse.
   The original `else` branch retains the spawn behavior for direct CLI
   invocation.

2. **Vitest config glob.** `frontend/vitest.config.ts:16` extended
   include array to `["src/**/*.{test,spec}.{ts,tsx}",
   "scripts/**/*.test.{js,mjs,ts}"]` so vitest discovers tests under
   `scripts/`.

3. **Test file (ESM).** Created `lighthouse-wrapper.test.mjs` (NOT .js)
   because vitest 4.x is ESM-only — the `import { describe, it,
   expect } from "vitest"` form works only in ESM modules. Used
   `createRequire(import.meta.url)` to import the CJS wrapper without
   forcing it to ESM.

   5 test cases (target was 4):
   - `extracts --url X positional form and preserves rest`
   - `extracts --url=X equals form and preserves rest`
   - `returns null url when no --url arg present`
   - `treats trailing --url as a rest arg when no value follows`
   - `handles --url=X mixed with other flags before and after`

### #52: stdlib-shadow regression test + docstring cleanup

1. **Docstring fixes (3, not 2 as researcher initially identified).**
   - `backend/econ_calendar/sources/__init__.py:4-5`:
     "backend.calendar.get_sources()" -> "backend.econ_calendar.get_sources()"
     and ".sources" suffix updated.
   - `backend/services/observability/__init__.py:5`:
     "backend/calendar/sources/*.py" -> "backend/econ_calendar/sources/*.py".
   - `scripts/migrations/add_calendar_events_schema.py:7`:
     "backend/calendar/normalize.py" -> "backend/econ_calendar/normalize.py"
     (researcher missed this one; caught during verification grep).

2. **Regression test dir + file.** Created:
   - `tests/regression/__init__.py` (empty marker).
   - `tests/regression/test_no_calendar_shadow.py` (101 lines).

3. **Test design.** Three pytest cases:
   - `test_calendar_imports_stdlib_when_cwd_is_backend`: subprocess
     with `cwd=backend` importing `calendar`; asserts resolved
     `__file__` does NOT contain "econ_calendar" and IS in stdlib path.
   - `test_calendar_in_stdlib_module_names`: assert `"calendar" in
     sys.stdlib_module_names` (Python 3.10+ canonical registry).
   - `test_no_backend_calendar_directory_exists`: `Path("backend/calendar")`
     must not exist; `Path("backend/econ_calendar")` must.

### Files touched

| Path | Action | Size |
|------|--------|------|
| `frontend/scripts/audit/lighthouse-wrapper.js` | edited | +12 lines (guard + module.exports) |
| `frontend/scripts/audit/lighthouse-wrapper.test.mjs` | CREATED | 60 lines |
| `frontend/vitest.config.ts` | edited | +1 glob pattern |
| `tests/regression/__init__.py` | CREATED | 0 lines (marker) |
| `tests/regression/test_no_calendar_shadow.py` | CREATED | 101 lines |
| `backend/econ_calendar/sources/__init__.py` | edited | docstring (2 lines) |
| `backend/services/observability/__init__.py` | edited | docstring (1 line) |
| `scripts/migrations/add_calendar_events_schema.py` | edited | docstring (1 line) |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |

## Verification (verbatim, immutable)

```
$ ! grep -rn "backend\.calendar\|backend/calendar" backend/ docs/ scripts/ 2>/dev/null | grep -v "__pycache__" | grep -v ".pyc" | grep -v "backend/econ_calendar" && \
  echo "stale-ref grep clean" && \
  python -m pytest tests/regression/test_no_calendar_shadow.py -v && \
  cd frontend && npx vitest run scripts/audit/lighthouse-wrapper.test.mjs
stale-ref grep clean
collected 3 items
test_no_calendar_shadow.py::test_calendar_imports_stdlib_when_cwd_is_backend PASSED
test_no_calendar_shadow.py::test_calendar_in_stdlib_module_names PASSED
test_no_calendar_shadow.py::test_no_backend_calendar_directory_exists PASSED
3 passed in 0.02s
 RUN  v4.1.4 /Users/ford/.openclaw/workspace/pyfinagent/frontend
 Test Files  1 passed (1)
      Tests  5 passed (5)
```

**Result: PASS.** Compound `&&` exits 0. Stale-ref grep clean. 3 pytest +
5 vitest tests all PASS.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | wrapper_exports_extractUrl | PASS | `require('./...').extractUrl` is a function (vitest tests prove it) |
| 2 | vitest_glob_extended | PASS | include array contains `scripts/**/*.test.{js,mjs,ts}` |
| 3 | wrapper_test_passes | PASS | 5/5 (target was 4) |
| 4 | regression_test_passes | PASS | 3/3 |
| 5 | docstrings_clean | PASS | grep returns 0 hits |

## Honest disclosures

1. **Test file is `.mjs` not `.js`.** Vitest 4.x is ESM-only; `.js`
   files in this repo default to CommonJS (no `"type": "module"` in
   `package.json`). Renamed `.test.js` -> `.test.mjs` after the first
   run failed with "Vitest cannot be imported in a CommonJS module".
   Vitest config glob extended to include `.mjs` accordingly.

2. **5 vitest cases, contract said 4.** Added a 5th
   (`handles --url=X mixed with other flags before and after`) for the
   interleaving edge case. Exceeds floor; not a violation.

3. **3 docstring fixes, contract said 2.** Verification grep caught a
   third stale reference at
   `scripts/migrations/add_calendar_events_schema.py:7` that the
   researcher's initial inventory missed (it was in `scripts/`, not
   `backend/`). Fixed inline.

4. **CJS-from-ESM interop pattern.** Used `createRequire(import.meta.url)`
   to import the CJS wrapper from the ESM test. Standard Node.js
   pattern; no hack.

5. **`test_calendar_imports_stdlib_when_cwd_is_backend`** uses a
   defensive `or "/lib/" in resolved_path` clause. The macOS Python
   stdlib path on this dev machine is e.g.
   `/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/...`
   — contains "lib" but not "python" alone. Both substrings checked.

6. **No mutation to wrapper logic.** Only the export + guard added.
   The argv-translation behavior tested by the new vitest cases is the
   exact same code that's been in production since 16.33.

## Closes

- Task list items #51 and #52
- masterplan step **phase-16.37**

## Next

Spawn Q/A. If PASS: log + flip + continue with the next bundle.
