## Research: phase-16.37 -- vitest unit test for extractUrl() + stdlib-shadow regression test

Tier assumed: simple (as specified by caller).

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://vitest.dev/guide/environment | 2026-04-25 | Official doc | WebFetch | "node is default environment"; per-file override via `// @vitest-environment node` docblock |
| https://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html | 2026-04-25 | Authoritative blog (Nick Coghlan, CPython core dev) | WebFetch | "using a local module name that shadows the name of a standard library or third party package" — sys.path[0] (cwd) checked before stdlib |
| https://docs.python.org/3/reference/import.html | 2026-04-25 | Official doc | WebFetch | sys.path ordering: cwd `''` first, then stdlib; `sys.modules` checked first on every import |
| https://realpython.com/python-import/ | 2026-04-25 | Authoritative blog | WebFetch | Detection pattern: inspect `module.__file__` after import; local shadow shows local path not stdlib path |
| https://www.pkgpulse.com/blog/vitest-3-vs-jest-30-2026 | 2026-04-25 | Industry blog | WebFetch | Vitest 3 `node` env is default for pure Node.js; "5.6x faster cold starts" over Jest; ESM-native |
| https://github.com/vitest-dev/vitest/discussions/2324 | 2026-04-25 | GitHub discussion (vitest maintainer) | WebFetch | Per-file env annotation confirmed: `// @vitest-environment node` overrides global jsdom env for a single file |
| https://betterstack.com/community/guides/testing/vitest-explained/ | 2026-04-25 | Authoritative community guide | WebFetch | vitest works with Node.js built-ins; `import { readFile } from 'fs/promises'` pattern; `npm test -- -t 'pattern'` for selective runs |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://vitest.dev/config/ | Official doc | Config overview page did not render detail content via WebFetch; key options obtained from environment subpage |
| https://vitest.dev/guide/ | Official doc | Getting Started page does not cover env config; used as index only |
| https://github.com/vitest-dev/vitest/discussions/4954 | GitHub discussion | Snippet sufficient -- "how to pass args to node process during tests" pattern noted |
| https://www.npmjs.com/package/mock-argv | npm package | Snippet sufficient -- mock-argv pattern noted; not needed since extractUrl takes argv as parameter |
| https://github.com/gforcada/flake8-builtins/issues/120 | GitHub issue | Snippet -- flake8-builtins shadow detection approach; different from subprocess-based regression test |
| https://bugs.python.org/issue22172 | Python bug tracker | Snippet -- confirmed sys.path[0]-first ordering is long-known behavior, not a bug to be fixed |
| https://github.com/microsoft/pylance-release/issues/2537 | GitHub issue | Snippet -- Pylance static shadow detection; not applicable to runtime pytest check |
| https://mail.python.org/pipermail/python-dev/2015-November/142096.html | Python-dev ML | Snippet -- historical context on stdlib shadow-naming incidents |
| https://pypi.org/project/pytest-regressions/ | PyPI | Snippet -- data/file regression plugin; not relevant for import-path check |
| https://tenthousandmeters.com/blog/python-behind-the-scenes-11-how-the-python-import-system-works/ | Blog | Snippet -- deep CPython internals; not needed beyond sys.path ordering |

### Recency scan (2024-2026)

Searched: "vitest unit test pure node js script argv parser 2026", "python stdlib shadow detection pytest regression test 2025", "vitest per-file environment annotation node jsdom 2025".

Result: No findings in the 2024-2026 window supersede the canonical approach. The per-file `@vitest-environment node` annotation and `__file__`-path assertion techniques are stable and well-established. The pkgpulse.com 2026 Vitest 3 vs Jest comparison confirms Vitest's node environment remains unchanged and recommended for pure Node.js scripts.

---

### Key findings

1. **extractUrl is a named function, not a module.exports export.** It is defined at `frontend/scripts/audit/lighthouse-wrapper.js:23` as `function extractUrl(argv)` but there is no `module.exports = { extractUrl }` -- it is consumed immediately at line 66 (`const { url, rest } = extractUrl(process.argv.slice(2))`). To test it in isolation the wrapper must either export it, OR the test can replicate the function inline (less ideal). Adding `if (require.main !== module) { module.exports = { extractUrl }; }` is the lightest-touch CJS export pattern. (Source: internal read, lighthouse-wrapper.js:23-40)

2. **vitest is already installed (^4.1.4) as a devDependency.** No new npm dep needed. (Source: frontend/package.json:51)

3. **vitest.config.ts constrains include to `src/**/*.{test,spec}.{ts,tsx}`** -- scripts/audit/ tests are outside this glob. Two options: (a) add `scripts/**/*.test.{js,ts}` to the include array, or (b) place the test at `frontend/src/` (wrong semantically). Adding to include is correct. (Source: frontend/vitest.config.ts:14)

4. **vitest.config.ts sets `environment: "jsdom"` globally** -- the wrapper test must override to node env via `// @vitest-environment node` at the top of the test file, otherwise DOM globals bleed in. (Source: vitest.dev/guide/environment, confirmed by vitest-dev/vitest discussion #2324)

5. **vitest.setup.ts imports `@testing-library/jest-dom/vitest`** -- this is a jsdom-specific setup. The per-file node environment annotation skips the setup file effects for that file per vitest docs. No conflict.

6. **Two stale docstrings found referencing old `backend.calendar` path:**
   - `backend/econ_calendar/sources/__init__.py:4-5`: docstring reads "backend.calendar.get_sources()" and "backend.calendar.sources" -- should be "backend.econ_calendar.get_sources()" / "backend.econ_calendar.sources". (Source: internal read, econ_calendar/sources/__init__.py:1-6)
   - `backend/services/observability/__init__.py:5`: docstring reads "backend/calendar/sources/*.py" -- should be "backend/econ_calendar/sources/*.py". (Source: internal read, observability/__init__.py:5)
   - `backend/autoresearch/monthly_champion_challenger.py:23`: `import calendar` -- this is a VALID stdlib import (for month-name lookup etc.), NOT a stale reference to the old package. Confirmed safe. (Source: grep output line 1)

7. **No `tests/regression/` directory exists.** New test file should go at `tests/regression/test_no_calendar_shadow.py`. The `tests/` directory exists with other tests. (Source: Bash find output)

8. **Python stdlib shadow test design.** The correct approach is a subprocess call: `subprocess.run([sys.executable, "-c", "import calendar; import sys; print(calendar.__file__)"], cwd=REPO_ROOT / "backend")`. Assert the output path contains the Python executable's stdlib path (e.g., contains `python3` or `lib/python`) and does NOT contain `econ_calendar`. The `sys.stdlib_module_names` frozenset (Python 3.10+) can be used as an additional sanity check that `calendar` is known-stdlib. (Sources: docs.python.org/3/reference/import.html; realpython.com/python-import/)

9. **extractUrl test cases.** Because the function takes argv as a parameter (not process.argv directly), it is purely functional with no side effects and no subprocess dependency. Test cases: (a) `--url X rest-flags` -> `{url: "X", rest: ["rest-flags"]}`, (b) `--url=X rest-flags` -> same, (c) no `--url` arg -> `{url: null, rest: [...]}`, (d) `--url` as last arg (edge) -> `{url: null, rest: ["--url"]}` (the loop condition `i+1 < argv.length` handles this -- url remains null since condition fails). Wait -- re-reading line 28: `if (a === "--url" && i + 1 < argv.length)` -- if `--url` is the last arg, condition is false, so it falls through to `out.push(a)`. url stays null, rest includes `--url`. Worth a test case. (Source: lighthouse-wrapper.js:26-39)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/scripts/audit/lighthouse-wrapper.js` | 87 | Argv translator + spawnSync lighthouse | extractUrl not exported; needs `module.exports` guard |
| `frontend/vitest.config.ts` | 18 | Vitest config | `include` glob covers only `src/**`; needs `scripts/**` added |
| `frontend/vitest.setup.ts` | 1 | jsdom setup | Not relevant for node-env test file |
| `frontend/package.json` | 53 | Package manifest | vitest ^4.1.4 already in devDependencies |
| `frontend/scripts/run-test.mjs` | 38 | npm test wrapper | Translates `--filter=X` to vitest positional; works for scripts/** once include glob is updated |
| `backend/econ_calendar/sources/__init__.py` | 11 | Calendar source registry | Lines 4-5 docstring stale: "backend.calendar" should be "backend.econ_calendar" |
| `backend/services/observability/__init__.py` | ~20 | Observability primitives | Line 5 docstring stale: "backend/calendar/sources" should be "backend/econ_calendar/sources" |
| `tests/regression/` | n/a | Regression test dir | Does NOT exist yet; must be created with `__init__.py` |

---

### Consensus vs debate

No debate. The per-file `@vitest-environment node` annotation is the documented, maintainer-confirmed approach for running node-env tests when the global config uses jsdom. The `module.__file__` assertion pattern for stdlib-shadow detection is canonical across CPython documentation and the community.

### Pitfalls (from literature)

1. **Do not import lighthouse-wrapper.js at module level without the `require.main !== module` guard.** The file calls `spawnSync` and `process.exit` at module scope (lines 66-86). If required/imported without the guard, it will try to launch lighthouse immediately. The export guard is mandatory. (Source: lighthouse-wrapper.js:66-86)
2. **Vitest jsdom global bleed.** Without `// @vitest-environment node`, the test runs under jsdom and browser globals may interfere with CJS require resolution or spawnSync behavior. (Source: vitest.dev/guide/environment)
3. **sys.path ordering under pytest.** When running pytest from repo root, `sys.path` may include `backend/` as a root, which could still shadow `calendar`. The subprocess approach (`cwd=REPO_ROOT/backend`) is more robust than direct import in the test process because it simulates exactly what a developer running `python` from that dir would get. (Source: docs.python.org/3/reference/import.html)
4. **`import calendar` in autoresearch/monthly_champion_challenger.py is valid.** Confirmed stdlib use, not a leftover from the old package. Do not touch. (Source: internal grep)

---

### Application to pyfinagent

**Task #51 -- vitest test for extractUrl:**
1. Edit `frontend/scripts/audit/lighthouse-wrapper.js`: add at bottom:
   ```js
   if (require.main !== module) {
     module.exports = { extractUrl };
   }
   ```
2. Add `"scripts/**/*.test.{js,ts}"` to the `include` array in `frontend/vitest.config.ts` (file:line frontend/vitest.config.ts:14).
3. Create `frontend/scripts/audit/lighthouse-wrapper.test.js` with `// @vitest-environment node` at top, 4 test cases.
4. Run via: `cd frontend && npx vitest run scripts/audit/lighthouse-wrapper.test.js`

**Task #52 -- docstring cleanup + stdlib shadow test:**
1. Fix `backend/econ_calendar/sources/__init__.py:4-5` -- replace "backend.calendar" with "backend.econ_calendar".
2. Fix `backend/services/observability/__init__.py:5` -- replace "backend/calendar/sources/*.py" with "backend/econ_calendar/sources/*.py".
3. Create `tests/regression/__init__.py` (empty) and `tests/regression/test_no_calendar_shadow.py` using subprocess + `__file__` assertion.
4. Run via: `python -m pytest tests/regression/test_no_calendar_shadow.py -v`

**Proposed verification command (adjusted from masterplan prompt):**
```bash
cd /Users/ford/.openclaw/workspace/pyfinagent && \
! grep -rn "backend\.calendar\|backend/calendar" backend/ docs/ scripts/ 2>/dev/null | grep -v ".pyc" | grep -v "__pycache__" && \
python -m pytest tests/regression/test_no_calendar_shadow.py -v && \
cd frontend && npx vitest run scripts/audit/lighthouse-wrapper.test.js
```

Note: `backend/econ_calendar/` paths contain the substring "backend/econ_calendar" which does NOT match "backend/calendar" (no "econ_") so the grep is safe.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total incl. snippet-only (17 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (wrapper, config, package.json, setup, existing tests, both stale docstrings, autoresearch false-positive)
- [x] Contradictions/consensus noted (none; consensus)
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/phase-16.37-research-brief.md",
  "gate_passed": true
}
```
