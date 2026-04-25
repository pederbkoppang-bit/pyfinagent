---
step: phase-16.34
cycle_date: 2026-04-24
verdict: PASS
agent: qa
---

# Q/A Critique -- phase-16.34

## Harness-compliance (5 items)

1. **Research gate**: PASS. `phase-16.34-research-brief.md` exists,
   tier=moderate, `gate_passed: true`, 8 external sources read in full
   (>=5 floor), 18 URLs total, recency scan performed (2025/2026
   sources cited), 13 internal files inspected. Brief reflects deeper
   work appropriate for moderate tier (root-cause analysis of stdlib
   shadow + Git history-preservation strategy + tool selection).
2. **Contract-before-GENERATE**: PASS. `contract.md` frontmatter
   `step: phase-16.34` (not stale 16.33). Research-gate summary cites
   the brief.
3. **Experiment results**: PASS. `experiment_results.md` frontmatter
   `step: phase-16.34`. Documents file rename + 13 import-line
   updates.
4. **Log-last**: PASS. `grep -c "phase-16.34" handoff/harness_log.md`
   = 0. Log not yet appended -- correct ordering (Q/A first, then
   log, then status flip).
5. **No verdict-shopping**: PASS. Prior critique was for 16.33
   (separate step, PASS). This is a fresh first Q/A spawn for 16.34.

## Deterministic checks

- old_dir_gone: yes (`backend/calendar/` removed)
- new_dir_exists: yes, file count: 7 entries
  (__init__.py, blackout.py, normalize.py, registry.py, sources/,
  watcher.py, __pycache__)
- live_refs_remain: 0 (`from backend.calendar` / `import backend.calendar`
  returns empty across `backend/`, `scripts/`, `frontend/`)
- cd_backend_pytest_exit: 0, passed: 7/7
  (`tests/api/test_sovereign.py` -- the long-broken command now
  works as written in masterplan 10.5.0)
- repo_root_pytest: 182 passed, 1 skipped, 7 warnings (no
  regression vs prior baseline)
- calendar_watcher_tests: 10/10 passed (audit prompt said 9; actual
  is 10, so coverage is one better than expected)
- git_log_follow_history: git status shows `RM backend/calendar/* ->
  backend/econ_calendar/*` (rename detected at >50% similarity --
  Git history preservation confirmed via the rename indicator)

## Stdlib-shadow elimination

- import_calendar_resolves_to_stdlib: yes
- stdlib_path:
  `/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/calendar.py`
- Critical: this is the smoking-gun proof the shadow is gone.
  Before the rename, `cd backend && python -c "import calendar"`
  resolved to `backend/calendar/__init__.py` and triggered a
  circular `ModuleNotFoundError`. Now resolves cleanly to stdlib.

## Cosmetic-references audit

- observability_init_clean: yes (line 5 reference is inside the
  module docstring as a path example for future contributors --
  not a live import)
- migration_script_clean: yes (line 7 reference is inside the
  module docstring describing where `event_id` is computed -- not a
  live import)
- Note: `backend/econ_calendar/sources/__init__.py` lines 4-5
  contain a docstring still saying `backend.calendar.get_sources()`
  / `backend.calendar.sources` -- this is INSIDE the renamed
  package's own docstring. It's stale (should now say
  `backend.econ_calendar`) but functionally inert -- no import
  resolution depends on it. Flagged as a cosmetic follow-up, not a
  blocker.

## Final-sweep (catch dynamic imports / string refs)

- final_grep_returns_empty: no, but ALL 3 hits are docstring/comment
  prose, NOT live imports:
  - `backend/services/observability/__init__.py:5` -- docstring
  - `scripts/migrations/add_calendar_events_schema.py:7` -- docstring
  - `backend/econ_calendar/sources/__init__.py:4-5` -- docstring
    inside the renamed package itself (newly stale)
- No `importlib`, no string-based `__import__`, no
  `"backend.calendar.X"` literal anywhere. Mechanical migration is
  complete for live code.

## LLM judgment

- **doc_only_left_alone_defensible**: Defensible scope discipline.
  Touching docstrings would require re-running tests and balloons
  the commit beyond the immutable success criterion. The bigger
  miss is that the rename surfaced a NEW stale docstring inside
  `econ_calendar/sources/__init__.py` that wasn't on Main's
  10-file enumeration -- a 1-minute follow-up to also rewrite that
  package's own docstring. Recommend FOLLOW-UP TICKET, not blocker.
- **regression_test_for_shadow_concern**: No test currently asserts
  "no `backend/X/` directory shadows a stdlib name". A defensive
  follow-up would be a tiny test in
  `backend/tests/test_no_stdlib_shadow.py` iterating `backend/*/`
  and asserting none collides with `sys.stdlib_module_names`.
  Suggest adding as a 16.35 follow-up.
- **not_committed_decision**: Defensible per CLAUDE.md "never
  commit unless asked". The `git mv` rename is currently staged
  (visible in `git status`). Peder should be prompted to commit
  this as a single atomic change before any further refactor lands
  -- otherwise the rename can drift.
- **9_final_genuinely_closed**: YES. The masterplan 10.5.0
  immutable verification command (`cd backend && pytest
  tests/api/test_sovereign.py`) now exits 0 with 7/7 passing. This
  was the third and last broken verification command from #9. The
  follow-up should be closeable.
- **mechanical_migration_complete**: YES for live code paths. No
  dynamic imports, no string refs, no `importlib`. The 3 remaining
  grep hits are all prose inside docstrings/comments.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All immutable criteria met: (1) cd backend pytest exits 0 with 7/7 PASS, (2) repo-root pytest 182 passed unchanged, (3) zero live `from backend.calendar` refs across backend/scripts/frontend, (4) `import calendar` resolves to stdlib (shadow eliminated), (5) git mv preserved history. Research-gate pass with 8 in-full sources, contract step matches, log not yet appended (correct order).",
  "violated_criteria": [],
  "violation_details": [],
  "follow_up_tickets": [
    "(cosmetic) Rewrite `backend/econ_calendar/sources/__init__.py` docstring to say `backend.econ_calendar.get_sources()` instead of stale `backend.calendar.get_sources()` -- 1-line fix",
    "(cosmetic) Optionally rewrite docstrings in `backend/services/observability/__init__.py:5` and `scripts/migrations/add_calendar_events_schema.py:7` to reflect new path",
    "(defensive) Add `backend/tests/test_no_stdlib_shadow.py` asserting no `backend/<name>/` collides with `sys.stdlib_module_names` -- prevents future regressions",
    "(operational) Peder to commit the staged rename as a single atomic commit per CLAUDE.md two-commit Git history-preservation strategy"
  ],
  "certified_fallback": false,
  "checks_run": [
    "directory_rename",
    "live_import_grep",
    "cd_backend_pytest_exit_code",
    "repo_root_pytest_regression",
    "calendar_watcher_unit_tests",
    "stdlib_shadow_resolution",
    "cosmetic_refs_audit",
    "dynamic_import_sweep",
    "research_gate_compliance",
    "contract_pre_generate",
    "log_last_ordering",
    "no_verdict_shopping"
  ]
}
```
