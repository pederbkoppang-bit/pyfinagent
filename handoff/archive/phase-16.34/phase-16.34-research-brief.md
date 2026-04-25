---
phase: "16.34"
step: "Rename backend/calendar -> backend/econ_calendar"
tier: moderate
researcher: claude-sonnet-4-6
date: 2026-04-24
gate_passed: true
---

## Research: phase-16.34 — backend/calendar -> backend/econ_calendar rename

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html | 2026-04-24 | doc/blog | WebFetch | "Never add a package directory inside a package directly to the Python path." CWD as sys.path[0] is the exact mechanism causing shadowing; names matching stdlib are loaded first. |
| https://docs.pytest.org/en/stable/explanation/pythonpath.html | 2026-04-24 | official doc | WebFetch | pytest `prepend` mode (default) inserts the directory of each test module to the front of sys.path; `python -m pytest` adds CWD to sys.path by std Python behaviour. Either way `backend/` lands at sys.path[0] when invoked as `cd backend && python -m pytest`. |
| https://www.py4u.org/blog/python-problem-with-local-modules-shadowing-global-modules/ | 2026-04-24 | blog | WebFetch | Concrete explanation: `calendar/` in CWD causes `import calendar` to resolve to the local package rather than stdlib. Fix #1 (recommended): rename the local module. |
| https://thelinuxcode.com/git-move-files-practical-renames-refactors-and-history-preservation-in-2026/ | 2026-04-24 | blog (2026) | WebFetch | Two-commit strategy: (1) `git mv` move-only commit so Git's rename detector fires at >50% similarity threshold; (2) separate import-update commit. Mixing heavy edits with the move causes Git to treat it as delete+add, breaking `git log --follow`. |
| https://medium.com/@rajsek/proper-way-to-rename-a-directory-in-git-repository-5bdec4c9cfd0 | 2026-04-24 | blog | WebFetch | `git mv <old> <new>` works for directories, preserves history. Case-sensitive renames need a two-step temp-name approach (not applicable here). |
| https://github.com/jlevy/repren | 2026-04-24 | tool/code | WebFetch | repren is a text-pattern refactor tool; can rename import paths simultaneously across file contents + filesystem. Not AST-aware but sufficient for unambiguous prefix patterns like `backend.calendar`. Dry-run mode available. |
| https://docs.python.org/3/tutorial/modules.html | 2026-04-24 | official doc (Python 3.14) | WebFetch | "The directory containing the input script (or the current directory when no file is specified) is placed at the beginning of the search path, ahead of the standard library path." This is the canonical statement of why `backend/calendar/` shadows stdlib `calendar`. |
| https://jdhao.github.io/2025/05/13/pytest_sys_path_issues/ | 2026-04-24 | blog (2025) | WebFetch | pytest rootdir != sys.path; `--rootdir` does not fix import errors. Only renaming or pyproject.toml `pythonpath` setting resolves the root cause. Confirmed `python -m pytest` adds CWD to sys.path via standard Python behaviour. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://mail.python.org/pipermail/python-dev/2015-November/142096.html | mailing list | 2015 vintage; covered by Coghlan's notes above |
| https://hackernoon.com/why-refactoring-how-to-restructure-python-package-51b89aa91987 | blog | 403 Forbidden |
| https://realpython.com/python-refactoring/ | blog | Covered by other sources; general refactoring, not package rename |
| https://github.com/microsoft/pylance-release/issues/127 | issue tracker | IDE-specific auto-rename; not applicable to CLI workflow |
| https://github.com/python/pymanager/security/advisories/GHSA-jr5x-hgm4-rrm6 | advisory | Security angle on CWD hijacking; not the pattern here |
| https://libcst.readthedocs.io/en/latest/why_libcst.html | doc | LibCST is appropriate for complex codemods; overkill for a single unambiguous prefix swap |
| https://sqlpey.com/git/git-file-move-history-preservation/ | blog | Duplicate coverage of git mv history topic |
| https://ankursingh.hashnode.dev/the-hitchhikers-guide-to-syspath | blog | sys.path tutorial; fully covered by Python docs above |
| https://linuxctl.com/p/git-preserve-history-when-moving-files/ | blog | Covered by thelinuxcode article above |
| https://www.golinuxcloud.com/git-rename-file-or-directory/ | blog | Basic git mv tutorial; no new information |

### Recency scan (2024-2026)

Searched: "Python package rename refactoring git mv import rewrite 2025", "libcst rope AST import rewrite Python package rename 2025 2026", "git mv rename Python package history preservation 2026", "Python pytest sys.path CWD package shadowing conftest.py fix 2025".

Result: two new 2025-2026 findings complement the canonical sources:
1. thelinuxcode.com (2026) explicitly codifies the two-commit strategy as current best practice, which refines older advice that bundled move + edit in one commit.
2. jdhao.github.io (2025-05-13) confirms pytest sys.path issues are still present in 2025 and that `--rootdir` is NOT a fix; only renaming or `pythonpath` config resolves the root cause.

No 2024-2026 work supersedes the core shadowing mechanism (Python docs are authoritative and stable across versions 3.8-3.14). No new tooling (libcst 1.8.6, rope 2026) changes the conclusion that sed-style substitution is sufficient for this specific single-prefix rename.

---

### Key findings

1. **Root cause confirmed** — When `cd backend && python -m pytest tests/api/test_sovereign.py`, Python adds `backend/` to `sys.path[0]` (standard Python behaviour for `-m` invocation). Any bare `import calendar` in that process resolves to `backend/calendar/__init__.py` instead of `cpython/.../calendar.py`. The local `__init__.py` immediately attempts `from backend.calendar.registry import ...`, which fails because `backend` is NOT on sys.path at that point (`backend/` is, not the parent directory). (Source: Python 3.14 docs, https://docs.python.org/3/tutorial/modules.html; confirmed by live test below.)

2. **Shadow confirmed by live test** — Running `cd backend && .venv/bin/python -c "import calendar"` produces: `ModuleNotFoundError: No module named 'backend.calendar'` at `backend/calendar/__init__.py:8`. This is the exact failure mode: stdlib calendar is shadowed, the local package's `__init__.py` loads instead, and then FAILS because `backend` is not resolvable from that sys.path. (Internal verification, 2026-04-24.)

3. **Fix: rename the package** — Renaming `backend/calendar/` to `backend/econ_calendar/` eliminates the name collision. There is no other package in `backend/` that conflicts with a stdlib module name (`agents`, `api`, `config`, `news`, `services`, `tasks`, `tests`, `models`, `utils`, `tools`, `backtest`, `metrics`, `db`, `alt_data`, `autoresearch`, `governance`, `intel`, `mcp`, `meta_evolution` — none are stdlib names). (Source: `ls backend/` + importlib check.)

4. **sed is safe for this rename** — The pattern `backend.calendar` is unambiguous: it only appears as a Python import prefix, not as a BQ table name (`calendar_events` is the table name — unrelated), not as a comment-only string that could cause false positives. The migration script references `backend/calendar/normalize.py` in a comment string (line 7) — that comment string is safe to rewrite. `grep -rn 'backend\.calendar\|backend/calendar'` returned exactly the files enumerated below; zero false-positive risk. (Source: internal grep, 2026-04-24.)

5. **Two-commit strategy is best practice** — Move commit first (`git mv`), then import-update commit. This ensures Git's rename detector (similarity threshold >50%) fires correctly, keeping `git log --follow backend/econ_calendar/` traversable. (Source: thelinuxcode.com 2026.)

6. **Side-effect import pattern** — `backend/calendar/__init__.py:27` has `from backend.calendar import sources as _sources  # noqa: F401` which is a deliberate side-effect import that triggers source registration. After rename this becomes `from backend.econ_calendar import sources as _sources`. The side-effect semantics are preserved; only the namespace changes. (Source: internal read of `__init__.py`.)

7. **No BQ schema changes needed** — BQ table name is `calendar_events` (defined in `backend/news/bq_writer.py:38`). This is a Python-layer rename only; the BQ table identifier does not embed the Python package name. The migration script `add_calendar_events_schema.py` references `backend/calendar/normalize.py` in a docstring comment only (line 7); that comment should be updated for accuracy but causes no runtime failure if missed.

---

### Internal code inventory

| File | Lines affected | Role | Status after rename |
|------|---------------|------|-------------------|
| `backend/calendar/__init__.py` | 8,14,18,19,27 | Package entry + side-effect sources import | Rename dir + update 5 import lines |
| `backend/calendar/sources/__init__.py` | 7 | Source registry side-effect loader | Rename dir + update 1 import line |
| `backend/calendar/watcher.py` | 21,22,26 | Calendar fetch orchestrator | Rename dir + update 3 import lines |
| `backend/calendar/sources/finnhub_earnings.py` | 26 | Finnhub earnings source | Rename dir + update 1 import line |
| `backend/calendar/sources/fed_scrape.py` | 26 | Fed FOMC scrape source | Rename dir + update 1 import line |
| `backend/calendar/sources/fred_releases.py` | 22 | FRED macro releases source | Rename dir + update 1 import line |
| `backend/tests/test_calendar_watcher.py` | 21,22,23,28 | Unit tests (9 test functions) | Update 4 import lines; no test logic changes |
| `scripts/smoketest/phase6_e2e.py` | 178 | E2E smoketest stage 5 | Update 1 import line |
| `scripts/migrations/add_calendar_events_schema.py` | 7 (comment only) | BQ schema migration | Update comment string for accuracy (non-critical) |
| `backend/services/observability/__init__.py` | 5 (docstring only) | Observability primitives | Update docstring for accuracy (non-critical) |

**Files with NO import-level changes needed:**
- `backend/news/bq_writer.py` — references `calendar_events` (BQ table name string), not the Python package; zero changes needed.
- `backend/calendar/registry.py` — no self-referential imports to `backend.calendar.*`
- `backend/calendar/blackout.py` — no self-referential imports to `backend.calendar.*`
- `backend/calendar/normalize.py` — no self-referential imports to `backend.calendar.*`

**Total import-bearing lines to update: 16 across 8 files** (plus 2 comment/docstring lines in 2 additional files).

**`backend/tests/tests/__init__.py` exists** — tests directory is already a proper package; no `conftest.py` is present anywhere in the project. pytest's `prepend` import mode applies.

---

### Consensus vs debate

**Consensus:** Rename is the correct fix. All five authoritative sources agree that renaming the shadowing module is the recommended first-line solution (Coghlan, py4u.org, Python docs). Alternative fixes (modifying sys.path in conftest.py, using `append` import mode, absolute imports) are fragile or non-applicable to this codebase structure.

**Minor debate:** sed vs AST tools. libcst/rope provide AST-awareness but are overkill for a single unambiguous prefix substitution. The pattern `backend\.calendar` has zero false-positive risk in this codebase (confirmed by grep). repren, sed, or Python `str.replace` with a targeted file list are all equally correct here.

### Pitfalls (from literature)

- P1: Do NOT mix the `git mv` commit with import edits. Git's rename detection requires content similarity >50%; editing all 16 import lines in the same commit as the move may cross the threshold but produces unreadable history. Two commits. (Source: thelinuxcode.com 2026.)
- P2: The `sources` sub-package also has `backend.calendar.sources` imports in its own `__init__.py`. Do not miss this file — it is a level deeper than the top-level calendar imports. (Internal audit.)
- P3: `backend/calendar/__init__.py:27` is a side-effect-only import (`_sources`). The rename must preserve this line (with updated namespace). Do NOT remove it thinking it is unused — the `# noqa: F401` marker signals intentional side-effect. (Internal audit.)
- P4: There is no `pytest.ini` / `pyproject.toml` in the project. pytest operates with default settings, meaning `prepend` import mode. The rename alone (no pytest config changes) is sufficient to fix the shadow. (Internal audit.)
- P5: `backend/tests/__init__.py` exists. Tests are a proper package. pytest will find the `backend` package root correctly from the repo root when invoked as `python -m pytest backend/tests/ -q`. No conftest.py manipulation needed.

---

### Application to pyfinagent — complete execution plan

**Verification command (immutable):**
```
cd backend && python -m pytest tests/api/test_sovereign.py -q 2>&1 | tail -3
```

**Step 1 — git mv (commit 1: move only)**
```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
git mv backend/calendar backend/econ_calendar
git commit -m "refactor: git mv backend/calendar -> backend/econ_calendar (step 1: move only)"
```

**Step 2 — sed rewrite (commit 2: import updates)**

Exact sed pattern (macOS BSD sed requires `-i ''`):

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent

# The 8 import-bearing files + 2 comment/docstring files
FILES=(
  backend/econ_calendar/__init__.py
  backend/econ_calendar/sources/__init__.py
  backend/econ_calendar/watcher.py
  backend/econ_calendar/sources/finnhub_earnings.py
  backend/econ_calendar/sources/fed_scrape.py
  backend/econ_calendar/sources/fred_releases.py
  backend/tests/test_calendar_watcher.py
  scripts/smoketest/phase6_e2e.py
  scripts/migrations/add_calendar_events_schema.py
  backend/services/observability/__init__.py
)

for f in "${FILES[@]}"; do
  sed -i '' 's/backend\.calendar/backend.econ_calendar/g' "$f"
  sed -i '' 's|backend/calendar|backend/econ_calendar|g' "$f"
done
```

**Why sed is safe here:**
- Pattern `backend\.calendar` appears exclusively as a Python import prefix; no BQ table names, no test regex strings, no URL strings contain this pattern.
- `grep -rn 'backend\.calendar\|backend/calendar'` returned exactly 10 files; all are enumerated above. Zero false-positive risk.
- `backend/news/bq_writer.py` uses `calendar_events` (no `backend.calendar` substring) — unaffected by the sed pattern.
- The sed pattern is not present in any test as a string literal being matched (confirmed: `grep -n 'calendar'` in test_calendar_watcher.py shows only import lines and the module docstring).

**Step 3 — verify**
```bash
# Immutable verification command
cd /Users/ford/.openclaw/workspace/pyfinagent/backend && python -m pytest tests/api/test_sovereign.py -q 2>&1 | tail -3

# No-regression check: full test suite from repo root
cd /Users/ford/.openclaw/workspace/pyfinagent && python -m pytest backend/tests/ -q 2>&1 | tail -5
```

**Why `cd backend && python -m pytest` no longer shadows after rename:**
`sys.path[0]` will be `backend/` as before, but `import calendar` will scan that directory and find NO `calendar/` subdirectory (it was renamed to `econ_calendar/`). Python falls through to the next sys.path entry which contains the stdlib `calendar.py`. The shadow is gone.

**Estimated execution time:** 3-5 minutes total (git mv: <1 min, sed loop: <1 min, pytest verification: 2-3 min).

**Pre-existing test risk:** None. No test uses `backend\.calendar` or `backend/calendar` as a regex string to be matched (verified by grep). The sed substitution changes import lines only; test logic (all 9 test functions in `test_calendar_watcher.py`) operates on function calls and return values, not on module path strings.

**Post-rename: repo-root invocation also works** — `python -m pytest backend/tests/ -q` from repo root sets `sys.path[0]` to the repo root (not `backend/`), so `backend` is importable as a package and `from backend.econ_calendar...` resolves correctly. The rename does not break repo-root invocations.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8 fetched in full)
- [x] 10+ unique URLs total incl. snippet-only (18 total: 8 full + 10 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (2025 + 2026 sources found and evaluated)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (10 files inspected; registry.py, blackout.py, normalize.py confirmed no self-referential imports)
- [x] Contradictions / consensus noted (sed vs AST tools; consensus on rename-first)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 10,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "report_md": "handoff/current/phase-16.34-research-brief.md",
  "gate_passed": true
}
```
