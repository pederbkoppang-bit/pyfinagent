---
step: "16.16"
title: "Backend correctness re-verification (pytest+AST+migration+health)"
tier: simple
date: 2026-04-24
author: researcher-agent
---

## Research: phase-16.16 — Backend correctness re-verification (pytest+AST+migration+health)

Tier assumed: `simple` (caller-specified). Floor of >=5 sources read in full still applies.

---

### Search queries run (3-variant discipline)

| Variant | Query |
|---------|-------|
| Year-less canonical | `pytest sys.path module shadowing stdlib calendar Python backend verification` |
| Year-less canonical | `import ast parse audit Python codebase syntax check scale` |
| Year-less canonical | `FastAPI health endpoint conventions pre-production go-live` |
| 2025 | `FastAPI health endpoint conventions pre-production go-live 2025` |
| 2025 | `pytest 8.x Python 3.13 3.14 import mode importlib breaking changes 2025` |
| 2026 | `FastAPI 0.115 pytest 8 Python 3.14 re-verification production backend 2026` |
| 2026 | `FastAPI Best Practices for Production: Complete 2026 Guide` |
| Year-less canonical | `BigQuery migration verify CLI pattern Python idempotent` |

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://docs.pytest.org/en/stable/explanation/pythonpath.html | 2026-04-24 | official doc | WebFetch | "The directory path containing each module will be inserted into the beginning of sys.path if not already there" — describes prepend vs append vs importlib modes and stdlib-shadowing risk |
| https://docs.pytest.org/en/stable/explanation/goodpractices.html | 2026-04-24 | official doc | WebFetch | "don't replicate standard library module names" — src-layout recommended; importlib mode avoids sys.path mutation entirely |
| https://docs.pytest.org/en/stable/changelog.html | 2026-04-24 | official doc | WebFetch | pytest 8.4.0 added Python 3.14 support; 8.4.2 fixed annotation evaluation crash in Py3.14 `TYPE_CHECKING` blocks; pythonpath config now initialises earlier (8.4.0) |
| https://www.index.dev/blog/how-to-implement-health-check-in-python | 2026-04-24 | authoritative blog | WebFetch | Health response must include: status, timestamp, version, per-component checks; HTTP 200 on pass, 503 on critical failure |
| https://medium.com/@bhagyarana80/fastapi-health-checks-and-timeouts-avoiding-zombie-containers-in-production-411a27c2a019 | 2026-04-24 | blog | WebFetch | `/health` must be a lightweight route that does NOT touch external services; readiness checks are separate; timeout-keep-alive and per-route `asyncio.wait_for()` prevent hangs |
| https://fastlaunchapi.dev/blog/fastapi-best-practices-production-2026 | 2026-04-24 | blog | WebFetch | pytest>=8.0.0 + pytest-asyncio>=0.23.0 + httpx>=0.27.0 as 2026 standard test stack; health endpoint checks db + cache; lifespan startup initialises dependencies before traffic |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://pytest-with-eric.com/introduction/pytest-pythonpath/ | blog | covered by official docs fetched in full |
| https://github.com/pytest-dev/pytest/issues/11960 | GH issue | snippet sufficient — confirms sys.path frustration is well-known; no new info |
| https://jdhao.github.io/2025/05/13/pytest_sys_path_issues/ | blog | 2025 practitioner post; snippet confirms prepend-mode shadowing on real projects |
| https://earthly.dev/blog/python-ast/ | blog | fetched; article is introductory only, no production-scale guidance |
| https://docs.cloud.google.com/bigquery/docs/reference/migration | official doc | BQ migration API; not relevant to --verify CLI pattern in this project |
| https://render.com/articles/fastapi-production-deployment-best-practices | blog | snippet adequate; no new health-endpoint specifics |
| https://github.com/zhanymkanov/fastapi-best-practices | repo | fetched; no health/lifespan content in this document |
| https://pythoneo.com/testing-fastapi-applications/ | blog | snippet covers same ground as sources already fetched in full |
| https://testdriven.io/blog/fastapi-crud/ | blog | snippet; CRUD focus, not health/verification |
| https://github.com/pytest-dev/pytest/issues/12044 | GH issue | importlib + assertion rewriting bug; known; fixed in 8.1.1 |

---

### Recency scan (2024-2026)

Searched explicitly for 2025 and 2026 content on: FastAPI health patterns, pytest 8.x import mode changes, Python 3.13/3.14 compatibility.

**Findings:**

- **pytest 8.4.0 (2025)**: `pythonpath` ini option now initialises earlier — affects plugins loaded via `-p`. Official Python 3.14 support added. Fixed: `--import-mode=importlib` crash when top-level directory has the same name as a stdlib module (directly relevant to `backend/calendar/`). This fix was backported to the stable branch.
- **pytest 8.4.2 (2025)**: Fixed annotation evaluation crash in Python 3.14 for modules using `TYPE_CHECKING` blocks without explicit `from __future__ import annotations`. This codebase may be affected if any backend module uses `TYPE_CHECKING` on Py3.14 without the future import.
- **FastAPI 0.115.x (2025)**: Added Python 3.14 TYPE_CHECKING (PEP 649) fix. Pydantic v1 no longer supported on Python 3.14 — requires Pydantic v2.
- **FastAPI production guide 2026**: pytest>=8.0.0 is the stated minimum for async FastAPI testing.
- **No superseding new literature** was found that changes the verification approach. Canonical sources remain current.

---

### Key findings

1. **pytest default import mode is `prepend`** — inserts root dir at the front of sys.path. This is the operative mode for this project (no pytest.ini or pyproject.toml found at repo root, so default applies). (Source: pytest docs, https://docs.pytest.org/en/stable/explanation/pythonpath.html)

2. **`backend/calendar/` stdlib shadow is a live risk under prepend mode.** When pytest prepends the repo root to sys.path, `import calendar` in any test or production module can resolve to `backend/calendar/__init__.py` instead of the Python stdlib `calendar`. The official mitigation is either `--import-mode=importlib` or `--import-mode=append` (prefers installed/stdlib over local). This project does NOT currently set either in config. (Source: pytest goodpractices doc + internal audit of `backend/calendar/__init__.py`)

3. **`--import-mode=importlib` is the right long-term fix but has known issues.** pytest 8.1.1 fixed a regression where importlib mode broke assertion rewriting. pytest 8.3.4–8.4.x fixed a crash specifically triggered when a top-level directory shares a name with a stdlib module — precisely the `backend/calendar/` scenario. (Source: pytest changelog, https://docs.pytest.org/en/stable/changelog.html)

4. **`ast.parse()` bulk scan is reliable but encoding-sensitive.** The verification command reads files with `open(f).read()` using the default system encoding. On macOS (UTF-8 default) this is fine, but any file with non-UTF-8 bytes (e.g. latin-1 comments) will raise `UnicodeDecodeError` before `ast.parse()` is even called, masking a potential `SyntaxError`. The current command does not specify `encoding='utf-8'`. Practical risk: low on this codebase, but worth noting. (Source: Python ast docs + earthly.dev blog)

5. **`/api/health` at `backend/main.py:331-366` returns `{"status": "ok", "service": "pyfinagent-backend", "version": "...", "mcp_servers": {...}, "limits_digest": "..."}` with HTTP 200.** This satisfies the `backend_health_200` criterion. The route is listed in `_PUBLIC_PATHS` at line 217, so it bypasses auth middleware — `curl -sS http://127.0.0.1:8000/api/health` will succeed without a Bearer token. (Source: internal audit of `backend/main.py`)

6. **Health endpoint is lightweight — no external I/O.** It uses `importlib.util.find_spec()` (in-process, no network) for MCP module checks and reads `CHANGELOG.md` from disk. No database call is made. This matches the recommendation to keep `/health` free of external dependencies. (Source: internal audit + Medium/index.dev best-practices sources)

7. **`scripts/migrations/create_strategy_deployments_view.py` exists and has a `--verify` flag.** The docstring at lines 1-29 explicitly documents `--verify` as a supported CLI mode (alongside `--apply`, `--dry-run`). Script was confirmed present on disk as of 2026-04-24. (Source: internal file read)

8. **`backend/tests/` has 20 test files + 1 api subdirectory.** Top-level files total approximately 153 test functions (sum of grep counts across all `.py` files). The `api/` subdirectory contains `test_sovereign.py`. No pytest config file detected — means `conftest.py` discovery and `prepend` import mode are the operative settings. (Source: internal audit)

9. **masterplan.json 16.16 entry matches the caller's verification command and success criteria exactly.** The stored `verification.command` and `verification.success_criteria` are character-for-character identical to what was provided in the research prompt. No discrepancy. (Source: internal read of `.claude/masterplan.json`)

10. **pytest 8.4.0 introduced earlier `pythonpath` initialisation** that may affect the order of path mutations when using the `-p` plugin loader. Low direct risk for this project but confirms the value of pinning pytest version in requirements. (Source: pytest changelog WebFetch, 2025 recency scan)

---

### Internal code inventory

| File | Lines read | Role | Status / Notes |
|------|-----------|------|----------------|
| `backend/main.py` | 1-399 (full) | FastAPI app entry, lifespan, /api/health | `/api/health` at line 331; public path at line 217; health returns status+version+mcp_servers+limits_digest |
| `backend/calendar/__init__.py` | 1-35 | Calendar watcher package entry | Shadow surface unchanged — imports from `backend.calendar.*` submodules; no stdlib `calendar` import visible in `__init__.py` itself, but any module that does bare `import calendar` risks hitting this package under prepend mode |
| `scripts/migrations/create_strategy_deployments_view.py` | 1-30 | BQ migration + view creation | Exists; `--verify` flag documented in docstring at lines 25-28 |
| `.claude/masterplan.json` | Full parse via Python | Step tracker | 16.16 entry: `status: pending`, `harness_required: true`; verification command and success_criteria confirmed identical to caller spec |
| `backend/tests/` (directory listing) | N/A | Test root | 20 .py files + api/ subdirectory; ~153 top-level `def test_` functions; no `pytest.ini` / `pyproject.toml` found |

---

### Consensus vs debate (external)

**Consensus:**
- `/api/health` should return HTTP 200 + structured JSON with at minimum `status`, `version`, and a timestamp. This codebase satisfies all three (status + version; no explicit timestamp but acceptable for a simple liveness check).
- `ast.parse()` glob-scan is the standard low-dependency syntax verification pattern for CI pipelines.
- pytest prepend mode is the default and carries stdlib-shadowing risk when local packages share stdlib names.

**Debate:**
- Whether `--import-mode=importlib` should replace prepend. The importlib mode fixes the calendar-shadow risk but had assertion-rewriting regressions in pytest 8.1.x (fixed in 8.1.1). For a re-verification step that runs the existing test suite unchanged, the risk of switching modes mid-step outweighs the benefit. The shadow risk is real but pre-existing and not newly introduced by this step.

---

### Pitfalls (from literature)

1. **Calendar shadow crash under prepend mode**: if any test imports stdlib `calendar` directly (e.g. for date arithmetic) it may import `backend.calendar` instead, causing an `AttributeError` on stdlib-only methods like `calendar.monthrange`. Review any test that does `import calendar` without a fully-qualified `from` import.
2. **`ast.parse()` encoding error masking**: `open(f).read()` without explicit encoding can raise `UnicodeDecodeError` before `ast.parse()` sees the file. If the command errors on encoding rather than syntax, the output will not contain `syntax_ok` and the criterion fails — but the failure message will look like an encoding error, not a syntax error.
3. **Health endpoint reads CHANGELOG.md from disk**: if `CHANGELOG.md` is absent or unreadable, the version falls back to `"6.0.0"` (silent, non-fatal). Not a blocker for `backend_health_200` but worth knowing.
4. **pytest 8.4.2 TYPE_CHECKING crash on Python 3.14**: if running Python 3.14 without `from __future__ import annotations` in modules that use `TYPE_CHECKING`, test collection can crash. This project is on Python 3.14 per `CLAUDE.md`. Check if this patch is in the installed pytest version.

---

### Application to pyfinagent (mapping findings to file:line anchors)

| Finding | File:line | Action for phase-16.16 |
|---------|-----------|------------------------|
| `/api/health` route registration | `backend/main.py:331` | Confirmed present; curl will return 200 |
| Health is a public path (no auth) | `backend/main.py:217` | Confirmed; curl without Bearer succeeds |
| Health payload fields | `backend/main.py:360-366` | Returns status+service+version+mcp_servers+limits_digest |
| Calendar stdlib shadow risk | `backend/calendar/__init__.py:1-27` | Pre-existing; no new action needed for this re-verification step |
| Migration --verify flag | `scripts/migrations/create_strategy_deployments_view.py:25-28` | Script exists and supports --verify; criterion bq_migration_verify_pass is reachable |
| masterplan 16.16 criteria match | `.claude/masterplan.json` (16.16 entry) | Exact match confirmed; no discrepancy |
| No pytest.ini / importmode set | repo root (no config file) | Prepend mode is operative; risk is pre-existing; no change needed for this step |

---

### Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources read in full)
- [x] 10+ unique URLs total (incl. snippet-only) (16 URLs collected)
- [x] Recency scan (last 2 years) performed + reported (see section above — pytest 8.4.x, FastAPI 0.115.x findings documented)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (main.py, calendar/__init__.py, migration script, masterplan, tests directory)
- [x] Contradictions / consensus noted (importlib vs prepend debate documented)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-16.16-research-brief.md",
  "gate_passed": true
}
```
