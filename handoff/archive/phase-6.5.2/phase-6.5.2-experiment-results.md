# Experiment Results — phase-6.5 / step 6.5.2 (Source registry + scanner core)

**Step:** 6.5.2 — second executable step under Path D.
**Date:** 2026-04-19.
**Cycle:** 1.

## What was built

Six new files; zero existing code changed.

1. `backend/intel/__init__.py` — package marker.
2. `backend/intel/source_registry.py` (~160 lines) — `SourceRow` dataclass + `load_from_yaml` (pure) + `upsert_sources` (fail-open BQ insert) + `load_active_sources` (fail-open BQ query with `kill_switch = FALSE` filter). Mirrors `backend/news/bq_writer.py:41-97` for `_resolve_target`, `_get_client`, and the never-raise write pattern.
3. `backend/intel/scanner.py` (~160 lines) — `DocumentCandidate` TypedDict matching `intel_documents` columns + `BaseScanner` with `scan(dry_run=False)`, `_do_scan`, `_fetch_http` (with EDGAR-style 60·2^attempt 403 backoff + 5·2^attempt 5xx backoff), `_stub_candidates` for dry-run, and intra-batch dedup via `(canonical_url, content_hash)` set.
4. `backend/tests/fixtures/intel_sources.yaml` — 3 sources: http active, rss active, http kill-switched.
5. `backend/tests/test_intel_source_registry.py` — 9 tests.
6. `backend/tests/test_intel_scanner.py` — 10 tests.

## File list

Created: 6 files.
Modified: 0 files.

## Verification command output

### Immutable (masterplan 6.5.2)

```
$ source .venv/bin/activate && pytest backend/tests/test_intel_source_registry.py backend/tests/test_intel_scanner.py -q
...................                                                      [100%]
19 passed in 3.69s
EXIT=0
```

19/19 passed. Both test modules green.

### Full regression (no_regressions implicit check)

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
130 passed, 1 skipped, 1 warning in 8.17s
```

Baseline before: 111 passed / 1 skipped (phase-6.5.1 close). Delta = +19 (the 9 registry + 10 scanner tests). Zero regressions on the previously-green surface.

### Syntax check

```
$ python -c "import ast; [ast.parse(open(p).read()) for p in ('backend/intel/source_registry.py','backend/intel/scanner.py','backend/tests/test_intel_source_registry.py','backend/tests/test_intel_scanner.py')]; print('SYNTAX OK')"
SYNTAX OK
```

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `registry_loads_all_configured_sources` | PASS | `test_load_from_yaml_returns_all_sources` asserts all 3 fixture rows parse; `test_load_from_yaml_preserves_metadata` verifies the RSS `feed_url` round-trips. |
| 2 | `scanner_dry_run_returns_candidates` | PASS | `test_dry_run_returns_stub_candidate` asserts `BaseScanner(src).scan(dry_run=True)` returns `[DocumentCandidate]` with every required key populated. |
| 3 | `tests_green` | PASS | 19/19 passed; exit 0. |

## Known caveats (transparency)

1. **No live HTTP or BQ exercised this cycle.** `_fetch_http` is implemented with `requests.get(...)` + the EDGAR-style 403 / 5xx backoff ladder, but it's only reached through `_do_scan` which is monkeypatched in the dedup test. A future integration test or phase-6.5.9 smoketest should exercise it with a stubbed HTTP server (e.g. `responses` or `httpretty`) before any live feed is wired.
2. **`upsert_sources` uses `insert_rows_json` not true UPSERT.** BQ streaming has no UPSERT; the contract warned callers about not double-loading. A future phase that needs idempotent re-seeding should switch to a MERGE statement or ingest-dedup by source_id.
3. **`load_active_sources` returns empty on any BQ error** (fail-open). In production with a working BQ this is fine; in a crippled environment, callers will see "no active sources" rather than an exception — that's the documented house pattern but worth flagging for ops.
4. **EDGAR backoff encoded but unused.** The 60·2^attempt backoff on 403 is coded per research-brief R1; it's not exercised this cycle because no EDGAR URL is fetched. Worth a unit test via `responses` in a later step.
5. **ASCII-only discipline enforced by tests.** Both `test_scanner_module_is_ascii_only` and `test_registry_module_is_ascii_only` decode the module files as ASCII (raises `UnicodeEncodeError` if any non-ASCII character slips in).

## Pre-Q/A self-check

- Immutable pytest command exit 0 (19/19).
- Full regression 130 passed / 1 skipped (baseline 111; +19 new; zero existing tests broken).
- All six new files have correct Python AST.
- `git status --short` shows only new files under `backend/intel/`, `backend/tests/` (two new test files + fixtures dir), and handoff artifacts. No production backend module outside the new package was modified.
- Handoff files phase-scoped: `phase-6.5.2-{contract,experiment-results,research-brief}.md`.
- Masterplan NOT flipped yet; log-last discipline preserved.
