# Sprint Contract — phase-6.5 / step 6.5.2 (Source registry + scanner core)

**Step id:** 6.5.2
**Phase:** phase-6.5 Global Intelligence Directive (Path D)
**Cycle:** 1
**Date:** 2026-04-19
**Tier:** moderate

Parallel-safe: all handoff files phase-scoped (`phase-6.5.2-*.md`).

## Research-gate summary

Researcher fetched 6 sources in full (Meltano/Singer spec, tldrfiling EDGAR rate-limit, dealcharts EDGAR best-practices, sec-edgar-api GitHub, Singer canonical SPEC, Airbyte YAML connector overview), 15 URLs collected, recency scan present (2024–2026), three-variant queries visible, 8 internal files inspected with file:line anchors. `gate_passed: true`. Brief at `handoff/current/phase-6.5.2-research-brief.md`.

## Hypothesis

The registry is a thin BQ-backed store seeded from a YAML fixture. The scanner is a `BaseScanner` class whose default `scan(dry_run=True)` returns 1 deterministic stub `DocumentCandidate` (satisfying `scanner_dry_run_returns_candidates`) and whose non-dry-run path fails open on network/BQ errors. Both modules follow `backend/news/` house patterns (fail-open BQ client, intra-batch dedup discipline documented, ASCII-only logger messages, lazy google-cloud-bigquery import). Tests run with zero real network and zero BQ auth.

## Immutable success criteria (copied verbatim from .claude/masterplan.json)

- `registry_loads_all_configured_sources`
- `scanner_dry_run_returns_candidates`
- `tests_green`

Interpretation:
- `registry_loads_all_configured_sources` ← `test_intel_source_registry.py::test_load_from_yaml_returns_all_sources` parses the fixture and asserts all 3 `SourceRow`s return (including the one with `kill_switch=true`). Plus `test_load_active_sources_filters_kill_switch` asserts the active-only query returns 2.
- `scanner_dry_run_returns_candidates` ← `test_intel_scanner.py::test_dry_run_returns_stub_candidate` asserts `BaseScanner(source).scan(dry_run=True)` returns a list with ≥1 `DocumentCandidate`, every required key present, ASCII-only, no network issued.
- `tests_green` ← `pytest backend/tests/test_intel_source_registry.py backend/tests/test_intel_scanner.py -q` exits 0.

## Plan steps

1. Create `backend/intel/__init__.py` (empty, package marker).
2. Create `backend/intel/source_registry.py`:
   - `SourceRow` dataclass
   - `load_from_yaml(path) -> list[SourceRow]` — pure, no BQ
   - `upsert_sources(rows, *, project=None, dataset=None) -> int` — fail-open via `insert_rows_json`
   - `load_active_sources(*, project=None, dataset=None) -> list[SourceRow]` — fail-open; filters `kill_switch = false`
3. Create `backend/intel/scanner.py`:
   - `DocumentCandidate` TypedDict matching `intel_documents` columns
   - `BaseScanner` class: `__init__(source)`, `scan(*, dry_run=False)`, `_do_scan`, `_fetch_http`, `_stub_candidates`
4. Create `backend/tests/fixtures/intel_sources.yaml` with 3 sources (2 active + 1 kill-switched) matching the research brief's YAML shape.
5. Create `backend/tests/test_intel_source_registry.py`:
   - `test_load_from_yaml_returns_all_sources` (3 rows, covering `registry_loads_all_configured_sources`)
   - `test_load_from_yaml_missing_file_returns_empty`
   - `test_source_row_dataclass_fields` (type-check all fields present)
   - `test_upsert_fail_open_no_bq_auth` (bad project → 0 rows, never raises)
   - `test_load_active_fail_open_no_bq_auth` (bad project → [] , never raises)
   - `test_kill_switch_preserved_in_yaml_load` (all 3 load; kill-switch field read correctly)
6. Create `backend/tests/test_intel_scanner.py`:
   - `test_dry_run_returns_stub_candidate` (covers `scanner_dry_run_returns_candidates`)
   - `test_scan_fail_open_on_network_error` (monkeypatch `_fetch_http` to raise; `scan()` returns [])
   - `test_document_candidate_fields_match_schema` (required keys align with `intel_documents` DDL)
   - `test_ascii_only_logger_message` (no non-ASCII in logger calls — grep the module file)
   - `test_scanner_skips_when_source_missing_url` (empty metadata → [])
7. Run immutable verification + full regression.
8. Write `phase-6.5.2-experiment-results.md`, spawn Q/A, log-last, flip.

## Out of scope

- No live HTTP or BQ call this cycle.
- No APScheduler cron wiring (that's phase-6.5.9 smoketest territory).
- No source-specific subclasses (all 4 dropped under Path D — the `BaseScanner` is the contract).
- No novelty scoring (phase-6.5.7).
- No YAML under `backend/config/` — fixture lives under `backend/tests/fixtures/` per research recommendation.

## Risk register (from researcher)

- **R1 EDGAR 403 loop** — the `_fetch_http` helper uses a 60 × 2^attempt backoff on 403 (even though EDGAR itself isn't exercised this cycle; discipline is encoded for when it is).
- **R2 Dedup race** — the scanner's dedup happens intra-batch (documented in module docstring, implemented via a `seen = set()` of `(canonical_url, content_hash)` pairs). Phase-6.5.7 will own any cross-batch dedup via embeddings.

## References

- `handoff/current/phase-6.5.2-research-brief.md`
- `handoff/current/phase-6.5.1-contract.md` (schema)
- `scripts/migrations/phase_6_5_intel_schema.py` (table shapes)
- `backend/news/bq_writer.py:61-72,75-97` (fail-open BQ client + never-raise insert)
- `backend/news/fetcher.py:132-175` (dry-run semantics)
- `backend/governance/limits_schema.py:68-83` (YAML + Pydantic pattern)
- `.claude/rules/security.md` (ASCII-only logger, EDGAR User-Agent)
- `.claude/masterplan.json` → phase-6.5 / 6.5.2 (immutable verification)
