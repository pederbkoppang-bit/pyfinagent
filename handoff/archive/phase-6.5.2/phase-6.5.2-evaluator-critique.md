# Q/A Evaluator Critique -- phase-6.5 / 6.5.2 Source registry + scanner core

**Verdict id:** `qa_652_v1`
**Date:** 2026-04-19
**Cycle:** 1 (Path D)
**Agent:** Q/A (single-agent, merged qa-evaluator + harness-verifier)

---

## 5-item protocol audit

| # | Audit item | Verdict | Evidence |
|---|---|---|---|
| 1 | Researcher spawn proof | PASS | `phase-6.5.2-research-brief.md` (21:07) present. JSON envelope: `external_sources_read_in_full=6`, `snippet_only_sources=9`, `urls_collected=15`, `recency_scan_performed=true`, `internal_files_inspected=8`, `gate_passed=true`. Three-variant queries visible (current-year 2026, last-2-year 2025, year-less canonical). Recency scan section present and populated. |
| 2 | Contract PRE-commit | PASS | mtimes: contract=1776625727 (21:08), experiment-results=1776625868 (21:11). Contract precedes results by 141s. Research brief (21:07) precedes contract. Order: research -> contract -> generate. |
| 3 | Experiment results present | PASS | `phase-6.5.2-experiment-results.md` contains verbatim immutable output (19 passed, EXIT=0), verbatim regression (130 passed, 1 skipped), file list (6 created, 0 modified), and 3-row criterion table. |
| 4 | Log-last discipline | PASS | Last `harness_log.md` cycle block is `phase=6.5.1 result=PASS` at 21:00 UTC. No 6.5.2 block yet. Main correctly deferred the append until after Q/A. |
| 5 | No verdict-shopping | PASS | First Q/A spawn on 6.5.2. No prior critique file exists. |

All 5 audit items PASS.

---

## Deterministic checks (A-I)

### A. Immutable command

```
$ source .venv/bin/activate && pytest backend/tests/test_intel_source_registry.py backend/tests/test_intel_scanner.py -q
...................                                                      [100%]
19 passed in 3.09s
EXIT=0
```

PASS -- 19/19 as claimed, exit 0.

### B. Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
...130 passed, 1 skipped, 1 warning in 7.46s
```

PASS -- matches claim (baseline 111 + 19 new = 130). Zero existing tests broken.

### C. File existence

All six files present with expected sizes:
- `backend/intel/__init__.py` (101B)
- `backend/intel/source_registry.py` (6951B, ~205 lines)
- `backend/intel/scanner.py` (6451B, ~196 lines)
- `backend/tests/fixtures/intel_sources.yaml` (565B)
- `backend/tests/test_intel_source_registry.py` (2690B)
- `backend/tests/test_intel_scanner.py` (3565B, 10 tests)

PASS.

### D. Scope

`git status --short` for step-relevant paths:
```
?? backend/intel/
?? backend/tests/fixtures/intel_sources.yaml
?? backend/tests/test_intel_scanner.py
?? backend/tests/test_intel_source_registry.py
```

PASS -- only NEW (untracked) files under `backend/intel/` and `backend/tests/`. No modifications to existing production modules. The repo-wide `git status` contains many unrelated `M` entries preexisting from earlier phases; none are attributable to 6.5.2.

### E. Criterion alignment

Targeted re-run:
```
backend/tests/test_intel_source_registry.py::test_load_from_yaml_returns_all_sources PASSED
backend/tests/test_intel_scanner.py::test_dry_run_returns_stub_candidate PASSED
backend/tests/test_intel_scanner.py::test_intra_batch_dedup_removes_duplicate_canonical_url PASSED
backend/tests/test_intel_source_registry.py::test_upsert_fail_open_no_bq_auth PASSED
```

- `registry_loads_all_configured_sources`: PASS -- fixture has 3 rows (2 active + 1 kill-switch); load test passes.
- `scanner_dry_run_returns_candidates`: PASS -- test asserts `len(cands)==1`, all required keys, `doc_type=="stub"`, non-empty `content_hash`, canonical_url prefix.
- `tests_green`: PASS -- 19/19 exit 0.

### F. Fail-open discipline

Read `backend/intel/source_registry.py:65-76`: `_get_client` returns `None` on ImportError (line 69-71) and on `bigquery.Client()` failure (line 74-76). `upsert_sources` lines 148-150 guard with `if client is None: return 0`; entire BQ block wrapped in try/except returning 0 on any exception (line 162-164). `load_active_sources` lines 177-179 do same; try/except returns `[]` (line 202-204). `test_upsert_fail_open_no_bq_auth` re-run confirms no raise.

PASS.

### G. Lazy BQ import

AST walk confirms `from google.cloud import bigquery` lives ONLY inside `_get_client` function body at line 68. No top-level bigquery import. Matches `backend/news/bq_writer.py` house pattern.

PASS.

### H. Dedup correctness

`backend/intel/scanner.py:110-119` -- `_dedup` keys on `(c.get("canonical_url",""), c.get("content_hash",""))` tuple in a `seen: set[tuple[str,str]]`. `test_intra_batch_dedup_removes_duplicate_canonical_url` (lines 88-96) feeds 3 identical dicts, asserts `len(result)==1`. Re-run PASSED.

PASS.

### I. ASCII discipline

Re-run (corrected: both tests live in `test_intel_scanner.py`):
```
test_intel_scanner.py::test_scanner_module_is_ascii_only PASSED
test_intel_scanner.py::test_registry_module_is_ascii_only PASSED
```

Fixture YAML decode-ascii check also passed. Module docstrings, logger calls, and test assertions are all ASCII.

PASS.

Minor note (non-blocking): the prompt specified `test_registry_module_is_ascii_only` would live in `test_intel_source_registry.py`; in practice Main co-located both ASCII-probes in `test_intel_scanner.py`. Both exist and pass; the naming is a style preference, not a defect.

---

## LLM judgment

**Schema alignment.** `DocumentCandidate` TypedDict (scanner.py:35-49) is a strict subset of the `intel_documents` DDL (migration lines 60-74): `doc_id, source_id, source_type, doc_type, published_at, ingested_at, title, authors, url, canonical_url, content_hash, raw_text, language, raw_payload`. All 14 DDL columns have a matching TypedDict key. No missing or mismatched field. `test_document_candidate_fields_align_with_schema` enforces the required-keys subset. PASS.

**EDGAR risk R1 (403 retry-loop) encoded.** `_fetch_http` line 146-147: `if resp.status_code == 403 and attempt < max_attempts: time.sleep(60 * (2**attempt))`. Matches tldrfiling.com's `60 * (2**attempt)` formula exactly. 5xx path (line 149-150) correctly uses the shorter `5 * (2**attempt)`. The 403 and 5xx paths are distinct -- no accidental collapse. PASS.

**Kill-switch invariant.** `load_active_sources` SQL (line 182-186) constructs `... WHERE kill_switch = FALSE`. The fixture includes `stub_disabled` with `kill_switch: true`; `test_kill_switch_preserved_in_yaml_load`-style tests confirm the YAML load preserves the flag. The active-only filter is enforced in the registry, not delegated to callers. PASS.

**Anti-rubber-stamp (caveats honesty).** Spot-checked all 5 caveats in `experiment-results.md`:
1. "No live HTTP or BQ exercised" -- TRUE; `_fetch_http` is only reached via monkeypatched `_do_scan` in `test_intra_batch_dedup`. Accurate.
2. "`upsert_sources` uses `insert_rows_json` not true UPSERT" -- TRUE; line 157 calls `client.insert_rows_json(table_ref, payload)`. No MERGE. Accurate.
3. "`load_active_sources` returns empty on BQ error" -- TRUE; line 203-204 try/except returns `[]`. Accurate.
4. "EDGAR backoff encoded but unused" -- TRUE; no live EDGAR URL is in the fixture. Accurate.
5. "ASCII-only discipline enforced by tests" -- TRUE; both module-probe tests PASS on clean ASCII decode.

Main did NOT overclaim. The caveats section is an honest gap-report, not puffery. PASS.

**Scope honesty.** The step-scoped `git status` is backend-intel-only: the `intel/` package (new), two test files (new), one fixture (new). No existing production file was touched. The repo-wide `git status` contains many preexisting `M` entries from prior phases; Main correctly did not claim authorship of those.

**Weakest-link for phase-6.5.9 smoketest (non-blocking, flagged for next contract).** If the 6.5.9 smoketest invokes `BaseScanner(src).scan(dry_run=False)` against the current fixture sources, `_do_scan` will attempt an HTTP GET to `https://stub.example.com/feed` (and `https://feeds.example.com/finance.rss`). Neither host resolves to a real feed. The `requests.get` call will either (a) raise a connection error -- caught at scanner.py:143-145, returns `[]` -- or (b) return a non-200 -- returns `[]` at 152-154. Both paths are fail-open and correct, but the smoketest will observe zero candidates from a live run unless 6.5.9 either (i) stands up a local HTTP fixture (`responses`/`httpretty`/`pytest-httpserver`), or (ii) accepts empty-pipe as the smoketest's expected outcome, or (iii) adds at least one real feed URL to the fixture before 6.5.9. This is the EXACT open-issue that was logged against phase-6.5 at 20:52 UTC (prompt-patch queue has no extractor feeding it post-drops); the 6.5.9 contract will need to address it explicitly.

---

## Output envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met (registry_loads_all_configured_sources, scanner_dry_run_returns_candidates, tests_green). 5-item protocol audit all PASS. Deterministic A-I all PASS on independent re-run (19/19 immutable, 130 passed/1 skipped regression, +19 delta). Fail-open discipline verified line-by-line; EDGAR 60*2^attempt 403 backoff correctly encoded; kill_switch filter enforced in registry SQL; DocumentCandidate TypedDict is a strict subset of intel_documents DDL. Caveats section is honest; no overclaim.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "5_item_protocol_audit",
    "syntax",
    "verification_command",
    "regression_full_suite",
    "file_existence",
    "scope_git_status",
    "criterion_targeted_rerun",
    "fail_open_discipline_line_read",
    "lazy_bq_import_ast",
    "dedup_correctness_test_rerun",
    "ascii_discipline_module_and_fixture",
    "schema_alignment_documentcandidate_vs_ddl",
    "edgar_r1_403_backoff_formula",
    "kill_switch_sql_filter",
    "anti_rubber_stamp_caveat_spotcheck",
    "scope_honesty",
    "weakest_link_next_step_65_9"
  ]
}
```

## Final Decision: **PASS** -- `qa_652_v1`

No blockers. Main may now:
1. Append the cycle block to `handoff/harness_log.md` (log-last).
2. Flip `.claude/masterplan.json` `phase-6.5.2.status`: `pending` -> `done`.

Flag carried forward to 6.5.9 contract: stub fixture URLs are unreachable; smoketest must either stand up a local HTTP fixture OR explicitly accept empty-pipe AS the asserted smoketest outcome. This is the same open-issue logged at 20:52 UTC against phase-6.5 decision, not a new blocker.
