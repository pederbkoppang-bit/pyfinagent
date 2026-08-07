# Contract — Step 83.0: news corpus persistence (tables + observable writer + source-agnostic)

- **Step id:** 83.0 (P0, phase-83; depends_on: none)
- **Tier (named field, per drain-goal rule):** T3 — executor Main (Opus 5, effort max); Q/A via qa-verdict Workflow (opus/max). No T4 needed: the hard analysis was done by the research gate.
- **Date:** 2026-08-07, full-day autonomous drain, cycle 168

## Research-gate summary

`handoff/current/research_brief_83.0.md` — gate_passed: **true** (8 external sources read in full / 35 URLs / recency scan performed / 16 internal files; envelope verbatim in the brief). The Workflow return was dropped (StructuredOutput never called) but the brief is complete on disk under write-first; the envelope in the brief is authoritative. Decisive findings:

1. **REQUIRED columns can only be created with the table** (Google, Modifying table schemas). Tables verified ABSENT live 2026-08-07 → this step is the ONLY window for `provenance NOT NULL`.
2. **`CREATE TABLE IF NOT EXISTS` is a no-op on an existing table** → the migration must read back `get_table().schema` and FAIL LOUD if the post-condition doesn't hold.
3. **`insert_rows_json` is strict by default** (`ignore_unknown_values=False`, `skip_invalid_rows=False`) — REFUTES the `bq_writer.py:23-25` docstring; a schema mismatch fails the whole batch and is currently swallowed into `return 0`.
4. **Rename blast radius measured: 16 `fetched_at` sites in 8 files — 9 change, 7 must NOT** (the `calendar_events` path; that table EXISTS live and a blind rename breaks it silently).

## Hypothesis

Creating the two tables with the amended one-shot schema, making the three real failure branches of `_insert_rows` increment a keyed counter + emit a WARNING, and threading `provenance` through `_normalize`/`run_once`/`_serialize_article` converts the silently-no-op corpus path into an auditable one, without touching the live news_screen path, the calendar_events writer, or the finnhub/benzinga adapters.

## Immutable success criteria (verbatim from `.claude/masterplan.json` 83.0)

1. "the migration is run and a test asserts the resulting news_articles schema contains BOTH a `published_at` TIMESTAMP column and a distinct `ingested_at` TIMESTAMP column, failing if either is absent or if the two resolve to the same column name"
2. "the schema carries a REQUIRED `provenance` STRING column, and a test asserts a row produced by the live fetch path records provenance='live' while a row produced by a backfill path records provenance='backfill', failing if either path leaves it null or if both paths emit the same value"
3. "a test injects an insert failure into the fail-open path and asserts the failure BOTH increments an observable counter AND emits a log record, while the writer still returns normally rather than raising -- both halves asserted in the same test"
4. "the observability guard is mutation-tested: deleting the counter increment from the failure branch makes the test FAIL. The test must assert the counter's numeric value strictly increases across the injected failure, not merely that the attribute exists -- a test that passes when the counter is never incremented does not count"
5. "the corpus writer is source-agnostic: a test asserts it accepts rows from at least two distinct registered source adapters, and asserts that neither the migration module nor bq_writer imports any module requiring ALPHAVANTAGE_API_KEY to be present, so the operator's licence decision can be made without reworking the schema"
6. "the measured absence of FINNHUB and BENZINGA keys is recorded in the step artifact with verbatim check output, and backend/news/sources/finnhub.py and backend/news/sources/benzinga.py are byte-unchanged, asserted by a committed diff over exactly those two paths"

**Verification command (immutable):** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_0_news_corpus_persistence.py -q`

**live_check (immutable):** "verbatim BigQuery output of `bq show --schema` for pyfinagent_data.news_articles and pyfinagent_data.news_sentiment showing published_at, ingested_at and provenance present with their modes, captured BEFORE the migration (showing both tables absent) and AFTER (showing both present); plus the verbatim counter value and log line produced by a deliberately failed write; plus the verbatim terminal output of the key-presence check for FINNHUB and BENZINGA" → artifact: `handoff/current/live_check_83.0.md` (BEFORE half already captured 2026-08-07 pre-GENERATE).

## Explicit decisions (stated, not implied)

- **D1 — `news_sentiment` also gets `ingested_at TIMESTAMP NOT NULL` + `provenance STRING NOT NULL`.** The live_check names both tables' schemas with those columns, and the one-shot REQUIRED window (research finding 1) closes today. For sentiment rows both values are truthful at scoring time (`ingested_at` = the actual write moment of the score row; scoring during a backfill run carries provenance='backfill'). `published_at` is NOT added to news_sentiment — it belongs to the article; the join is article_id.
- **D2 — the refuted `bq_writer.py:23-25` docstring is corrected** in the same commit (observability = not lying about failure semantics).
- **D3 — the 7 calendar_events `fetched_at` sites are frozen**: post-edit assertion that `git diff` over `backend/econ_calendar/`, `scripts/migrations/add_calendar_events_schema.py`, and the calendar sections of `bq_writer.py`/tests is empty of `fetched_at` changes.
- **D4 — two discovered defects are QUEUED as their own research-gated steps, not fixed here** (standing rule `feedback_queue_discovered_defects_in_masterplan`): (i) live news path (news_screen.py) is disjoint from the corpus — headlines that move the book are still not captured; (ii) `api_call_log` table absent while `api_call_log.py:148` writes to it fail-open. Added to `.claude/masterplan.json` as pending steps in the masterplan edit that closes 83.0.
- **D5 — Alpha Vantage licence decision stays OPEN** (operator ask #6); nothing in this step imports or depends on it.

## Correction (cycle 3, 2026-08-07) — D1's sentiment claim was false when written

D1's sentence "scoring during a backfill run carries provenance='backfill'" was FALSE as delivered in cycles 1-2: `ScorerResult` has no provenance field, `score_ladder` takes no such kwarg, and `_serialize_sentiment` defaulted every reachable path to `'live'` (proven by the cycle-2 Q/A's behavioural probe — `news_sentiment.provenance` was a constant). Cycle 3 makes the claim true at the write seam: `write_news_sentiment(..., provenance=)` stamps rows lacking their own value, and `phase6_e2e` threads `"backfill"|"live"` into it. Precedence: explicit row value > kwarg > "live". The original D1 text above is left unedited as the record of what was claimed.

## Plan

1. Amend `scripts/migrations/add_news_sentiment_schema.py`: `ingested_at` (rename), `provenance STRING NOT NULL` in both DDLs, `ingested_at NOT NULL` in news_sentiment; add `_verify()` post-condition (schema read-back, non-empty assert, mode check, loud SystemExit with the drop/recreate warning); keep `--dry-run` skipping both DDL and verify.
2. `backend/news/bq_writer.py`: module counter `_WRITE_FAILURES` (dict per table) + `threading.Lock` + `write_failure_count()` + `reset_write_failures_for_test()` + `_record_failure(table, reason, detail)` called at the three real failure branches (`client_absent`, `insert_errors`, `exception`) and NOT at empty-input; `_serialize_article` emits `ingested_at` + `provenance` (defensive default "live", `_VALID_PROVENANCE` guard); `_serialize_sentiment` emits `ingested_at` + `provenance`; docstring fix (D2). ASCII-only log strings.
3. `backend/news/fetcher.py`: `NormalizedArticle` gains `ingested_at`/`provenance`; `_normalize(raw, source_name, provenance="live")`; `run_once(..., provenance="live")` threads it; smoke assertion updated. (`published_at` fabrication is NOT touched here — that is 83.0.1's scope.)
4. Update the two existing red-going tests in `backend/tests/test_bq_writer.py` (field-set equality) in the same commit.
5. New `backend/tests/test_phase_83_0_news_corpus_persistence.py` per the brief's fixture strategy: fake at `_get_client` seam; criteria 1+2 schema tests against live schema with checked-in snapshot fallback (non-empty oracle asserted first); criterion 3 both halves in ONE test + negative control; criterion 5 registry test with `clear_registry()` fixture + teardown re-import, structural sys.modules import-chain assertion; criterion 6 byte-unchanged diff assertion over exactly the two adapter paths.
6. Run the migration live; capture AFTER `bq show --schema` for both tables; deliberately fail a write (fake client with error payload via a one-shot script) and capture counter + log line verbatim into `live_check_83.0.md`.
7. Mutation matrix (each must flip a test RED, restored after): (m1) delete counter increment; (m2) delete `logger.warning`; (m3) `+1`→`+0`; (m4) move `_record_failure` into empty-input branch; (m5) make both provenance paths emit "live"; (m6) point criterion-1 test at a schema map forced empty (oracle-vacuity control). Record kill/survive per row; re-run the WHOLE matrix if tests change.
8. `experiment_results_83.0.md` → qa-verdict Workflow → transcribe verdict verbatim → harness_log append → masterplan flip (with D4's two new steps added in the same masterplan edit, both `pending`).

## References

- `handoff/current/research_brief_83.0.md` (8 read-in-full sources, URLs + access dates therein)
- Google Cloud: Modifying table schemas; DDL reference; tabledata.insertAll REST (via brief)
- Installed SDK `Client.insert_rows_json` source (via brief)
- `backend/services/observability/api_call_log.py` (house counter idiom)
- CLAUDE.md BQ rules 4-5 (migrations for change; no DROP without owner approval)
