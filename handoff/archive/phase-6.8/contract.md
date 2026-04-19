# Sprint Contract -- phase-6.8 End-to-End Smoketest + 24h Backfill

**Written:** 2026-04-19 PRE-commit.
**Step id:** `phase-6.8` (final step of phase-6 News & Sentiment Cron).
**Parallel-safety:** phase-specific filename.

## Research-gate summary

Researcher spawned today. Envelope: `{tier: moderate, external_sources_read_in_full: 7, snippet_only_sources: 10, urls_collected: 17, recency_scan_performed: true, internal_files_inspected: 15, gate_passed: true}`. Brief at `handoff/current/phase-6.8-research-brief.md` (189 lines; 4 recency findings).

Staked research rec: **create `backend/news/bq_writer.py` as single BQ writer module** for all 3 phase-6 tables (`news_articles`, `news_sentiment`, `calendar_events`) using `insert_rows_json` (matching the proven `api_call_log.py:141` pattern). Centralises BQ auth + dataset resolution + fail-open + the dedup/idempotency policy (at-least-once, downstream SELECT DISTINCT for event_id).

Other decisions grounded in the brief:
- Smoketest = **serial pipeline** (not parallel) -- Airbyte guide 2024-2026 practitioner consensus.
- `insert_rows_json` over Storage Write API for <200 rows -- BQ official docs.
- At-least-once semantics acceptable: intra-batch dedup runs before insert; event_id is sha256-deterministic and consumers SELECT DISTINCT.
- 24h backfill = **re-run of existing fetchers with rate-limit headroom**, NOT a historical API pull (Finnhub free-tier has no time-range filter).
- Wire benzinga + alpaca to phase-6.7 observability primitives using `finnhub.py` as the reference pattern.
- Add `bq_dataset_observability` as an explicit Settings field (remove duck-type fallback).

## Hypothesis

A single `backend/news/bq_writer.py` module + a serial `scripts/smoketest/phase6_e2e.py` runner can exercise the full news-sentiment-calendar pipeline end-to-end without infra auth failures (fail-open at BQ + Slack), produce a JSON pass/fail summary, and serve as the 24h-backfill driver via a `--backfill` flag that swaps in live sources.

## Success criteria

NOTE: masterplan step `verification: null`. Defined here per discipline.

**Functional:**
1. New module `backend/news/bq_writer.py` exporting:
   - `write_news_articles(articles: list[NormalizedArticle], *, project: str | None = None, dataset: str | None = None) -> int`
   - `write_news_sentiment(results: list[ScorerResult], *, project: str | None = None, dataset: str | None = None) -> int`
   - `write_calendar_events(events: list[CalendarEvent], *, project: str | None = None, dataset: str | None = None) -> int`
   All three: use `client.insert_rows_json`, fail-open on missing BQ / auth, return rows inserted (0 on any failure), log at WARNING on failure.
2. `backend/news/fetcher.py:115-124` `_write_batch_to_bq` stub replaced with `from backend.news.bq_writer import write_news_articles` + call. Import is guarded at function scope so tests of `fetcher.run_once(dry_run=True)` don't require BQ.
3. `backend/news/sources/benzinga.py` hardened with phase-6.7 observability (rate_limit + retry + log_api_call + raise_cron_alert), mirroring finnhub.py. Default rate-limit source key: `benzinga`.
4. `backend/news/sources/alpaca.py` same hardening.
5. `scripts/smoketest/phase6_e2e.py` implements the 10-step serial flow from the research brief:
   - Parses CLI: `--backfill` (default false), `--dry-run` (default true), `--sources finnhub,benzinga,...`.
   - Serial execution of fetch -> BQ write news_articles -> score_ladder -> BQ write news_sentiment -> calendar fetch -> BQ write calendar_events -> flush api_call_log + llm_call_log -> Slack heartbeat (optional).
   - Writes a JSON summary to stdout + an audit JSONL entry under `handoff/audit/phase6_smoketest.jsonl` with timestamps + per-stage row counts + errors list.
   - Exits 0 when the pipeline completes (even if BQ writes returned 0 due to auth absence -- the smoketest validates the code paths, not infra).
   - Exits 1 only on a real Python exception that escapes the fail-open boundary.
6. `backend/config/settings.py` gets a new explicit field: `bq_dataset_observability: str = "pyfinagent_data"`.
7. Smoketest has a dedicated `--help` surface documenting the two modes.
8. **Tests:** `backend/tests/test_bq_writer.py` with >=4 tests covering: (a) writers return 0 when google.cloud.bigquery is absent / auth fails; (b) `write_news_articles` serializes NormalizedArticle -> row dict correctly; (c) `write_news_sentiment` serializes ScorerResult -> row dict correctly; (d) `write_calendar_events` serializes CalendarEvent -> row dict correctly. Tests must NOT make real BQ calls (patch the client or assert via fail-open path only).
9. **Smoketest runnable in dry-run mode** without BQ credentials: `python scripts/smoketest/phase6_e2e.py --dry-run` completes and emits a JSON summary, exit 0.

**Correctness verification commands:**
- Syntax: `python -c "import ast; ast.parse(open('backend/news/bq_writer.py').read())"` (etc for each new / modified file).
- Import: `python -c "from backend.news.bq_writer import write_news_articles, write_news_sentiment, write_calendar_events; print('ok')"`.
- Pytest: `pytest backend/tests/test_bq_writer.py backend/tests/test_observability.py backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py -q` -- all pass (cumulative 30+ tests from prior steps + new ones, zero regressions).
- Smoketest dry-run: `python scripts/smoketest/phase6_e2e.py --dry-run` -- exit 0, prints JSON with all stages reported.

**Non-goals (explicit scope):**
- NOT running live 24h backfill against real APIs (no session API keys; smoke is CLI-plumbing only in --dry-run mode; --backfill flag exists but will require Peder's run in a follow-up).
- NOT implementing BQ `MERGE` for idempotency (at-least-once acceptable per research).
- NOT switching from `insert_rows_json` to Storage Write API (research-backed deferral).
- NOT wiring Gemini/OpenAI llm_call_log (already-disclosed phase-6.7 follow-up debt).
- NOT touching fed_scrape / fred_releases / alphavantage / alpaca calendar sources in this cycle (only news benzinga + alpaca per contract criterion 3/4).

## Plan steps

1. Add `bq_dataset_observability` Settings field.
2. Create `backend/news/bq_writer.py` with the three writer functions.
3. Replace `_write_batch_to_bq` stub in `fetcher.py`.
4. Harden `backend/news/sources/benzinga.py` + `alpaca.py` with observability.
5. Create `scripts/smoketest/phase6_e2e.py`.
6. Create `backend/tests/test_bq_writer.py`.
7. Run verification, capture into `phase-6.8-experiment-results.md`.

## References

- `handoff/current/phase-6.8-research-brief.md` (189 lines)
- `backend/news/fetcher.py:115-124` (stub to replace)
- `backend/news/sources/finnhub.py:53-147` (observability wiring pattern)
- `backend/services/observability/api_call_log.py:108-152,245-288` (insert_rows_json pattern)
- `backend/services/cycle_health.py:74-78` (JSONL heartbeat pattern for smoketest audit)
- External read-in-full: 3x BigQuery Write API docs, BQ ingestion methods Medium, idempotent pipelines Medium, OneUptime ETL Feb 2026, Airbyte ETL testing guide.

## Researcher agent id

`aeac146e0133c5c06`
