# Contract — Step 83.0.1: news timestamp point-in-time integrity (NULL + quarantine, honest ingest, one-session embargo)

- **Step id:** 83.0.1 (P0, phase-83; depends_on: 83.0 — done, commit 06911cb5)
- **Tier (named field):** T3 — executor Main (Opus 5, effort max); Q/A via qa-verdict Workflow (opus/max).
- **Date:** 2026-08-07, autonomous drain, cycle 170

## Research-gate summary

`handoff/current/research_brief_83.0.1.md` — gate_passed: **true** (5 external sources read in full / 23 URLs / recency scan / 17 internal files; envelope on the rail AND in the brief). Decisive findings:

1. **(A) verdict with vendor certainty**: BigQuery permits a NULLABLE time-unit partitioning column — verbatim: *"While the mode of the column can be REQUIRED or NULLABLE, it cannot be REPEATED"* (partitioned-tables Limitations); `ALTER COLUMN DROP NOT NULL` has exactly three restrictions, none about partition columns; NULL rows land in the `__NULL__` partition. No quarantine table, no recreate, no owner-approved DROP.
2. **The step premise is incomplete — FOUR fabrication sites, not one.** finnhub.py:143-145, benzinga.py:146-149, alpaca.py:145-148 substitute wall-clock UPSTREAM, so the named fetcher fallback (now at fetcher.py:102) is **dead code on the live path**; alpaca is the LIVE fabrication site (its keys are set). Mode 4: malformed non-empty vendor strings pass every presence check → the quarantine predicate must be PARSE-based at the `_normalize` chokepoint.
3. `ingested_at` stays REQUIRED — the backfill-RUN timestamp is truthful and satisfies criterion 3.
4. **effective_trade_date RuleA**: `cal.date_to_session(pub_date + 1 day, direction="next")` — measured 0 violations over 2022-01-01..2026-06-30 with min sessions in (pub, eff] == 1 exactly; the `next_session(date_to_session(...))` alternative double-embargoes weekend/holiday news. `exchange_calendars==4.13.2` already pinned; reuse `backend/backtest/markets.py::get_trading_calendar`. **markets.py `is_trading_day` fails OPEN — do not use it; the embargo derivation fails CLOSED into quarantine.**
5. NextOpen entry is the field-standard anchor (EarningsInOne, arXiv:2606.29734); the one-session embargo costs nothing for the slow thematic channel (sentiment IC peaks at the next open).
6. Live table verified: 9 rows, `streaming_buffer: None` → the ALTER is unobstructed TODAY, run it before the next ingest.

## Immutable success criteria (verbatim from `.claude/masterplan.json` 83.0.1)

1. "a fixture RawArticle carrying no published_at produces a persisted row whose published_at is NULL, and increments a quarantine counter; the test asserts the emitted value is NULL, not merely that a counter exists"
2. "the NULL-publication guard is mutation-tested: restoring the `or _now_iso()` fallback at backend/news/fetcher.py makes the criterion-1 test FAIL"
3. "for provenance='backfill' rows the ingest timestamp is never a value earlier than the publication timestamp, asserted by a test over a fixture backfilling a 2022-dated article, which fails if ingested_at is written as the article's own era"
4. "an `effective_trade_date` is derived and persisted, and a test asserts that across a committed fixture corpus the minimum of (effective_trade_date minus the calendar date of published_at) is at least one trading day, and that the fixture's simulated entry price is the session OPEN rather than the publication-day close"
5. "the embargo guard is mutation-tested: setting the embargo to zero days makes the criterion-4 assertion FAIL"
6. "the count of rows quarantined for a missing publication timestamp is RECORDED for the fixture corpus rather than asserted against a threshold"

**Verification command (immutable):** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_0_1_news_timestamp_pit.py -q`

**live_check (immutable):** "verbatim fetcher output over a fixture article with no publication timestamp showing published_at NULL and the quarantine counter incrementing; plus a verbatim 5-row dump from pyfinagent_data.news_articles showing published_at, ingested_at, provenance and effective_trade_date together, including at least one provenance='backfill' row whose ingested_at is not fabricated into the article's own era" → `handoff/current/live_check_83.0.1.md`. NOTE: the quarantine-row query must NOT carry a `DATE(published_at)` partition predicate or the `__NULL__`-partition row is silently excluded.

## Explicit decisions

- **D1 — schema route (A):** `ALTER TABLE news_articles ALTER COLUMN published_at DROP NOT NULL` + `ADD COLUMN effective_trade_date DATE` (NULLABLE by BQ rule — correct: a quarantined row has none), executed live FIRST while `streaming_buffer` is None; migration DDL + `REQUIRED_MODES` updated to match so a fresh CREATE equals the altered table (post-condition verifier reused as the oracle).
- **D2 — scope widened to the three adapter fabrication sites** (finnhub/benzinga/alpaca `or datetime.now()` substitutions removed so truth reaches the chokepoint) because criterion 2's named line is dead code on the live path — a literal-scope fix would ship nothing. The quarantine predicate is PARSE-based (`fromisoformat`-style strict parse; missing, empty, and malformed all → NULL + quarantine) at `_normalize`, the single seam that catches all four modes and any future adapter.
- **D3 — `ingested_at` stays REQUIRED**; for backfill rows it records the backfill-RUN moment (truthful; ≥ published_at for historical articles).
- **D4 — effective_trade_date RuleA, fail-CLOSED**: calendar resolved via `markets.py::get_trading_calendar` (article's market when resolvable, XNYS default); resolution or derivation failure → `effective_trade_date` NULL + quarantine. `is_trading_day` (fail-open) is not used. Behind a single named constant `_EMBARGO_DAYS = 1` so the criterion-5 mutant is a one-token edit.
- **D5 — 83.0's `test_c6_finnhub_benzinga_byte_unchanged` is RETIRED with a tombstone comment.** Its criterion was discharged at the 83.0 commit (06911cb5 shows the empty diff); as a living test it would forbid every future edit to those files, including this step's authorized repair. Disclosed here so the Q/A does not read the red-then-retired test as evasion.
- **D6 — priced concession, recorded:** the one-session embargo permanently forecloses fast numeric-surprise strategies on this corpus (Martineau 2022 / Christensen et al. 2025: multi-day PEAD gone for non-microcaps since ~2006). Deliberately accepted: the operator's channel is weeks-to-months and the slow qualitative signal peaks exactly at the embargoed entry.
- **D7 — quarantine counter** is a fetcher-module counter (`_QUARANTINE` + `quarantine_count()` + `reset_quarantine_for_test()`), mirroring the 83.0 `_WRITE_FAILURES` idiom — ingest-semantics events, distinct from write failures.

## Plan

1. Live ALTER (D1) + migration file update; post-condition re-verified.
2. `fetcher.py`: `_parse_published_at()` strict parse; `_normalize` NULL+quarantine path; `effective_trade_date` derivation (D4); `ingested_at` semantics per D3; smoke updated.
3. Adapters: remove the three wall-clock substitutions (D2), byte-minimal.
4. `bq_writer.py`: serializer emits `effective_trade_date`; NULL `published_at` passes through.
5. Committed fixture corpus `backend/tests/fixtures/news_pit_corpus.json` (~10-14 articles: missing/empty/None/malformed timestamps, 2022 backfill article, intraday weekday, Fri/Sat/Sun, MLK + July-4 + Christmas holidays) — the weekday case is what keeps the zero-day mutant killable.
6. New test file (criteria 1-6, both halves of criterion 1 in one test, negative controls: successful parse and empty batch move nothing) + per-adapter seam tests (missing-timestamp vendor payloads → NULL) + oracle updates in the 83.0 test file (published_at NULLABLE + effective_trade_date) + D5 tombstone.
7. Mutation matrix, run for real with red output pasted: (m1) restore `or _now_iso()` at fetcher; (m2) `_EMBARGO_DAYS = 0`; (m3) restore `datetime.now()` substitution in alpaca.py (the LIVE site — kills the per-adapter test); (m4) delete the quarantine increment; (m5) make `_parse_published_at` accept malformed strings.
8. Live captures per the plan (fetcher stdout; ALTER output; 5-row dump incl. backfill row + `__NULL__`-partition quarantine row); backfill row produced via `phase6_e2e.py --backfill` (stub source, provenance='backfill' threading landed in 83.0).
9. `experiment_results_83.0.1.md` → qa-verdict Workflow → transcribe → harness_log → flip. Re-derive every fenced measurement AFTER the final edit (Cycle-169 lesson).

## References

`research_brief_83.0.1.md` (vendor docs: partitioned-tables Limitations, managing-table-schemas column relaxation, DDL ALTER COLUMN DROP NOT NULL; EarningsInOne arXiv:2606.29734 read in full via pdfplumber; FNSPID arXiv:2402.06698 field-norm comparison; in-repo markets.py calendar audit with the fail-open warning).
