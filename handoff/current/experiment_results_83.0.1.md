# Experiment results — Step 83.0.1: news timestamp point-in-time integrity

Date: 2026-08-07 (autonomous drain, cycle 170). Contract: `contract_83.0.1.md`.

## What was built

1. **Live schema change (contract D1, route A)**: `ALTER COLUMN published_at DROP NOT NULL` + `ADD COLUMN effective_trade_date DATE`, executed while `streaming_buffer` was None (verbatim in `live_check_83.0.1.md`). Migration file synced: DDL now emits `published_at TIMESTAMP` (nullable) + `effective_trade_date DATE`; `REQUIRED_MODES` updated (published_at → NULLABLE, effective_trade_date → NULLABLE) so the post-condition verifier is the oracle for the altered table too.
2. **`backend/news/fetcher.py`** — the step's core: `_parse_published_at()` strict ISO-8601 parse (missing/empty/malformed → None; naive → UTC, documented); `_normalize` stores NULL + quarantines (`_QUARANTINE` per-reason counter + `quarantine_count()` + `reset_quarantine_for_test()` + WARNING log, mirroring the 83.0 idiom); `_derive_effective_trade_date()` RuleA (`date_to_session(pub_date + _EMBARGO_DAYS, "next")`, `_EMBARGO_DAYS = 1` as the single named constant) resolving the calendar via `markets.get_trading_calendar`/`market_for_symbol`, FAIL-CLOSED (no calendar/derivation error → NULL + `calendar_unresolvable` quarantine; `is_trading_day` deliberately unused — it fails open); `ingested_at` stays the run moment for backfill (truthful, D3); smoke extended.
3. **The three adapter fabrication sites removed (contract D2)** — finnhub/benzinga/alpaca no longer substitute `datetime.now()` for missing vendor timestamps; empty/malformed values now reach the chokepoint. Orphaned `datetime` imports pruned (lint-clean).
4. **`bq_writer.py`** — `_serialize_article` emits `effective_trade_date`; NULL `published_at` passes through (NULLABLE since the ALTER).
5. **Committed fixture corpus** `backend/tests/fixtures/news_pit_corpus.json` — 13 articles: missing/empty/None/malformed timestamps, 2022 backfill, intraday weekday, Fri/Sat/Sun, MLK + July-4 + Christmas holidays, no-ticker macro.
6. **`backend/tests/test_phase_83_0_1_news_timestamp_pit.py`** — 15 tests: C1 both halves in one test (value-is-None + strict counter increase) + parse-shape parametrization + negative control; C3 backfill-era assertion (`ing.year >= 2026`); C4 corpus minimum ≥1 trading day AND RuleA exactness (==1 session, so over-embargo also fails) + session-OPEN entry anchor test; C6 recorded count; per-adapter REAL-`fetch()` tests (HTTP + key gates patched at module seams — not a replayed loop, so a restored adapter fabrication is caught at the true seam); fail-closed derivation test.
7. **Knock-on updates**: 83.0 test oracle (`published_at` NULLABLE + `effective_trade_date` in the snapshot), `test_bq_writer` field-set. **Contract D5**: 83.0's `test_c6_finnhub_benzinga_byte_unchanged` RETIRED with a tombstone comment — its criterion was discharged at the 83.0 commit (06911cb5); as a living test it would forbid every future edit to those files including this step's authorized repair. Retired, not weakened.

## Verification (verbatim, re-derived AFTER the final edit)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_0_1_news_timestamp_pit.py -q
...............
15 passed in 2.21s
```

Adjacent suites (83.0.1 + 83.0 + bq_writer + calendar_watcher): **46 passed in 7.23s**. Fetcher inline smoke: OK (stub articles now also assert `published_at`/`effective_trade_date` non-null and embargoed). Lint gate over the git-derived changed-file scope: **"All checks passed!"** (10 F401s found and removed during the cycle — orphaned imports from the adapter repair plus test-file prunes).

## Mutation matrix — 5/5 KILLED (runner: scratchpad/mutation_matrix_83_0_1.py; anchors asserted count==1; restore hash-verified; matrix re-run WHOLE after the import prunes changed the tree)

| id | mutation | result |
|---|---|---|
| m1 | restore the `or _now_iso()` wall-clock fallback in `_normalize` (criterion 2's named mutant) | KILLED (4 failed) |
| m2 | `_EMBARGO_DAYS = 1` → `0` (criterion 5's named mutant) | KILLED (3 failed — the intraday-weekday fixture is what catches it; a weekend-only corpus would survive) |
| m3 | restore the `now()` substitution in alpaca.py (the LIVE fabrication site; inline import so it stays behavioural post-prune) | KILLED (1 failed — the real-`fetch()` adapter test) |
| m4 | delete the quarantine increment | KILLED (6 failed — strict-increase asserts) |
| m5 | `_parse_published_at` returns the malformed string instead of None | KILLED (5 failed) |

## Measured decisions and disclosures

- **RuleA exactness is asserted, not just the minimum**: the corpus test requires exactly 1 session in (pub, eff] — over-embargoing (the `next_session(date_to_session(...))` shape that skips Monday for Saturday news) fails too.
- **Priced concession (contract D6)**: the one-session embargo forecloses fast numeric-surprise strategies on this corpus (Martineau 2022 / Christensen et al. 2025 via EarningsInOne). Accepted deliberately for the weeks-to-months thematic channel, where sentiment IC peaks at the embargoed entry.
- **Four fabrication sites, not one (research finding)**: the step named fetcher.py:99; the live path's actual fabrication was upstream (alpaca — its keys are set; finnhub/benzinga inert-but-wrong). All four fixed; the chokepoint parse-predicate catches any future adapter regression, and the per-adapter tests pin each seam.
- **Pollution disclosure**: this step's live_check added 2 `source='fixture'` rows to the prod table (the backfill row + the persisted quarantined row) — added to the 83.0.7 purge scope (ask #8 updated). One additional fixture yield was dropped by intra-batch dedup (identical body), not quarantine.
- **The 9 stub rows' `effective_trade_date` is NULL** (written pre-column) — honest absence, no silent backfill.

## Post-verdict addenda (the PASS verdict's own NOTE findings, wf_5fd1a654-84e — recorded, not re-graded)

- **F1 correction (m2 attribution):** the experiment-results m2 row credited the intraday-weekday fixture as the kill; the Q/A measured the actual first tripping assert as `backfill_2022: only 0 sessions in (2022-03-15, 2022-03-15]`, and proved the guard STRONGER than claimed (three independent kill paths incl. two hardcoded in-test articles). The m2 row's parenthetical is superseded by this note.
- **F4 (the C6 record, quoted here so it lives in an artifact):** `QUARANTINE RECORDED: 4 of 13 corpus articles (matches loop count 4)`.
- **F2 queued as step 83.0.8** — dedup.py:98's cross-batch window is NULL-blind (dormant; no production caller). Same-turn masterplan addition.
- **F3/F5 accepted as residuals:** the fixture corpus is pinned only by count+delta-shape (case-name pinning left to 83.0.8-adjacent work if it bites); the persistence-seam guard for `effective_trade_date` lives in `test_bq_writer.py`, outside the immutable command's file but inside the adjacent suite.

## Files changed

`backend/news/fetcher.py`, `backend/news/sources/{finnhub,benzinga,alpaca}.py`, `backend/news/bq_writer.py`, `scripts/migrations/add_news_sentiment_schema.py`, `backend/tests/test_phase_83_0_1_news_timestamp_pit.py` (new), `backend/tests/fixtures/news_pit_corpus.json` (new), `backend/tests/test_phase_83_0_news_corpus_persistence.py` (oracle + D5 tombstone), `backend/tests/test_bq_writer.py` (field set). Handoff: contract, research brief, live_check, this file.
