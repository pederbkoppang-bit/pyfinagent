# Experiment results — Step 83.0: news corpus persistence

Date: 2026-08-07 (autonomous drain, cycle 168). Contract: `contract_83.0.md`.

## What was built

1. **`scripts/migrations/add_news_sentiment_schema.py`** — `fetched_at` → `ingested_at` in the news_articles DDL; `provenance STRING NOT NULL` added to BOTH tables; `ingested_at TIMESTAMP NOT NULL` added to news_sentiment (contract D1 — one-shot REQUIRED window); new `REQUIRED_MODES` map + `verify_post_condition()` that reads the schema back via `get_table().schema`, asserts non-empty, and exits LOUD on drift (CREATE TABLE IF NOT EXISTS is a no-op on an existing table). `--dry-run` unchanged.
2. **`backend/news/bq_writer.py`** — swallowed-write observability: `_WRITE_FAILURES` per-table counter + `threading.Lock`, `write_failure_count()`, `reset_write_failures_for_test()`, `_record_failure(table, reason, detail)` at the three real failure branches (`client_absent` / `insert_errors` / `exception`), deliberately NOT at the empty-input return; `_serialize_article` emits `ingested_at` (old-key compat lookup) + `provenance` with `_VALID_PROVENANCE` guard; `_serialize_sentiment` emits `ingested_at` (write-moment default) + `provenance`; module docstring corrected (contract D2 — the "BQ ignores unknown keys" claim was refuted by the research gate: `insert_rows_json` is strict by default).
3. **`backend/news/fetcher.py`** — `NormalizedArticle` gains `ingested_at`/`provenance`; `_normalize(..., provenance="live")`; `run_once(..., provenance="live")` threads it; inline smoke updated. The `published_at or _now_iso()` fallback is untouched — that is 83.0.1's scope, noted in a comment.
4. **`backend/tests/test_bq_writer.py`** — the two field-set-equality tests + fixtures updated in the same change (research: they go RED otherwise). Calendar sections untouched.
5. **`backend/tests/test_phase_83_0_news_corpus_persistence.py`** — NEW, 9 tests covering C1–C6 (schema oracle live-with-snapshot-fallback, asserted non-empty first; C3 both halves in one test + two negative controls; C5 isolated-registry two-adapter flow + structural sys.modules import-chain assertion; C6 git-diff over exactly the two adapter paths, `check=True` so git failure fails the test rather than skipping).
6. **The migration was RUN LIVE**: both tables created in `pyfinagent_data`; post-condition verifier passed. BEFORE/AFTER `bq show --schema` captures + the deliberate failed-write capture are in `live_check_83.0.md`.

## Files changed

`scripts/migrations/add_news_sentiment_schema.py`, `backend/news/bq_writer.py`, `backend/news/fetcher.py`, `backend/tests/test_bq_writer.py`, `backend/tests/test_phase_83_0_news_corpus_persistence.py` (new). Handoff: `contract_83.0.md`, `research_brief_83.0.md`, `live_check_83.0.md`, this file. `backend/news/sources/finnhub.py` + `benzinga.py` byte-unchanged (asserted by test C6 and `git diff HEAD --name-only` = empty over those paths).

## Verification (verbatim)

Immutable command:

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_0_news_corpus_persistence.py -q
.........                                                                [100%]
9 passed
```

Full adjacent suite: `pytest backend/tests/test_phase_83_0_news_corpus_persistence.py backend/tests/test_bq_writer.py -q` → **20 passed in 6.79s**. Fetcher inline smoke: `python backend/news/fetcher.py` → `phase-6.2 smoke: OK, n_articles=3`.

Rename blast radius verified: `grep -rn "fetched_at" --include="*.py" backend scripts` post-change returns ONLY the 7 protected calendar_events sites (watcher.py:51/:92, add_calendar_events_schema.py:49, bq_writer.py calendar serializer, test_calendar_watcher.py:245, test_bq_writer.py calendar section) plus comments and the intentional old-key compat lookup at bq_writer.py:174.

## Mutation matrix — 6/6 KILLED (runner: scratchpad/mutation_matrix_83_0.py; anchors asserted count==1; restore hash-verified per mutant)

| id | mutation | result |
|---|---|---|
| m1 | delete counter increment (`_WRITE_FAILURES[table] = ... + 1` removed) | KILLED (1 failed) |
| m2 | delete the `logger.warning` in `_record_failure` | KILLED (1 failed) |
| m3 | `+ 1` → `+ 0` (strict-increase check) | KILLED (1 failed) |
| m4 | count empty input as failure (negative-control guard) | KILLED (1 failed) |
| m5 | `_normalize` emits `provenance="live"` unconditionally | KILLED (1 failed) |
| m6 | schema-oracle vacuity: `_resolve_schema` returns `{}` | KILLED (1 failed) |

**Disclosure (m6 rewrite):** m6's first formulation mutated the snapshot-fallback line (`return dict(snapshot)` → `return {}`) and **SURVIVED** — the live BigQuery read succeeds on this rig, so the fallback is unreachable and the mutant was equivalent here. The matrix was corrected to short-circuit `_resolve_schema` itself (unconditional `return {}` after the docstring), which the non-empty-oracle assertion then kills. Recorded per `feedback_mutation_test_guards_and_fixtures` (first mutate the guard you catch yourself defending).

## Follow-up — cycle 2 (2026-08-07, after Q/A CONDITIONAL wf_b184df52-3e7)

The cycle-1 verdict (transcribed verbatim in `evaluator_critique_83.0.md`) found all 6 criteria MET and capped at CONDITIONAL on two WARNs. Both are fixed; one new incident was discovered while re-verifying and is disclosed + queued below.

**WARN-1 (--backfill provenance falsehood) — FIXED by threading the kwarg.** `scripts/smoketest/phase6_e2e.py` fetch stage now passes `provenance="backfill" if backfill else "live"` (the Q/A's alternate remedy; chosen over queue-only because the fix is 4 lines at the exact seam the Q/A named). Verified: `grep -n 'provenance=' scripts/smoketest/phase6_e2e.py` shows the threaded call.

**WARN-2 (lint gate unrecorded; 2x pre-existing F401) — FIXED and recorded.** The two dead imports at `backend/news/fetcher.py:39` (`NewsSource`, `clear_registry`) are removed — no consumer imports them via fetcher (`backend/news/__init__.py` takes both from `backend.news.registry` directly). A third F401 (`import time`, `scripts/smoketest/phase6_e2e.py`) entered the derived scope via the WARN-1 fix and was removed the same way. Lint gate now green, verbatim:

```
$ FILES=$( { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } )
$ echo "$FILES" | tr '\n' '\0' | xargs -0 uvx ruff check --select F821,F401,F811
All checks passed!
```

Re-verification after both fixes: immutable command + adjacent suite → **20 passed in 6.94s**; fetcher inline smoke OK; `phase6_e2e.py` full pipeline smoke → `ok: True` across all 8 stages.

**INCIDENT disclosed + queued as 83.0.7 (stub pollution; $0 spent).** Re-verifying WARN-1 exposed that the e2e smoke's `--dry-run` is NOT dry for the BQ writer stages — explicit write attempts run regardless (its own :108-111 comment). While the tables were absent this silently no-opped (the 83.0 defect itself); with the tables now live, my three smoke runs today wrote **9 `source='stub'` rows into news_articles and 9 rows into news_sentiment**. Measured: `SUM(cost_usd) = 0.0`; the 6 `claude-haiku-4-5` sentiment rows are scorer ERROR records (`RuntimeError('anthropic client not initialized')`, latency ~0.03ms) — **no LLM API spend occurred** (finbert is local CPU). The rows sit in the streaming buffer (DML-undeletable for ~30-90 min), so the purge is queued as step **83.0.7** with a qualified-DELETE plan and operator notification, per CLAUDE.md BQ rule 4. Nothing in the 83.0 criteria depends on table row contents (schema + writer-behaviour tests only).

## Follow-up — cycle 3 (2026-08-07, after Q/A CONDITIONAL #2 wf_1ff464d6-6f1)

Cycle-2 verdict (verbatim in `evaluator_critique_83.0.md`): both cycle-1 WARNs confirmed fixed by execution; capped by two NEW findings. Both are now closed:

**WARN-1 (sentiment provenance constant 'live'; contract D1 claim false) — FIXED at the write seam.** `write_news_sentiment` gains a `provenance` kwarg; `_serialize_sentiment(result, default_provenance)` precedence: explicit row value (if valid) > kwarg (if valid) > "live". `phase6_e2e` threads `"backfill" if backfill else "live"` into the sentiment write. `contract_83.0.md` carries an appended Correction section — the original D1 text is preserved as the record of the false claim. Guard: `test_c2_sentiment_writer_stamps_default_provenance` (all three precedence cases).

**WARN-2 (cycle-2 remediation unguarded) — FIXED with a behavioural test.** `test_smoke_pipeline_threads_provenance_at_both_seams` runs the real `_run_pipeline` with recorders patched at the fetch seam, all three BQ writers, the calendar watcher, observability flushes, and `_slack_heartbeat` (zero network, zero BQ), asserting the captured `provenance` kwarg at BOTH seams for backfill=True and False. Deleting either threading turns it red — proven in the extended matrix below.

**Mutation matrix re-run WHOLE (tests changed): 9/9 KILLED.**

| id | mutation | result |
|---|---|---|
| m1-m6 | (cycle-1 rows, unchanged) | all KILLED |
| m7 | delete the sentiment default stamp in `_serialize_sentiment` | KILLED (1 failed) |
| m8 | revert the phase6_e2e fetch-seam threading | KILLED (1 failed) |
| m9 | revert the phase6_e2e sentiment-seam threading | KILLED (1 failed) |

Suite after cycle 3: **22 passed** (`test_phase_83_0_news_corpus_persistence.py` 11 + `test_bq_writer.py` 11). Immutable command re-run separately: **11 passed in 4.91s** (measured; an earlier draft of this section said 12 — corrected against the run).

## Scope disclosures

- **The live news path is DISJOINT from this corpus** (research SCOPE FINDING): `news_screen.py` (RSS → LLM → local file cache → `screener.py:334`) never imports `backend.news` and still persists nothing. 83.0 is the schema prerequisite, not the capture of what moves the book. Queued as its own step (83.0.4) in the same masterplan edit that closes this step.
- **`api_call_log` table ABSENT** while `api_call_log.py:148` writes to it fail-open (same defect class, different table). Queued as 83.0.5.
- **FINNHUB/BENZINGA keys absent** (criterion 6): presence checked via the settings object (direct `.env` grep is permission-denied in this session); verbatim output in `live_check_83.0.md`. Both adapters degrade to `[]` when the key is empty — reachable-but-inert, byte-unchanged.
- **Alpha Vantage licence decision stays OPEN** (operator ask #6); nothing here imports it (test C5 enforces).
