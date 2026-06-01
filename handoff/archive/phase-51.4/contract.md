# Contract -- phase-51.4: cron repairs (autoresearch graceful-skip + weekly_data_integrity BQ wiring)

**Step id:** 51.4 | **Priority:** P2 (integrity; operator-reported issue 2) | **depends_on:** 49.3
**Date:** 2026-06-01 | **harness_required:** true | **$0 LLM** | **no pip** | isolated maintenance jobs (NOT the trading loop)

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher `aaf3be5a051c5de01`: gate_passed=true, tier moderate, 6 sources read in full, 16 URLs, recency scan, 8 internal + 2 LIVE probes). Decisive + LIVE-verified:
- **Bug A (autoresearch):** `run_memo.py:155` sets `EMBEDDING="huggingface:..."` but `langchain_huggingface` + `sentence_transformers` are ABSENT -> `GPTResearcher.__init__` builds `Memory(...)` unconditionally -> `from langchain_huggingface import ...` -> ModuleNotFoundError every 02:00 -> exit 1 + ERROR file. **32 ERROR files, ZERO successful memos ever.** NOT on any critical path (the live loop / Layer-3 gate / rotation `backend/autoresearch/` package do NOT consume run_memo.py memos -- different systems). NO zero-dep embedding swap exists (no OPENAI_API_KEY, no Ollama). FIX: a graceful preflight (NOT pip, NOT feature-removal).
- **Bug B (weekly_data_integrity):** `_default_fetch_counts` `:84` `BigQueryClient()` (missing required `settings`) -> TypeError fail-open -> empty {}; `:86` `client.query(sql)` -> AttributeError (no generic query()). The __TABLES__ row-count SQL is already correct + is a FREE metadata read (LIVE-proven today: bytes_billed=0, 2.7s, 9 tables).

## Hypothesis
(A) A `importlib.util.find_spec` preflight in `run_memo.py main()` (after the `os.environ.setdefault` loop at :159, before the run at :164) that detects the EMBEDDING provider's backing module is absent, logs an actionable "pip install langchain-huggingface sentence-transformers to enable" message, and `return 0` -> the nightly job STOPS silently failing (clean exit-0 skip, no ERROR file) WITHOUT a pip or removing the feature; it self-enables once the operator installs the dep. (B) `BigQueryClient(get_settings())` + `client.client.query(sql).result(timeout=30)` makes `_default_fetch_counts` return a populated `{table_id: row_count}` dict so the weekly data-integrity drift check can actually compute. Both are isolated maintenance plumbing -> the working trading path is untouched.

## Success criteria (IMMUTABLE -- verbatim from masterplan step 51.4)
1. weekly_data_integrity constructs BigQueryClient(get_settings()) and executes a real __TABLES__ row-count query returning a populated dict (not {}), so the data-integrity drift check can actually compute and alert
2. the autoresearch nightly job either succeeds (produces a non-ERROR memo) OR is explicitly disabled / owner-gated with a recorded decision -- it must STOP silently failing every night
3. no change to the working trading path (both jobs are isolated maintenance plumbing)
4. live_check records the weekly_data_integrity real-count proof + the autoresearch decision/outcome

**Verification command:** `pytest backend/tests/test_phase_51_4_crons.py` + `ast.parse(weekly_data_integrity.py, run_memo.py)` + `test -f live_check_51.4.md`.
**live_check:** REQUIRED -- weekly_data_integrity returns real row-counts (run `_default_fetch_counts()` -> populated dict); autoresearch preflight -> exit 0 clean skip, no new ERROR file.

## Plan steps (GENERATE)
1. **Bug B (weekly_data_integrity.py):** `:84` `BigQueryClient()` -> `BigQueryClient(get_settings())` (lazy `from backend.config.settings import get_settings`); `:86` `client.query(sql)` -> `client.client.query(sql).result(timeout=30)` mapped to a `{table_id: row_count}` dict. (Do NOT add a generic BigQueryClient.query() helper -- single caller; `.client.query().result()` is the 60+-site idiom.) Keep the existing fail-open try/except.
2. **Bug A (run_memo.py):** insert a preflight in `main()` after the `os.environ.setdefault` loop (:159) + the ANTHROPIC_API_KEY guard, before `return asyncio.run(_main_async(args))` (:164): parse the provider from `os.environ["EMBEDDING"]`, map (`huggingface`->`langchain_huggingface`), `importlib.util.find_spec`; if absent -> log/print "autoresearch skipped: pip install langchain-huggingface sentence-transformers to enable" + `return 0`. The existing broad try/except stays as the backstop. Keep the huggingface EMBEDDING config (self-enables on install).
3. **Tests:** `backend/tests/test_phase_51_4_crons.py` -- (a) the run_memo preflight returns 0 + emits the skip message when find_spec is monkeypatched None, and does NOT return 0 (proceeds) when present (monkeypatch find_spec truthy + stub _main_async to avoid a real run); (b) weekly_data_integrity `_default_fetch_counts` maps BQ rows to a dict (monkeypatch the BQ client to return fake __TABLES__ rows -> assert populated dict; and that it constructs BigQueryClient WITH settings). $0, no network.
4. **Verify:** pytest; ast.parse both; LIVE: `_default_fetch_counts()` -> populated dict (real free __TABLES__ read); `run_memo.py --topic-index 0` -> exit 0 + skip message + NO new ERROR file ($0 -- returns before GPTResearcher). Capture into live_check_51.4.md.
5. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 51.4 -> done.

## Safety / scope notes
- **Isolated maintenance jobs.** Diff = weekly_data_integrity.py + run_memo.py + the new test. NO trading-loop / paper-trading / risk-guard / rotation-package change.
- **No pip** (owner-gated): Bug A is a graceful-skip, NOT a pip install. The operator's enable path (pip langchain-huggingface) is recorded; the preflight self-enables once present.
- **Bug B is a FREE metadata read** (LIVE-proven bytes_billed=0); the existing fail-open try/except stays so a BQ hiccup never crashes the weekly job.
- **autoresearch != rotation:** `scripts/autoresearch/run_memo.py` (literature memo) is a DIFFERENT system from the `backend/autoresearch/` rotation package -> this touches neither rotation nor the trading loop.
- Live activation of the weekly_data_integrity fix needs the slack-bot to pick up the code (next Monday 05:00 UTC fire after a restart); the `_default_fetch_counts()` direct call is the $0 gate proof.
- $0 LLM; no pip; no spend; no DROP/DELETE.

## References
- handoff/current/research_brief.md (51.4 gate)
- scripts/autoresearch/run_memo.py:155 (EMBEDDING), :159 (setdefault loop), :164 (run) ; run_nightly.sh + com.pyfinagent.autoresearch.plist
- backend/slack_bot/jobs/weekly_data_integrity.py:84,86 (_default_fetch_counts) ; backend/db/bigquery_client.py:22 (__init__ settings), :35 (.client) ; backend/config/settings.py:468 (get_settings)
- BQ __TABLES__ row_count (free metadata read; TABLE_STORAGE.total_rows is the modern alt -- soft, not blocking)
