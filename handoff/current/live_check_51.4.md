# live_check -- phase-51.4: cron repairs (autoresearch graceful-skip + weekly_data_integrity BQ wiring)

**Step:** 51.4 | **Date:** 2026-06-01 | **Result shape:** (B) weekly_data_integrity returns a
populated `{table_id: row_count}` dict from a real FREE `__TABLES__` read; (A) autoresearch
preflight exits 0 with a clean actionable skip and NO new ERROR file. $0 LLM, no pip.

## Bug B -- weekly_data_integrity real __TABLES__ read (criterion #1)
```
$ python -c "from backend.slack_bot.jobs.weekly_data_integrity import _default_fetch_counts; print(_default_fetch_counts())"
populated dict? True | n_tables = 9
sample: {'alt_13f_holdings': 110, 'alt_congress_trades': 7262, 'alt_finra_short_volume': 0, 'llm_call_log': 370, 'risk_intervention_log': 0}
```
- Was `{}` before (BigQueryClient() TypeError fail-open + nonexistent client.query()). Now `BigQueryClient(get_settings())` + `client.client.query(sql).result(timeout=30)` returns 9 real per-table row counts.
- FREE metadata read (research probe today: bytes_billed=0). The drift check can now actually compute + alert.
- 4/4 unit tests pass (incl. `test_weekly_data_integrity_returns_populated_dict` asserting populated dict AND construction WITH a settings arg).

## Bug A -- autoresearch graceful preflight (criterion #2)
```
$ ANTHROPIC_API_KEY=preflight-test-dummy python scripts/autoresearch/run_memo.py --topic-index 0
exit=0
autoresearch skipped: embedding provider 'huggingface' needs 'langchain_huggingface', which is not installed. Enable with: pip install langchain-huggingface sentence-transformers
ERROR files before=32 after=32 (delta should be 0)
```
- Was: ModuleNotFoundError -> exit 1 + a new `handoff/autoresearch/YYYY-MM-DD-ERROR-topicNN.md` EVERY night (32 ERROR files, zero successful memos).
- Now: a `importlib.util.find_spec` preflight detects `langchain_huggingface` is absent -> prints an actionable enable message -> **exit 0** (clean skip), **NO new ERROR file** (32 -> 32), **$0 LLM** (returns before GPTResearcher is constructed). The job self-enables once the operator runs `pip install langchain-huggingface sentence-transformers`.
- The dummy ANTHROPIC_API_KEY is only to pass the existing env guard; the preflight returns before any Anthropic call, so it is never used (no spend).

## Criterion-by-criterion

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | weekly_data_integrity constructs BigQueryClient(get_settings()) + a real __TABLES__ query returning a populated dict (not {}) | the 9-table dict above + the unit test | PASS |
| 2 | autoresearch either succeeds OR is explicitly disabled/owner-gated w/ a recorded decision -- must STOP silently failing nightly | preflight -> exit 0 clean skip, NO new ERROR file; decision recorded (graceful-skip, NOT pip/feature-removal; operator enable path = pip langchain-huggingface) | PASS |
| 3 | no change to the working trading path (isolated maintenance plumbing) | diff = weekly_data_integrity.py + run_memo.py + new test ONLY; autoresearch run_memo.py != the rotation `backend/autoresearch/` package; no trading-loop/paper-route/risk-guard touched | PASS |
| 4 | live_check records the weekly_data_integrity real-count proof + the autoresearch decision/outcome | this file | PASS |

## Decision recorded (autoresearch, criterion #2)
**Resolution = graceful-skip, NOT pip (owner-gated) and NOT feature removal.** autoresearch is NOT
on any critical path (the live loop / Layer-3 research gate / rotation package do not consume
run_memo.py memos -- the only reference is a never-fired doc convenience). The huggingface
EMBEDDING config is preserved; the preflight self-enables the moment the operator runs the pip
install. This STOPS the silent nightly failure today without spending owner-gated pip approval.

## Scope / notes
- Isolated maintenance jobs; the trading loop is untouched.
- Bug B keeps its fail-open try/except (a BQ hiccup -> {} as before, never crashes the weekly job).
- Live activation: the weekly_data_integrity fix takes effect on the next Monday 05:00 UTC fire after a slack-bot restart; the autoresearch skip takes effect on the next 02:00 fire (or immediately on a manual run). The direct `_default_fetch_counts()` call + the run_memo invocation above are the $0 gate proofs.
- __TABLES__ is officially deprecated in favor of INFORMATION_SCHEMA.TABLE_STORAGE.total_rows -- a soft modernization, NOT scoped here (it still works, free).
