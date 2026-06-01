---
name: cron-maintenance-jobs
description: phase-51.4 cron repairs -- autoresearch run_memo.py huggingface-dep skip + weekly_data_integrity BQ wiring; the two-name autoresearch collision; client.client.query idiom
metadata:
  type: project
---

phase-51.4 fixed two ISOLATED maintenance-job bugs (NOT the trading loop). Researched 2026-06-01.

**Bug A -- autoresearch nightly launchd job (`scripts/autoresearch/run_memo.py`).**
- `:155` sets `EMBEDDING: "huggingface:BAAI/bge-small-en-v1.5"`; gpt-researcher `GPTResearcher.__init__` builds `Memory(...)` UNCONDITIONALLY at `.venv/.../gpt_researcher/agent.py:173`, and `memory/embeddings.py:159-162` does `from langchain_huggingface import HuggingFaceEmbeddings`. `langchain_huggingface` + `sentence_transformers` are ABSENT -> ModuleNotFoundError every 02:00 -> ERROR file + exit 1. ZERO successful memos ever (32 ERROR files; `root_cause.md` is a post-mortem, not a memo).
- Fix = graceful preflight in `main()` AFTER the `os.environ.setdefault` loop (after :159), BEFORE `asyncio.run`: parse provider from `EMBEDDING`, `importlib.util.find_spec(backing_module)`; if None -> print actionable "pip install langchain-huggingface sentence-transformers" message + `return 0` (clean skip, launchd records success). No pip, no launchd unload, feature self-enables once dep installed.
- NO zero-dep embedding swap exists: no OPENAI_API_KEY (openai/custom 401), no Ollama daemon. graceful-skip is the right call. Enable path = `pip install langchain-huggingface sentence-transformers` (CPU/free/no-key, owner-gated).
- **CRITICAL NAME COLLISION:** `scripts/autoresearch/run_memo.py` (nightly literature MEMO) is UNRELATED to the `backend/autoresearch/` PACKAGE (strategy-rotation/promotion: cron.py, promoter.py, gate.py, friday_promotion.py). Fixing/disabling run_memo.py has ZERO effect on the rotation package. Do NOT conflate.
- run_memo.py is NON-critical-path: nothing in the live loop / Layer-3 gate consumes the memos at runtime (only `.claude/context/research-gate.md:22` references them as an optional research-gate convenience that has never fired). launchd: `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` -> `run_nightly.sh` -> run_memo.py, Hour=2.

**Bug B -- weekly_data_integrity (`backend/slack_bot/jobs/weekly_data_integrity.py::_default_fetch_counts` :78-90).**
- `:84` `BigQueryClient()` bare -> TypeError (`__init__(self, settings: Settings)` at `bigquery_client.py:22` requires settings). `:86` `client.query(sql)` -> AttributeError (no generic `query()` method; only `query_latest_signal_state`). Both swallowed by fail-open `except` -> empty `{}` -> always 0 drifts -> functionally inert NO-OP.
- Fix: `:84` -> `BigQueryClient(get_settings())` (universal idiom; `get_settings()` at `settings.py:468` loads from env; cf. `yfinance_tool.py:127`). `:86` -> `client.client.query(sql).result(timeout=30)` (reach the underlying `google.cloud.bigquery.Client` exposed at `bigquery_client.py:35`; the `.client.query(...).result()` idiom is used 60+ times in that file). Do NOT add a generic `query()` helper for one caller.
- The __TABLES__ SQL (`SELECT table_id, row_count FROM \`{project}.{dataset}.__TABLES__\``, dataset=pyfinagent_data) is correct + a FREE metadata read: LIVE-proven bytes_billed=0, 2.7s, 9 tables. `__TABLES__` is officially DEPRECATED (use `INFORMATION_SCHEMA.TABLE_STORAGE.total_rows`, region-qualified) but still works -- soft-modernization note, not a blocker. `INFORMATION_SCHEMA.TABLES` does NOT have row_count (only TABLE_STORAGE does).

**$0 verification:** Bug B -> `python -c "from backend.slack_bot.jobs.weekly_data_integrity import _default_fetch_counts; print(_default_fetch_counts())"` -> populated dict. Bug A -> `python scripts/autoresearch/run_memo.py --topic-index 0; echo $?` (dep absent) -> "skipped: ..." line + exit 0 + no new ERROR file (spends $0 LLM; gate returns before GPTResearcher built).

**Why:** operator-reported issue 2 (cron health); stop nightly failure noise + make the inert integrity check actually run, without pip/feature-removal.
**How to apply:** if asked to "fix the cron" or touch run_memo.py / weekly_data_integrity, these are the exact lines + the pip-free fixes. Remember the two-name autoresearch collision before reasoning about impact.
