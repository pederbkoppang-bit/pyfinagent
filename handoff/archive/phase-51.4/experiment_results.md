# experiment_results -- phase-51.4: cron repairs (autoresearch graceful-skip + weekly_data_integrity BQ wiring)

**Step:** 51.4 | **Date:** 2026-06-01 | **$0 LLM** | no pip | isolated maintenance jobs | GENERATE complete

## What was changed (operator-reported issue 2 -- the last of the 4)

| File | Change |
|------|--------|
| `backend/slack_bot/jobs/weekly_data_integrity.py:84-90` | **Bug B:** `BigQueryClient()` -> `BigQueryClient(get_settings())` (was missing the required settings arg -> TypeError fail-open -> `{}`); `client.query(sql)` -> `client.client.query(sql).result(timeout=30)` (no generic `.query()` exists). Now returns a populated `{table_id: row_count}` dict from a FREE `__TABLES__` read. |
| `scripts/autoresearch/run_memo.py` | **Bug A:** NEW `_embedding_preflight()` helper + a call in `main()` (after the env setdefault loop, before the run): if the configured EMBEDDING backend module is absent (`importlib.util.find_spec`), print an actionable "pip install ..." message + `return 0` (clean skip). Stops the nightly ModuleNotFoundError -> exit-1 + ERROR-file spam WITHOUT a pip or removing the feature. |
| `backend/tests/test_phase_51_4_crons.py` | **NEW** 4 tests: preflight skips (backend absent) / proceeds (present) / proceeds (unknown provider); weekly_data_integrity returns a populated dict + constructs the client WITH settings. |

## Research basis (gate PASSED, LIVE-verified)
`research_brief.md` (researcher `aaf3be5a051c5de01`, 6 sources, gate_passed). Both bugs revalidated; both fixes proven $0/pip-free/feature-preserving. `__TABLES__` is a FREE metadata read (probe: bytes_billed=0). autoresearch is NOT critical-path; `run_memo.py` (literature memo) != `backend/autoresearch/` (rotation package). NO zero-dep embedding swap exists (no OPENAI_API_KEY, no Ollama) -> graceful-skip is the correct resolution.

## Verification command output (verbatim)

### Syntax
```
OK  scripts/autoresearch/run_memo.py
OK  backend/slack_bot/jobs/weekly_data_integrity.py
OK  backend/tests/test_phase_51_4_crons.py
```

### pytest (phase-51.4 -- 4 tests)
```
$ python -m pytest backend/tests/test_phase_51_4_crons.py -q
....                                                                     [100%]
4 passed in 0.73s
```

### LIVE $0 proofs -> handoff/current/live_check_51.4.md
- Bug B: `_default_fetch_counts()` -> populated dict, **9 tables** (alt_congress_trades=7262, llm_call_log=370, ...). Was `{}`.
- Bug A: `run_memo.py --topic-index 0` -> **exit=0** + skip message + **NO new ERROR file** (32 -> 32). $0.

## Byte-identity / safety
- Isolated maintenance jobs; the trading loop + paper-trading routes are untouched (diff = the 2 jobs + the new test).
- Bug B keeps its fail-open try/except (a BQ error -> {} as before).
- Bug A: NO pip (owner-gated); the feature is preserved (config kept) and self-enables on the pip install. autoresearch != rotation package -> rotation untouched.

## Artifact shape
- `_embedding_preflight() -> str | None` (skip message or None)
- `_default_fetch_counts() -> {table_id: int}` (populated)

## Operator note + decision recorded
- **autoresearch decision (criterion #2): graceful-skip** (recorded). Enable path = owner-approved `pip install langchain-huggingface sentence-transformers`; the preflight self-enables once present.
- Live activation: weekly_data_integrity fix -> next Mon 05:00 UTC after a slack-bot restart; autoresearch skip -> next 02:00 fire. The direct-call proofs above are the $0 gate evidence.

## Session milestone
51.4 closes the LAST of the operator's 4 reported issues (weekend digest 51.3, dead signals 51.1, tech concentration 51.2-measured, broken crons 51.4). Remaining roadmap: calendar_events table, 50.6 UI, and MEASURE Monday's first multi-market cycle.
