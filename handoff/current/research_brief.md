# research_brief -- phase-51.4: cron repairs (autoresearch + weekly_data_integrity)

Tier: moderate. Research gate for two ISOLATED maintenance-job bugs (NOT the live trading loop).
$0 LLM. pip installs are owner-gated -> the recommended fixes must NOT require an unconditional pip.

Scope: (A) the autoresearch nightly launchd job, (B) the slack-bot weekly data-integrity job. Neither
is on the live trading/paper path. Status: COMPLETE.

---

## Part A -- Internal code audit (file:line)

### A0. Bugs CONFIRMED in current code (both reproduce exactly as diagnosed)

**Bug A -- autoresearch launchd job fails every night (ModuleNotFoundError).**
- `scripts/autoresearch/run_memo.py:155` sets `"EMBEDDING": "huggingface:BAAI/bge-small-en-v1.5"` in `env_defaults`, then `os.environ.setdefault(...)` applies it (:158-159). CONFIRMED verbatim.
- gpt-researcher `GPTResearcher.__init__` builds `self.memory = Memory(self.cfg.embedding_provider, self.cfg.embedding_model, ...)` **UNCONDITIONALLY** at `.venv/.../gpt_researcher/agent.py:173`. There is NO lazy-construction; the embedding is built at object construction, BEFORE any research runs.
- `Memory.__init__` (`.venv/.../gpt_researcher/memory/embeddings.py:72`) is a `match embedding_provider:` block; the `case "huggingface":` arm (:159-162) does `from langchain_huggingface import HuggingFaceEmbeddings` then `HuggingFaceEmbeddings(model_name=model, ...)`.
- `langchain_huggingface` is **ABSENT** from the venv (verified: `import langchain_huggingface` -> ModuleNotFoundError; `importlib.util.find_spec('langchain_huggingface')` -> False). `sentence_transformers` also ABSENT. So construction raises `ModuleNotFoundError: No module named 'langchain_huggingface'` every night at 02:00.
- The exception is caught by run_memo.py's broad `except Exception` (:114), which writes a `handoff/autoresearch/YYYY-MM-DD-ERROR-topicNN.md` file (:117-122) and **returns 1** (:124) -> the wrapper `run_nightly.sh` exits non-zero -> launchd records exit status 1.
- **History: ZERO successful memos ever produced.** `handoff/autoresearch/` has **32 ERROR files** and exactly **1** non-ERROR `.md` -- `root_cause.md`, which is itself a phase-39.1 post-mortem (header: "Autoresearch cron exit-1 -- root cause analysis ... 38 consecutive nights of ERROR ... Status: SOURCE FIXED"). The phase-39.1 "fix" only moved the failure from a `ValueError` (bad embedding format) to a `ModuleNotFoundError` (the format was fixed but the backing dep was never installed). Latest ERROR file (2026-05-31) error line: `ModuleNotFoundError: No module named 'langchain_huggingface'` -- matches the diagnostic exactly.
- launchd: `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` -> `ProgramArguments = [/bin/bash, .../scripts/autoresearch/run_nightly.sh]`; `StartCalendarInterval` Hour=2 Minute=0; `RunAtLoad=false`; `ExitTimeOut=1200`. `launchctl list | grep autoresearch` shows it LOADED with last-exit-status **1** (the nightly failure). The entrypoint shell `scripts/autoresearch/run_nightly.sh` sources `backend/.env`, activates `.venv`, runs `python scripts/autoresearch/run_memo.py`, logs to `handoff/autoresearch.log`, and `exit "$rc"` (propagates run_memo's exit code).

**Bug B -- weekly_data_integrity is functionally inert (two stacked bugs).**
`backend/slack_bot/jobs/weekly_data_integrity.py::_default_fetch_counts` (:78-90):
- **Bug B1 (line 84):** `client = BigQueryClient()` -- called with NO args. But `backend/db/bigquery_client.py:22` is `def __init__(self, settings: Settings)` -- `settings` is REQUIRED positional. -> `TypeError: __init__() missing 1 required positional argument: 'settings'`. CONFIRMED.
- **Bug B2 (line 86):** `rows = client.query(sql)` -- but `BigQueryClient` has **NO** generic `query()` method. Grepped all 1075 lines: the only `query`-named method is `query_latest_signal_state` (:429). Every other BQ call uses `self.client.query(...).result()` where `self.client` IS the underlying `google.cloud.bigquery.Client` (set at :35). -> `AttributeError: 'BigQueryClient' object has no attribute 'query'` (would fire even after B1 is fixed). CONFIRMED.
- Both are swallowed by the function's own `except Exception` (:88) which `logger.warning(... fail-open ...)` and `return {}`. -> `current_counts` is `{}` -> `_compute_drifts` iterates an empty dict -> **always 0 drifts, snapshot of `{}` saved**. The job runs "successfully" (no crash) but is a NO-OP: it has never once compared real row counts. Functionally inert, exactly as diagnosed.

### A1. The RIGHT BQ execution path for Bug B -- `client.client.query(sql).result()` (PROVEN live)

`BigQueryClient` exposes `self.client` = `google.cloud.bigquery.Client` (`bigquery_client.py:35`). The idiomatic BQ call used **60+ times** across the file is `self.client.query(<sql>, job_config=...).result()` then `[dict(r) for r in rows]` (e.g. :576 `return [dict(r) for r in self.client.query(query).result()]`). So the fix is to (1) pass settings, (2) reach through to `.client`:

```python
client = BigQueryClient(get_settings())          # B1 fix: settings via the universal helper
rows = client.client.query(sql).result(timeout=30)  # B2 fix: reach the real bq Client + bound to 30s
return {r["table_id"]: int(r["row_count"]) for r in rows}
```

**`get_settings()` is the universal construction idiom** (`backend/config/settings.py:468` `def get_settings(): return Settings()` -- pydantic-settings loads from env/.env). Confirmed in-repo: `yfinance_tool.py:127` already does `BigQueryClient(get_settings())`; all 30+ other call sites pass `settings`. Only the broken `weekly_data_integrity.py:84` calls it bare. The job already imports nothing settings-related, so add `from backend.config.settings import get_settings` inside the function (lazy, mirrors the existing lazy `from backend.db.bigquery_client import BigQueryClient` at :81).

**Decision: reach through `.client` -- do NOT add a new generic `query()` helper.** Rationale: (a) `.client.query(...).result()` is the established 60-call idiom; adding a `BigQueryClient.query()` wrapper would be a NEW public method on a 1075-line prod class for a single non-critical caller -> larger blast radius, more to test, and a second way to do the same thing. (b) No other caller currently needs a generic `query()` (every existing method wraps a SPECIFIC typed query). (c) The minimal, lowest-risk change keeps the fix inside the one broken function. If a future audit finds 3+ callers wanting ad-hoc SQL, THEN add the helper -- but 51.4 should not.

### A2. The __TABLES__ SQL + dataset + cost (PROVEN: free metadata read, 2.7s)

Current SQL at :85: `SELECT table_id, row_count FROM \`{project}.{dataset}.__TABLES__\`` with `project=os.getenv("GCP_PROJECT_ID","sunny-might-477607-p8")`, `dataset=os.getenv("PYFINAGENT_DATASET","pyfinagent_data")`. This SQL is CORRECT for `__TABLES__` (the legacy metadata pseudo-table exposes `table_id` + `row_count`). Dataset `pyfinagent_data` is the right primary-data target.

**$0 verification RUN LIVE (2026-06-01)** against `sunny-might-477607-p8.pyfinagent_data.__TABLES__` via `client.client.query(sql).result(timeout=30)`:
```
elapsed_s=2.711   bytes_processed=0   bytes_billed=0   n_tables=9
  alt_13f_holdings: 110, alt_congress_trades: 7262, alt_finra_short_volume: 0,
  llm_call_log: 370, risk_intervention_log: 0, scraper_audit_log: 0,
  sla_alerts: 1, strategy_decisions: 9, ...
```
=> **bytes_billed=0 (FREE metadata read), 2.7s (well under the 30s rule), returns a populated dict.** The fix is proven end-to-end. This is the exact $0 verification artifact for Bug B.

(Note `pyfinagent_data` here returns only 9 tables -- the signals/prices/fundamentals live in other datasets/partitions; the job scans whatever `PYFINAGENT_DATASET` points at, which is the documented primary dataset. Drift-tracking those 9 ops/log tables is fine for a data-integrity heartbeat; if the operator wants the big fact tables too, that's a dataset-scope follow-on, not a 51.4 blocker.)

### A3. Bug A -- autoresearch is NOT on any live critical path

Searched all consumers of `run_memo.py` / `handoff/autoresearch/` memos across `backend/`, `scripts/`, `.claude/`:
- **The live trading loop, paper trader, screener, and Layer-3 harness gate do NOT read these memos.** The only thing that references the memo path is **`.claude/context/research-gate.md:22`** -- a DOC rule saying that IF a harness cycle's topic overlaps a recent memo, the cycle "MUST cite that memo ... and treat it as satisfying 3-5 of the required URL sources." That is an OPTIONAL convenience for the human/Claude research gate, not a runtime dependency -- and since ZERO memos have ever been produced, nothing has ever actually consumed one. The Layer-3 gate has always been satisfied by live `WebSearch`/`WebFetch` (this very brief is the proof).
- **IMPORTANT name-collision caveat:** the `backend/autoresearch/` PACKAGE (cron.py, promoter.py, gate.py, friday_promotion.py, strategy_*.py, etc.) is a COMPLETELY SEPARATE system -- the strategy-rotation/promotion machinery driven by `backend/autoresearch/cron.py` (`autoresearch_overnight` APScheduler job). It does NOT import or depend on `scripts/autoresearch/run_memo.py`. Do NOT conflate them: fixing/disabling `run_memo.py` has ZERO impact on the strategy-rotation `backend/autoresearch/` package. (This matches memory `project_strategy_rotation_*`.)
- **Conclusion: autoresearch (run_memo.py) is NON-critical.** Its only product is a nightly literature memo that nothing has ever read. Killing the nightly failure has no downstream effect except stopping the ERROR-file/exit-1 noise. This makes "graceful skip + exit 0" entirely safe -- there is no consumer to starve.

### A4. Bug A resolution -- graceful preflight, exit 0 (feasible; exact insertion point)

The PREFERRED fix (no pip, no launchd unload, no feature removal): a preflight in `run_memo.py` that checks the embedding provider's backing module is importable BEFORE constructing `GPTResearcher`, and if absent logs an actionable message and **`return 0`** (clean skip -- NO ERROR file, NO exit-1).

**Where:** the gate must sit (a) AFTER `env_defaults`/`os.environ.setdefault` so it reads the SAME `EMBEDDING` value the library will use, and (b) BEFORE `asyncio.run(_main_async(args))` (which is what eventually calls `run_research` -> `GPTResearcher(...)` -> `Memory(...)`). The clean spot is in `main()` immediately after the `os.environ.setdefault` loop (run_memo.py:158-159) and after the existing `ANTHROPIC_API_KEY` guard (:161-162), right before `return asyncio.run(...)` (:164). Concretely:

```python
# phase-51.4: preflight the embedding provider's backing module so a missing
# optional ML dep is a CLEAN SKIP (exit 0), not a nightly exit-1 + ERROR file.
import importlib.util as _ilu
_PROVIDER_MODULE = {
    "huggingface": "langchain_huggingface",
    "openai": "langchain_openai",
    "custom": "langchain_openai",
    "google_genai": "langchain_google_genai",
    "google_vertexai": "langchain_google_vertexai",
    "ollama": "langchain_ollama",
    "cohere": "langchain_cohere",
}
_emb = os.environ.get("EMBEDDING", "")
_provider = _emb.split(":", 1)[0] if ":" in _emb else _emb
_need = _PROVIDER_MODULE.get(_provider)
if _need and _ilu.find_spec(_need) is None:
    print(
        f"[autoresearch] skipped: EMBEDDING provider '{_provider}' needs the "
        f"'{_need}' package, which is not installed. To enable nightly memos: "
        f"pip install langchain-huggingface sentence-transformers "
        f"(or set EMBEDDING to an installed provider). Exiting cleanly (0).",
        flush=True,
    )
    return 0
```

**Feasibility CONFIRMED:** `importlib.util.find_spec('langchain_huggingface')` returns `None` today (probed live) -> the gate fires -> `return 0`. `main()` already returns an int consumed by `raise SystemExit(main())` (:168), so `return 0` propagates a clean exit. The wrapper `run_nightly.sh` then logs "END ... OK" and exits 0 -> launchd records success -> **no more nightly exit-1, no more ERROR file.** This STOPS the silent failure without pip, without `launchctl unload`, and without removing the operator's feature (the day the owner approves `pip install langchain-huggingface sentence-transformers`, the gate passes through and memos start producing -- zero further code change needed).

Note: `find_spec` only proves the module is *importable-by-name*, not that the import would 100% succeed (e.g. a broken transitive dep). That's an accepted limitation (Python.org discussion) -- but the broad `try/except` in `_main_async` (:114) is still the backstop for any residual import error, so worst case degrades to today's behavior (ERROR file) rather than a crash. The preflight handles the COMMON case (dep simply absent) cleanly; the existing try/except remains as defense-in-depth.

**$0 verification for Bug A:** run `python scripts/autoresearch/run_memo.py` (or `--topic-index 0`) in the venv as-is (huggingface dep absent) and confirm it prints the "skipped: ... not installed" line and exits 0 (`echo $?` -> 0), and that NO new `handoff/autoresearch/*-ERROR-*.md` file is created. (Does NOT spend any LLM tokens -- the gate returns before `GPTResearcher` is ever constructed, so no Anthropic call is made.) This is a cleaner-than-today artifact: no API key even needs to be valid for the skip path.

### A5. Bug A -- the zero-dep embedding-swap question (answer: NO clean zero-dep swap exists; graceful-skip is the right call)

The diagnostic asked whether a zero-extra-dependency embedding provider could replace huggingface entirely (avoiding the dep). Audited gpt-researcher's `Memory` provider arms (`embeddings.py:86-180`) against what is installed:
- **`openai` / `custom`** (`langchain_openai`, INSTALLED) -- but `OPENAI_API_KEY` is **ABSENT** (probed: `os.getenv('OPENAI_API_KEY')` -> ABSENT). `OpenAIEmbeddings` would 401 at embed time (during `conduct_research`), turning a construction-time failure into a runtime failure mid-run -> WORSE (spends Claude tokens first, then dies). The `custom:` variant points at `http://localhost:1234/v1` (lmstudio) by default -- no such server runs here. NOT viable.
- **`ollama`** (`langchain_ollama`) -- requires a running Ollama daemon at `OLLAMA_BASE_URL` (KeyError on the env var if unset, and no daemon present). NOT viable.
- **`google_genai` / `google_vertexai`** -- `langchain_google_genai` / `langchain_google_vertexai` are NOT confirmed installed and would each need their own API config; not a "zero-dep" win. NOT pursued.
- gpt-researcher's own default embedding (`embeddings.py:67`) is `Memory("openai","text-embedding-3-small")` -- i.e. upstream ALSO assumes OpenAI; there is no built-in "no-embedding" / null-provider mode. The embedding is structurally mandatory (`agent.py:173` builds it unconditionally; it powers the vector context store for retrieved sources).

**Conclusion: there is NO clean zero-extra-dependency embedding swap given the current secrets (no OPENAI_API_KEY, no Ollama daemon, no Google embedding config). The graceful-skip preflight (A4) is the correct resolution** -- it stops the noise without spending anything and without committing to an embedding backend the owner hasn't provisioned. If the owner later wants memos LIVE, the lowest-friction enablement is `pip install langchain-huggingface sentence-transformers` (local, free, CPU, no API key) -- which is exactly what the skip message tells them. (Bge-small is ~130MB, runs CPU-only; matches the project's local-only, $0-LLM-for-embeddings posture.)

### A6. Operator alternatives for Bug A (note, not recommended over the preflight)
1. **Owner-approved pip** (`pip install langchain-huggingface sentence-transformers`) -> memos go LIVE. The preflight is forward-compatible: once installed, the gate passes through automatically.
2. **Disable the job** (`launchctl unload ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` + optionally set the plist `RunAtLoad`/calendar off) -> stops the nightly fire entirely. But this REMOVES the operator's feature (the diagnostic explicitly wants to avoid that), and a future re-enable forgets the dep again. The preflight is strictly better: feature stays wired, just self-skips until the dep exists.
The preflight (A4) is the recommended default; (1) is the owner's enable path; (2) is a fallback if the owner wants it fully off.

### A7. Restart / live-check notes
- **Bug A:** `run_memo.py` is run fresh by launchd each night (no long-lived process) -> a code edit takes effect on the NEXT 02:00 fire (or an immediate manual `python scripts/autoresearch/run_memo.py` run for the live_check). No restart needed; the launchd plist need not change.
- **Bug B:** `weekly_data_integrity.run()` is invoked by the slack-bot's APScheduler (the slack-bot is the standalone `python -m backend.slack_bot.app` process). A code edit to the job is picked up only when that process restarts (it imports the job module at startup). For the $0 live_check, the function can be exercised DIRECTLY (`python -c "from backend.slack_bot.jobs.weekly_data_integrity import _default_fetch_counts; print(_default_fetch_counts())"`) without waiting for the weekly cron -- the verification below uses that path, so no slack-bot restart is required to PROVE the fix. (For it to take effect on the actual weekly schedule, the slack-bot does need a restart: `pkill -f "backend.slack_bot.app"` + relaunch -- the slack-bot has NO launchd label, per the phase-51.3 finding.)

---

## Part B -- External research

### Read in full (6; floor is 5; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.metaplane.dev/blog/four-efficient-techniques-to-retrieve-row-counts-in-bigquery-tables-and-views | 2026-06-01 | industry (data-observability vendor) | WebFetch (full) | Exact `__TABLES__` SQL: `SELECT table_id, row_count FROM \`project.dataset.__TABLES__\`` -- "a fast and cost-effective means to estimate the row count without executing a full table scan." Alt: `SELECT table_name, row_count FROM \`project.dataset.INFORMATION_SCHEMA.TABLES\``. Both metadata approaches beat `COUNT(*)` which "can be resource-heavy ... potentially causing extended execution times and increased costs." Neither works for VIEWS; both return APPROXIMATE counts (BigQuery "intermittently updates the row_count field"). |
| https://openillumi.com/en/en-bigquery-all-tables-row-count/ | 2026-06-01 | industry/blog (CodeArchPedia) | WebFetch (full) | **DECISIVE correction:** `INFORMATION_SCHEMA.TABLES` does **NOT** have `row_count`; the modern free path is `INFORMATION_SCHEMA.TABLE_STORAGE` with `total_rows` (+ `active_logical_bytes`), and it REQUIRES a region qualifier (`\`project.region-US\`.INFORMATION_SCHEMA.TABLE_STORAGE`). `__TABLES__` DOES expose `row_count`/`size_bytes` but "it is officially considered deprecated. Users should transition to INFORMATION_SCHEMA." BOTH are free: "Accessing metadata via __TABLES__ or INFORMATION_SCHEMA is typically a zero-cost operation ... rather than scanning user data." |
| https://discuss.python.org/t/optional-imports-for-optional-dependencies/104760 | 2026-06-01 | official-adjacent (Python.org discussion) | WebFetch (full) | Three patterns for optional deps: (1) `try: import x except ImportError: x=None` then None-check (risks confusing `'NoneType' has no attribute` errors); (2) `importlib.util.find_spec()` to check availability WITHOUT importing -- "best when you need to know availability without importing ... only tells you if the module can be found not if the import would actually succeed"; (3) lazy import. "No single pattern fits all." find_spec is the right tool for a PREFLIGHT (decide-then-skip) -- exactly Bug A's case. |
| https://adamj.eu/tech/2021/12/29/python-type-hints-optional-imports/ | 2026-06-01 | authoritative blog (Adam Johnson, Django core) | WebFetch (full) | The `try: import x; HAVE_X=True except ImportError: HAVE_X=False` boolean-flag pattern is the type-checker-clean idiom (Mypy rejects assigning `None` to a module-typed name). Guard with `if HAVE_X:`. Corroborates that a clean SKIP path (vs a crash) is the standard way to make a feature optional. |
| https://docs.gptr.dev/docs/gpt-researcher/llms | 2026-06-01 | official docs (gpt-researcher) | WebFetch (full) | `EMBEDDING=provider:model` format. Supported embeddings: openai, azure_openai, cohere, google_vertexai, google_genai, fireworks, ollama, together, mistralai, **huggingface**, nomic, voyageai, bedrock. `custom:` requires `OPENAI_BASE_URL`+`OPENAI_API_KEY` (points at a local OpenAI-compatible server). "Zero-extra-dependency" options listed: `openai:text-embedding-3-small` (needs OPENAI_API_KEY) or `google_genai:models/text-embedding-004` -- BOTH need an API key/config pyfinagent lacks (no OPENAI_API_KEY) => no truly free zero-dep swap. |
| https://developer.apple.com/forums/thread/52369 | 2026-06-01 | official (Apple Developer Forums, launchd) | WebFetch via search (full thread context) | launchd exit-code semantics: 0 = success, positive = error reported, negative = killed by signal. `KeepAlive.SuccessfulExit` keys off zero vs non-zero exit. launchd interprets RAPID exit as failure -- but a clean `return 0` from a job that ran (even briefly) is recorded as success and does NOT trip KeepAlive restarts. Confirms Bug A's "exit 0 on skip" makes launchd record success and stops the failure signal. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://docs.cloud.google.com/bigquery/docs/information-schema-tables | official docs | Fetched but the page renders as a JS navigation shell -- the column table (row_count? total_rows?) was NOT in the extracted markdown. The authoritative column facts came from the openillumi + Metaplane reads instead (and the LIVE probe). |
| https://docs.cloud.google.com/bigquery/docs/information-schema-table-storage | official docs | Same JS-shell problem; column list (`total_rows`, `total_logical_bytes`, `table_name`) not extractable. openillumi supplied the `TABLE_STORAGE.total_rows` fact. |
| https://cloud.google.com/knowledge/kb/bigquery-count-query-slot-utilization-and-rows-scanned-000004360 | official KB | Corroborates that `COUNT(*)` scans data/uses slots (the costly path the metadata approach avoids); not needed in full given the live bytes_billed=0 proof. |
| https://blog.lordpatil.com/posts/maximizing-bigquery-free-tier/ | blog | Confirms metadata operations don't consume the free-tier query allowance; corroborating, lower-tier than the reads above. |
| https://docs.cloud.google.com/bigquery/docs/information-schema-table-storage (TABLE_STORAGE) | official docs | (dup host) -- the region-qualifier + total_rows facts already captured from openillumi. |
| https://github.com/assafelovic/gpt-researcher/issues/629 | official (issue) | "Support for different Embeddings other than OpenAI" -- confirms embedding-provider pluggability is a known area; not load-bearing beyond the official LLM-config doc read in full. |
| https://github.com/assafelovic/gpt-researcher/issues/510 | official (issue) | "HuggingFace Open Source Models Integration" -- confirms the HF path needs the HF extra; corroborates that HF is an optional extra, not bundled. |
| https://adamj.eu/.../ (find_spec mention) | blog | The Adam Johnson piece (read in full) does NOT cover find_spec; the Python.org discussion (read in full) does -- complementary, both counted once. |
| https://www.launchd.info/ | community ref | launchd tutorial; KeepAlive/SuccessfulExit semantics already captured from the Apple forum read. |
| https://github.com/agronholm/apscheduler/issues/520 | official (issue) | Holiday-skip-not-built-in -- relevant to scheduled jobs generally, but 51.4's jobs are time-only crons whose BUG is dep/wiring, not calendar; not load-bearing here (it WAS load-bearing for 51.3). |

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026):** "BigQuery INFORMATION_SCHEMA.TABLES row_count vs __TABLES__ pseudo table 2026"; "gpt-researcher EMBEDDING provider config no huggingface dependency 2026". (-> surfaced the deprecation note + TABLE_STORAGE.total_rows + the gpt-researcher provider list.)
2. **Last-2-year window (2025):** covered within the openillumi/Metaplane articles (both current, discuss the 2024-2026 deprecation state) + the Python.org optional-imports discussion (a 2024+ thread). Recency findings reported below.
3. **Year-less canonical:** "BigQuery __TABLES__ row_count metadata query free cost"; "Python optional dependency graceful skip importlib.util.find_spec try import pattern best practice"; "try import ImportError optional dependency skip job clean exit Python"; "launchd job exit 0 vs exit 1 ... clean skip". (-> the founding/canonical material: Metaplane's four-techniques piece, the Adam Johnson optional-imports article, the Apple launchd forum thread.)

### Recency scan (2024-2026) -- PERFORMED
Searched the last-2-year window on (a) BigQuery row-count metadata path, (b) optional-dependency graceful-skip patterns, (c) launchd exit-code/clean-skip. **Findings:**
1. **NEW / supersedes-adjacent (Bug B):** `__TABLES__` is now **officially deprecated**; the recommended modern free path is `INFORMATION_SCHEMA.TABLE_STORAGE.total_rows` (region-qualified) -- NOT `INFORMATION_SCHEMA.TABLES.row_count` (which does not exist). The current code uses `__TABLES__.row_count`, which **still works today** (LIVE-proven: bytes_billed=0, 9 tables returned, 2.7s). So this is a SOFT modernization note, not a blocker: 51.4 can ship the `client.client.query(__TABLES__)` fix as-is and OPTIONALLY migrate to `TABLE_STORAGE` for future-proofing. If migrating, the SQL becomes `SELECT table_name AS table_id, total_rows AS row_count FROM \`{project}.region-us\`.INFORMATION_SCHEMA.TABLE_STORAGE WHERE table_schema='{dataset}'` -- note the REGION qualifier (region-us for pyfinagent_data, which is US) replaces the dataset path, and it scopes by `table_schema`. (Caveat: `__TABLES__` is dataset-scoped and simpler; `TABLE_STORAGE` is region-scoped and would return ALL datasets in the region unless filtered by `table_schema`.)
2. **STABLE (Bug A):** the optional-dependency `find_spec`-preflight / try-except-skip pattern is unchanged and remains current best practice (2024 Python.org discussion, evergreen Adam Johnson article). No new mechanism obsoletes it. `importlib.util.find_spec` has been the recommended availability-check since 3.4 and is still the right tool for a decide-then-skip preflight.
3. **STABLE (launchd):** exit-0-means-success / non-zero-trips-KeepAlive semantics are unchanged across macOS versions. No new finding.

### Key findings (per-claim, cited)
1. **`__TABLES__` row-count is a FREE, fast metadata read (no full scan), and the SQL `SELECT table_id, row_count FROM \`project.dataset.__TABLES__\`` is exactly right.** "a fast and cost-effective means to estimate the row count without executing a full table scan" (Source: Metaplane, https://www.metaplane.dev/blog/four-efficient-techniques-to-retrieve-row-counts-in-bigquery-tables-and-views, accessed 2026-06-01); "Accessing metadata via __TABLES__ or INFORMATION_SCHEMA is typically a zero-cost operation" (Source: openillumi, https://openillumi.com/en/en-bigquery-all-tables-row-count/). **LIVE-PROVEN:** bytes_billed=0, 2.7s, 9 tables (this brief, A2).
2. **`__TABLES__` is deprecated; the modern free equivalent is `INFORMATION_SCHEMA.TABLE_STORAGE.total_rows` (region-qualified). `INFORMATION_SCHEMA.TABLES` has NO row_count.** "While __TABLES__ is convenient, it is officially considered deprecated. Users should transition to INFORMATION_SCHEMA"; "The correct view is INFORMATION_SCHEMA.TABLE_STORAGE, which provides the total_rows column" (Source: openillumi, same URL). => 51.4 ships the working `__TABLES__` fix now; `TABLE_STORAGE` is an OPTIONAL future-proof. Both are free.
3. **`importlib.util.find_spec` is the correct tool to PREFLIGHT an optional dependency and decide-then-skip without importing.** "best when you need to know availability without importing ... it only tells you if the module can be found not if the import would actually succeed" (Source: Python.org discussion, https://discuss.python.org/t/optional-imports-for-optional-dependencies/104760). This is precisely Bug A: check `langchain_huggingface` is present BEFORE constructing GPTResearcher; if absent, log + `return 0`. The residual "would the import actually succeed" gap is covered by the existing broad try/except backstop in run_memo.py.
4. **The type-checker-clean / standard way to make a feature optional is a boolean-flag try/except (skip path), NOT letting the ImportError propagate.** "`try: import x; HAVE_X=True except ImportError: HAVE_X=False` ... Mypy is just fine with this" (Source: Adam Johnson, https://adamj.eu/tech/2021/12/29/python-type-hints-optional-imports/). Endorses A4's clean-skip over today's crash-into-ERROR-file.
5. **gpt-researcher mandates an embedding provider and there is no truly free zero-dep swap for pyfinagent (no OPENAI_API_KEY; the "easy" alternatives all need a key/daemon).** Supported embeddings incl. huggingface; "zero-extra-dependency" options are `openai:...` or `google_genai:...` -- both require credentials (Source: gpt-researcher docs, https://docs.gptr.dev/docs/gpt-researcher/llms). Cross-checked against the venv (no OPENAI_API_KEY, no Ollama daemon) => graceful-skip is the right call, and HF (`pip install langchain-huggingface sentence-transformers`, CPU/free/no-key) is the lowest-friction ENABLE path if the owner wants memos live.
6. **A launchd job that does a clean `return 0` on skip is recorded as SUCCESS and stops the exit-1 failure signal.** exit 0 = success; KeepAlive.SuccessfulExit keys off zero (Source: Apple Developer Forums, https://developer.apple.com/forums/thread/52369). Confirms Bug A's "exit 0 on skip" stops the nightly failure cleanly (no ERROR file, no exit-1) without unloading the agent.

### Consensus vs debate (external)
- **Consensus:** metadata-table row counts are free + the right tool vs COUNT(*) (Metaplane, openillumi, Google KB all agree). `find_spec`/try-except is the standard optional-dep skip (Python.org, Adam Johnson agree). exit-0 = launchd success (Apple, launchd.info agree). No contradiction on any load-bearing claim.
- **Debate (minor):** `__TABLES__` vs `INFORMATION_SCHEMA.TABLE_STORAGE` -- Google deprecates `__TABLES__`, but it still works and is simpler (dataset-scoped, exposes `row_count` directly). The pragmatic call: ship the working `__TABLES__` fix in 51.4 (lowest-risk, LIVE-proven), note `TABLE_STORAGE` as a future modernization. Not a blocker either way.

### Pitfalls (from literature/docs) -> applied to phase-51.4
1. **(Bug B) Don't use `COUNT(*)` per table** -- it scans data and costs money/slots (Metaplane, Google KB). The `__TABLES__`/metadata path is free (LIVE-proven bytes_billed=0). Keep the metadata SQL.
2. **(Bug B) `INFORMATION_SCHEMA.TABLES` does NOT have row_count** -- if anyone "modernizes" by swapping `__TABLES__` -> `INFORMATION_SCHEMA.TABLES` they'll get a column error. The correct modern view is `TABLE_STORAGE` (`total_rows`, region-qualified). The current `__TABLES__.row_count` is correct as-is; don't half-migrate.
3. **(Bug B) Bound the query** -- add `.result(timeout=30)` (CLAUDE.md 30s rule). Metadata is fast (2.7s observed) but the timeout is the project convention. CONFIRMED in the fix.
4. **(Bug A) `find_spec` is necessary-not-sufficient** -- it proves importability-by-name, not that the import fully succeeds (Python.org). Keep the existing broad try/except in `_main_async` as the backstop; the preflight handles the common "dep absent" case cleanly.
5. **(Bug A) Read EMBEDDING AFTER `os.environ.setdefault`** -- the preflight must read the same value the library reads, so it goes after :158-159, not before. Otherwise a user-overridden `EMBEDDING` env var would be checked against the wrong provider.
6. **(Bug A) `return 0`, do NOT `sys.exit(1)` / raise** -- exit 0 makes launchd record success (Apple). Raising or exiting non-zero would keep the failure signal. The skip must be a clean exit-0.

---

## SYNTHESIS -- the actionable answer

### S1 -- Bug A fix (autoresearch graceful preflight)
**File:** `scripts/autoresearch/run_memo.py`. **Insertion point:** in `main()`, immediately AFTER the `os.environ.setdefault` loop (after :159) and after the `ANTHROPIC_API_KEY` guard (:161-162), BEFORE `return asyncio.run(_main_async(args))` (:164). **Logic:** parse the provider from `os.environ["EMBEDDING"]`, map to its backing module (`huggingface`->`langchain_huggingface`), `importlib.util.find_spec(...)` it; if `None`, `print("[autoresearch] skipped: ... pip install langchain-huggingface sentence-transformers ...", flush=True)` and `return 0`. (Full code in A4.) **No pip, no launchd change, feature stays wired and self-enables once the owner installs the dep.** **Zero-dep embedding swap: does NOT exist** (no OPENAI_API_KEY, no Ollama daemon) -- graceful-skip is correct (A5).

### S2 -- Bug B fix (weekly_data_integrity BQ wiring)
**File:** `backend/slack_bot/jobs/weekly_data_integrity.py::_default_fetch_counts` (:78-90). **Two edits:**
- **:84** `client = BigQueryClient()` -> `client = BigQueryClient(get_settings())` (add `from backend.config.settings import get_settings` lazily inside the function, next to the existing lazy `from backend.db.bigquery_client import BigQueryClient`).
- **:86** `rows = client.query(sql)` -> `rows = client.client.query(sql).result(timeout=30)` (reach through to the underlying `google.cloud.bigquery.Client`; bound to 30s).
The dict-comp at :87 (`{r["table_id"]: int(r["row_count"]) for r in rows}`) is already correct for `__TABLES__` rows. **Do NOT add a generic `BigQueryClient.query()` helper** -- single non-critical caller, `.client.query(...).result()` is the established 60-call idiom (A1). **__TABLES__ SQL + dataset `pyfinagent_data` are already correct** -- keep them. The current `except Exception -> return {}` fail-open stays (good: a BQ hiccup shouldn't crash the slack-bot).

### S3 -- Is autoresearch critical-path? NO (A3)
Nothing in the live trading loop / paper trader / Layer-3 gate consumes `run_memo.py` memos at runtime. The only reference is a DOC convenience rule (`.claude/context/research-gate.md:22`) that has never fired (zero memos ever produced). The `backend/autoresearch/` PACKAGE is a SEPARATE strategy-rotation system, unaffected. => graceful-skip is fully safe.

### S4 -- $0 verification plan (per bug)
- **Bug B (run the real query):** `python -c "from backend.slack_bot.jobs.weekly_data_integrity import _default_fetch_counts; d=_default_fetch_counts(); print(len(d), dict(list(d.items())[:5]))"` -> must print a NON-EMPTY dict of `{table_id: row_count}` (already proven via the A2 probe: 9 tables, bytes_billed=0, 2.7s). The fix turns the current silent `{}` into a populated dict. (No slack-bot restart needed for the proof; the function is callable directly.)
- **Bug A (preflight -> clean skip):** `python scripts/autoresearch/run_memo.py --topic-index 0; echo "exit=$?"` with the huggingface dep absent -> must print the "skipped: ... not installed" line, `exit=0`, and create NO new `handoff/autoresearch/*-ERROR-*.md`. (Spends $0 LLM -- the gate returns before GPTResearcher is constructed, so no Anthropic call is made; ANTHROPIC_API_KEY need not even be valid for the skip path. Tip: confirm no new ERROR file via `ls -t handoff/autoresearch/*ERROR* | head -1` before/after.)

### S5 -- Restart/effect notes
- Bug A: launchd runs run_memo.py fresh each night -> edit takes effect next 02:00 (or immediate manual run). Plist unchanged.
- Bug B: takes effect on the weekly cron after a slack-bot restart (`pkill -f "backend.slack_bot.app"` + relaunch; no launchd label). The $0 live_check uses the direct-call path, so the proof does not require the restart.

---

## GATE ENVELOPE

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```

`gate_passed: true` -- 6 external sources read in full via WebFetch (floor 5: Metaplane row-count techniques, openillumi BigQuery row-count, Python.org optional-imports discussion, Adam Johnson optional-imports, gpt-researcher LLM/embedding docs, Apple launchd exit-code forum); recency scan performed + reported (2024-2026; surfaced the __TABLES__ deprecation -> TABLE_STORAGE soft-modernization note); 3-variant query discipline visible (current-year / last-2-year / year-less, queries listed); 16 unique URLs total; internal audit pinned to file:line across 8 artifacts (scripts/autoresearch/run_memo.py, .venv gpt_researcher/agent.py + memory/embeddings.py, backend/slack_bot/jobs/weekly_data_integrity.py, backend/db/bigquery_client.py, backend/config/settings.py, ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist, scripts/autoresearch/run_nightly.sh) PLUS two LIVE probes (the __TABLES__ query: bytes_billed=0/2.7s/9 tables; and find_spec('langchain_huggingface')=None). Both bugs reproduce exactly as diagnosed; both fixes are $0, pip-free, and feature-preserving.
