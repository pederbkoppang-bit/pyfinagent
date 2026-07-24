# Research Brief -- Step 75.9 (BigQuery fail-closed dedup, parameterization, 30s-timeout sweep, cost guard)

**Tier:** moderate | **Audit-class:** false (denominators enumerated in step text; job is to VERIFY + flag drift)
**Researcher spawn:** 2026-07-23 | **Executor target:** sonnet-4.6/high
**BOUNDARY:** paper-only, no schema/table changes, historical_macro frozen

> WRITE-FIRST skeleton created at spawn; sections filled incrementally as sources are read.

## Status: COMPLETE -- gate_passed: true (6 sources read in full, recency scan done, not audit-class)

---

## 0. DRIFT SUMMARY (executor must apply these corrections)

The 2026-07-19 anchors have drifted. Measured 2026-07-23:

| # | Step-text claim | MEASURED reality | Action |
|---|-----------------|------------------|--------|
| D1 | "12 migration files" (gap6-09) | **13** distinct files, **20** untimed `.result()` call sites | Use the 13-file worklist in §2 |
| D2 | `backend/backtest/sortino.py:114` | file is at **`backend/metrics/sortino.py:114`** | corrected path |
| D3 | `backend/signals/pead_signal.py:342` | file is at **`backend/services/pead_signal.py:342`** | corrected path |
| D4 | `backend/tools/sector_calendars.py:200` | file is at **`backend/services/sector_calendars.py:200`** | corrected path |
| D5 | `backend/services/cost_budget_watcher.py:104` | file is at `backend/slack_bot/jobs/cost_budget_watcher.py` and has **NO `.result()`/`.query()` at all** -- PHANTOM site | drop from worklist; note in test that this site is absent |
| D6 | `backend/autoresearch/harness_autoresearch.py:196` | file is at **`backend/api/harness_autoresearch.py:196`** | corrected path |
| D7 | `monthly_approval_api.py:141` | untimed `.result()` is at **:143** (`job.result()`) | corrected line |
| D8 | "34 construction sites" (perf-11) | **34 is correct for `BigQueryClient(` in `backend/api/`+`backend/services/` only**; repo-wide non-test = **45** `BigQueryClient(` / **49** `bigquery.Client(` | keep 34 as the migrate-at-least scope; state full denominator |
| D9 | dedup fail-open named only prices+fundamentals | `_get_existing_macro` (data_ingestion.py:271-278) has the **identical** fail-open pattern but `historical_macro` is FROZEN | do NOT fix macro in 75.9; queue a separate step |

---

## 1. Internal Audit (load-bearing) -- re-anchored, verbatim file:line

### (a) data-bq-01 -- fail-open dedup (data_ingestion.py)
- `_get_existing_price_dates` def :78-92; the swallow is **:91-92** `except Exception:` / `return set()`. Dedup consumed at :117 `existing = self._get_existing_price_dates(batch)`; gate at :142 `if (ticker, date_str) in existing: continue`; insert at :167 `errors = self.client.insert_rows_json(table, sub)`.
- `_get_existing_fundamentals` def :178-190; swallow **:189-190**. Consumed :198-200 (loop building `existing`), gate :222, insert :257.
- `_get_existing_macro` :271-278 swallow :277-278 -- SAME BUG, but FROZEN table (D9): out of scope.
- NEITHER dedup call is wrapped in a try/except inside `ingest_prices`/`ingest_fundamentals`, so re-raising propagates straight out BEFORE any `insert_rows_json` -- satisfies criterion 1 ("ZERO insert_rows_json calls and surfaces the error").

### (b) data-bq-02 -- f-string SQL
- `get_agent_memories` bigquery_client.py **:496-512**; injection surface **:500** `where = f"WHERE agent_type = '{agent_type}'"` and **:506** `LIMIT {int(limit)}`. Already wrapped in try/except -> `[]` (fail-open OK for memories; non-money).
- **Template to mirror** (same file): `get_recent_reports` **:278-282** -- `bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("limit","INT64",limit)])`. (Note its `.result()` at :281 is itself UNTIMED -- fix under (c).)
- data_ingestion ticker list: **:82** (prices) + **:180** (fundamentals) `ticker_list = ", ".join(f"'{t}'" for t in tickers[:100])` interpolated into `WHERE ticker IN ({ticker_list})`. Comment at :81 falsely says "Use parameterized IN clause" -- it is NOT parameterized.
- **Template to mirror** = cache.py **:100-113** (§3): `ArrayQueryParameter("tickers","STRING",tickers)` + `WHERE ticker IN UNNEST(@tickers)`.

### (d) data-bq-06 -- no maximum_bytes_billed anywhere
- `grep maximum_bytes_billed backend/ scripts/` = **0 hits** (confirmed 2026-07-23). bigquery_client.py builds **24 separate `bigquery.QueryJobConfig(...)`** instances (lines 278,325,339,349,371,452,526,554,570,587,631,637,655,675,703,764,795,848,899,946,961,981,1036,1045). No shared factory exists -> factory must be CREATED. Criterion only requires factory (default 5 GiB) + >=1 call-path adoption.

### (e) py-core-03 -- skill_optimizer bare except
- **:172-176** = LOGGED sibling template (`logger.warning(...); return []`). **:180-189** = the bug: outcome query at :185 untimed `.result()`, **:188 `except Exception:` / :189 `pass`** (log-free) -> agents scored against empty outcomes. Also first query :173 untimed. Fix: log warning + `timeout=30` on :173 & :185; replace bare pass with warning (mark run degraded).

### (f) perf-11 -- per-request BigQueryClient construction
- `get_bq_client()` does NOT exist yet; `@lru_cache` absent from bigquery_client.py. **Template**: settings.py **:612-613** `@lru_cache()` / `def get_settings()`. `BigQueryClient.__init__(self, settings)` (:22-35) re-parses `json.loads(gcp_credentials_json)` + builds a fresh `bigquery.Client` every call.
- Hot spots CONFIRMED: `api/paper_trading.py` builds `BigQueryClient(settings)` **8x** inline (:82,:126,:181,:263,:287,:303,:441,:472); `api/performance_api.py:59`; `api/reports.py:24-25` `_get_bq` returns `BigQueryClient(settings)` per-request via `Depends`.
- **DESIGN NOTE (thread-safety + hashability)**: make `get_bq_client()` ZERO-arg (`@lru_cache; def get_bq_client(): return BigQueryClient(get_settings())`) mirroring `get_settings`. Do NOT `@lru_cache` on a function taking `settings` -- Pydantic `Settings` is not reliably hashable and would raise/leak. Singleton is shared across FastAPI threads -> external §3 confirms `google.cloud.bigquery.Client` is safe to share read-side across threads.

## 2. .result() timeout worklist (MEASURED per file, 2026-07-23)

### bigquery_client.py -- 25 `.result(` total, **18 UNTIMED** (only 7 timed):
Untimed BQ lines: **281, 295, 352, 374, 455, 470, 508, 540, 571, 580, 590, 632, 676, 704, 986, 1037, 1048, 1069**.
Money-path DML: **:540** (`_run_dml_with_retry`, all paper_positions/paper_trades MERGEs -> timeout=60), **:571** (`upsert_paper_portfolio` INSERT -> 60), **:580** (`get_paper_positions` -> 30). Compliant template in-file: **:533** `.result(timeout=30)`.
(The step's "all of bigquery_client.py incl :571/:580" is an UNDERCOUNT: 18 sites, not 3.)

### 13 external files (paths corrected):
| File:line (CORRECTED) | Untimed call | timeout to add |
|---|---|---|
| paper_trader.py:1245 | `...result()` | 30 |
| services/cycle_health.py:474 | `list(bq.client.query(sql).result())` | 30 |
| **metrics**/sortino.py:114 | `list(client.query(sql).result())` | 30 |
| api/paper_trading.py:1127 | `list(bq.client.query(...).result())` | 30 (note :1164 is `future.result()` -- ThreadPool, NOT BQ, leave) |
| api/performance_api.py:82 | `list(bq.client.query(sql).result())` | 30 |
| **services**/pead_signal.py:342 | `list(bq.client.query(query).result())` | 30 |
| **services**/sector_calendars.py:200 | `list(bq.client.query(query).result())` | 30 |
| agents/skill_optimizer.py:173 & :185 | 2 untimed | 30 each |
| ~~cost_budget_watcher.py:104~~ | PHANTOM (D5) | drop |
| autoresearch/slot_accounting.py:139 | `list(client.query(sql,job_config=cfg).result())` | 30 + module client (gap3-08) |
| **api**/harness_autoresearch.py:196 | `list(client.query(sql,job_config=cfg).result())` | 30 |
| api/monthly_approval_api.py:**143** | `job.result()` | 30 |

### 13 migration files (D1 -- the real denominator, 20 call sites), timeout=60:
add_efficiency_snapshots.py:96 | add_external_flow_today_column.py:70,77 | add_round_trip_schema.py:36 | add_session_budget_to_llm_call_log.py:85 | add_ticker_to_llm_call_log.py:81 | create_alpha_velocity_table.py:81,96 | create_data_source_events_table.py:93 | create_directive_versions_table.py:77,92 | create_historical_fx_rates_table.py:64 | create_options_snapshots_table.py:104 | create_promoted_strategies_table.py:96 | create_strategy_deployments_view.py:114,116,118,143 | phase_32_1_add_stop_advanced_at_R.py:71,78.
Compliant convention: `add_llm_call_log.py:70 job.result(timeout=60)`. (These are DDL scripts; adding `timeout=` does NOT change schema -- boundary-safe. Executor must NOT re-run them.)

### gap3-08 -- slot_accounting fresh client per call:
`_default_bq_query_count` builds `bigquery.Client(project=project)` at **:134** per call (untimed `.result()` :139); `_default_bq_insert` builds a fresh client at **:118**. Fix: module-level client + timeout=30. Both already fail-open (try/except -> 0/False).

## 5. Vacuous-pass traps + MANDATORY mutation matrix

The 6 criteria are mostly SOURCE/AST SCANS -- high vacuous-pass risk (per `feedback_measure_dont_assert_claims` + `feedback_mutation_test_guards_and_fixtures`). Only criterion 1 is behavioral.

| Crit | Vacuous-pass trap | Guard the test MUST have |
|---|---|---|
| 1 (dedup) | **Fixture cannot represent failure**: if the yf.download mock returns an EMPTY frame, "zero inserts" is vacuously true even fail-OPEN (no rows to insert). | Mock `yf.download` to return a NON-EMPTY OHLCV frame so fail-open WOULD call `insert_rows_json`; mock `client.query.side_effect=RuntimeError`; assert BOTH `pytest.raises(RuntimeError)` (error surfaces) AND `client.insert_rows_json.assert_not_called()`. Template: `test_64_3_learnings_reader.py:21-48` + `test_strategy_decisions_heartbeat.py:46-61`. |
| 2 (params) | `assert "ScalarQueryParameter" in src` passes even if the f-string ALSO stays (dead param). | ALSO assert ABSENCE: no `f"WHERE agent_type = '"` in bigquery_client src; no `IN ({ticker_list})` / no `", ".join(f"'{t}'"` in data_ingestion src. Symmetric present+absent. |
| 3 (timeout sweep) | (i) regex too loose passes on `.result()`; (ii) **`all([])` trap** -- empty file list = vacuous True; (iii) scan silently SKIPS a drifted/missing path (D2-D6) -> the very sites at risk hide. | Use AST (parse, find every `.result` Call, require a `timeout` kw). Assert the enumerated file list is NON-EMPTY and every expected path EXISTS (hard-fail on missing, so drift can't hide). Exclude `future.result()` (ThreadPool) explicitly (paper_trading.py:1164). |
| 4 (factory) | `assert factory_default == 5*1024**3` alone is a constant/tautology assertion. | ADD behavioral adoption: build a job_config via the factory, assert `.maximum_bytes_billed == 5 GiB`, AND mock `client.query` on >=1 real method and assert the passed `job_config.maximum_bytes_billed` is set. |
| 5 (skill_opt/slot) | source scan `assert "logger.warning" in src` passes even if bare `pass` remains elsewhere. | Assert NO bare `except Exception:\n    pass` remains in the outcomes block (AST: except handler body is not a lone `Pass`); assert slot_accounting module-level client is reused (identity across 2 calls) + timeout present. |
| 6 (lru singleton) | "imported by" is an import-string scan -- `from x import get_bq_client` unused still passes. | Behavioral: `assert get_bq_client() is get_bq_client()` (identity). For the 3 api files assert they actually CALL it (not just import); ideally assert the inline `BigQueryClient(settings)` hot-spots are gone. |

**Mutation matrix the executor MUST run (each must FLIP the test red):**
1. Revert dedup to `return set()` -> crit-1 red (insert called / no raise).
2. Blank the yf.download fixture -> prove crit-1 goes vacuously GREEN (then restore non-empty; this proves the fixture can represent the failure -- mutate the STUB).
3. Re-insert the f-string in get_agent_memories AND data_ingestion -> crit-2 red.
4. Delete ONE `timeout=` from (a) bigquery_client.py, (b) one external file, (c) one migration -> crit-3 red x3.
5. Point the crit-3 scan at a renamed/missing path -> must ERROR (denominator guard), not skip-green.
6. Set factory default to 4 GiB and separately remove the adoption -> crit-4 red x2.
7. Restore `except: pass` in skill_optimizer -> crit-5 red.
8. Make `get_bq_client` return a fresh instance each call -> crit-6 red.

## 6. BOUNDARY risk analysis -- fail-closed dedup is SAFE on every caller

**Empty-result vs query-exception (the core distinction):**
- Successful query returning 0 rows (empty/first-run table): `{(r[..],r[..]) for r in rows}` = `set()` -> gate `in existing` is always False -> **insert all rows (legitimate)**. This path is UNCHANGED by the fix.
- Query raises (timeout / permission / transient): currently `except: return set()` -> **also inserts all rows -> DUPLICATE (ticker,date) bars**. This is the ONLY path the fix changes: log + re-raise instead.
- First-run safety: `run_full_ingestion` calls `_ensure_tables_exist()` (:352) BEFORE `ingest_prices` (:354), so the dedup table exists -> the query SUCCEEDS-empty, never throws "table not found". Fail-closed does NOT break cold-start ingestion.

**Every caller of ingest_prices/ingest_fundamentals/run_full_ingestion and abort behavior:**
| Caller | Context | Abort effect | Verdict |
|---|---|---|---|
| `slack_bot/jobs/daily_price_refresh.py:82` (`run_production`) | AUTONOMOUS scheduler (only per-cycle caller) | caught by fail-open try/except **:86-87** -> logs, returns `{written:0}`, scheduler continues; idempotent by-day heartbeat retries next run | SAFE / self-heals |
| `backtest_engine.py:1303` (`_auto_ingest_if_needed`) | only fires when `prices_count==0` (cold start -> query succeeds-empty, fail-closed inert); any throw caught at **:1307** "non-fatal" | SAFE |
| `api/backtest.py:201` | manual `POST /ingest` via `asyncio.to_thread`, wrapped -> `HTTPException(500)` at :208-210 | SAFE (user sees error; correct fail-closed) |
| `scripts/migrations/extend_historical_data.py:94` | manual backfill script | fails loudly | SAFE (correct) |

**Conclusion:** NO caller relies on the fail-open `set()` for correct operation; it only ever produced silent duplicate inserts on error. Fail-closed is strictly safer everywhere and the sole autonomous caller already contains the exception. Paper-only boundary honored (no schema/table change; dedup logic only).

## 3. External research

### Read in full (>=5 required; counts toward gate)
| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 1 | https://docs.cloud.google.com/bigquery/docs/parameterized-queries | 2026-07-23 | Official doc | "you can use parameters to protect queries made from user input against SQL injection." + "Parameters cannot be used as substitutes for identifiers, column names, table names, or other parts of the query." Both WHERE values AND LIMIT are parameterizable; `ScalarQueryParameter`/`ArrayQueryParameter` + `... IN UNNEST(@states)`. |
| 2 | https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.job.QueryJob | 2026-07-23 | Official doc | `result(timeout=)` docstring: "The number of seconds to wait for the underlying HTTP transport before using `retry`. If `None`, wait indefinitely... If unset... we still wait indefinitely for the job to finish." Raises `concurrent.futures.TimeoutError`. Bounds the CLIENT-SIDE HTTP wait, does NOT cancel the BQ job. |
| 3 | https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.job.QueryJobConfig | 2026-07-23 | Official doc | `maximum_bytes_billed`: "Maximum bytes to be billed for this job or None if not set." (per-job, default None). `job_timeout_ms`: "If this time limit is exceeded, BigQuery might attempt to stop the job." (only THIS bounds actual job execution). |
| 4 | https://googleapis.github.io/google-api-python-client/docs/thread_safety.html | 2026-07-23 | Official doc | "The httplib2.Http() objects are not thread-safe." Applies to the OLDER api-python-client (httplib2). NOTE: `google-cloud-bigquery` uses the google-auth/`requests` transport, NOT httplib2 -- so this is context, not the definitive bigquery.Client answer. |
| 5 | https://github.com/googleapis/python-bigquery/issues/1922 | 2026-07-23 | GitHub (maintainer repo) | `QueryJob.result()` hung >6 DAYS after the job actually completed in 2 min (google-cloud-bigquery 3.21.0). Non-deterministic; reporter used PROCESS-level timeout as workaround. **Direct evidence the untimed-`.result()` indefinite-hang risk in data-bq-03 is real and current.** |
| 6 | https://docs.cloud.google.com/bigquery/docs/best-practices-costs | 2026-07-23 | Official doc | maximum_bytes_billed: "the number of bytes that the query reads is estimated BEFORE the query execution. If the number of estimated bytes is beyond the limit, then the query fails without incurring a charge." Example: "Error: Query exceeded limit for bytes billed: 1000000. 10485760 or higher required." Caveat: on CLUSTERED tables the estimate is an UPPER BOUND -> can fail even when actual bytes would be under. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Why not read in full |
|-----|-----------------------|
| github.com/googleapis/python-bigquery/issues/129 | thread-safety docs request, no resolution text in page |
| github.com/googleapis/google-cloud-python/issues/3522 | "increment threadsafety when httplib2 removed" -- implies bigquery.Client threadsafety improved post-httplib2 |
| github.com/googleapis/python-bigquery/issues/1165 | retry `deadline` overridden by `timeout` in result() (google-api-core deprecation the finding flags) |
| github.com/googleapis/google-cloud-python/issues/14274, .../python-bigquery-storage/696 | Storage READ client concurrent-construction deadlock (NOT the query client) |
| github.com/googleapis/python-bigquery/issues/6301, 4135, 7831 | result() 500/timeout/done()-stuck reports (corroborate hang risk) |
| docs.cloud.google.com/bigquery/docs/cached-results | cache hit -> 0 bytes billed (cap inert on cache hits) |
| docs.cloud.google.com/bigquery/docs/custom-quotas | project/user daily byte quotas (org-level cost guard, complementary) |
| oneuptime.com .../bigquery-parameterized-queries (2026-02-17); .../custom-cost-controls (2026-02-17) | 2026 practitioner restatements (recency) |
| cloud.google.com/blog/.../controlling-your-bigquery-costs | maximum_bytes_billed cost-control guidance |
| docs.cloud.google.com/bigquery/docs/samples/bigquery-query-params-arrays | ArrayQueryParameter code sample |
| hevodata.com/learn/bigquery-parameterized-queries | parameterized-query primer |

### Key findings mapped to pyfinagent (file:line)
1. **Params protect values, not identifiers** (Src 1). -> get_agent_memories (bigquery_client.py:500,506): parameterize the `agent_type` VALUE (`WHERE agent_type = @agent_type`) and the `LIMIT` (`LIMIT @limit`); the table name f-string at :498/:503 STAYS (identifier, cannot be a param -- and it is internal, not user input). data_ingestion ticker list (:82,:180): `ArrayQueryParameter("tickers","STRING",tickers)` + `IN UNNEST(@tickers)`, exactly cache.py:108-109.
2. **`result(timeout=)` = client-wait ceiling, not job-cancel** (Src 2,5). -> the data-bq-03 sweep's `timeout=30/60` matches the project's existing bigquery_client.py:533 convention; it prevents the worker thread hanging on a dropped HTTP transport (the #1922 6-day-hang failure mode). It does NOT abort the BQ job -- if hard job caps are ever wanted, that's `job_timeout_ms` (out of scope for 75.9; do NOT add it -- would change job behavior).
3. **maximum_bytes_billed is a pre-execution kill switch, fails free** (Src 3,6). -> data-bq-06 factory default 5 GiB (5*1024**3 = 5368709120). Fails the job BEFORE scanning if the estimate exceeds 5 GiB, no charge. Clustered-table upper-bound caveat: 5 GiB is generous vs pyfinagent table sizes, so false-fails are unlikely, but document the caveat. Cache hits bill 0 bytes so the cap never bites them.
4. **bigquery.Client thread-sharing** (Src 4 + snippets 129/3522). Official docs still lack a crisp statement, BUT: (i) the httplib2 non-thread-safety (Src 4) does NOT apply -- modern bigquery.Client uses the requests transport; (ii) the documented DEADLOCK is on CONCURRENT CONSTRUCTION of the Storage READ client, which perf-11 avoids two ways: it uses the QUERY client, and `@lru_cache` constructs EXACTLY ONCE under CPython's internal lock, then shares. Sharing one query client to SUBMIT queries across threads is the common, recommended pattern. -> perf-11 zero-arg lru singleton is safe; the finding's premise (repeated credential JSON parse + fresh pools per call) is the real waste being removed.

## 4. Recency scan (last 2 years, 2024-2026)

Searched 2026- and 2025/2024-scoped variants for all six topics. Result: **no API-behavior change supersedes the canonical official semantics.** The parameterized-query, `maximum_bytes_billed`, and `result(timeout=)` contracts are stable. Newer material is (a) 2026-02 practitioner posts (oneuptime) that RESTATE the same official behavior, and (b) live BUG reports -- python-bigquery #1922 (result() indefinite hang, google-cloud-bigquery 3.21.0) and storage-client deadlock #14274 -- which CONFIRM rather than overturn the finding's rationale. `job_timeout_ms` (added to QueryJobConfig in recent releases) is the only genuinely newer API surface; it is the correct tool for hard job caps but is deliberately OUT of scope for 75.9 (the step scopes `result(timeout=)`, not job cancellation). Three-variant discipline: current-year (2026 oneuptime/systemsarchitect hits), last-2-year (#1922 2024, #14274), year-less canonical (official docs.cloud.google.com references + thread_safety.html).

## Consensus vs debate
Consensus (official + practitioner): parameterize values; cap bytes billed per job; never rely on an untimed `.result()`. Debate/nuance: the ONLY genuinely contested point is whether `result(timeout=)` reliably bounds wall-clock -- #1922 shows it can still hang, so the honest framing for Q/A is "timeout=30 massively reduces but does not 100% eliminate the hang surface; it is the project's adopted convention (bigquery_client.py:533), not a hard guarantee." No source argues AGAINST adding the timeout.

## Pitfalls (from literature + code)
- Adding a param but leaving the f-string in place = dead param (vacuous crit-2 pass) -- assert absence.
- `timeout=` on `result()` != job cancel; do not oversell it or add `job_timeout_ms` (scope creep).
- A too-low `maximum_bytes_billed` on a clustered table can false-fail (upper-bound estimate) -- 5 GiB default mitigates.
- lru_cache on a function taking non-hashable `Settings` will error/leak -- keep `get_bq_client()` zero-arg.
- Fail-closed dedup must distinguish empty-result (insert) from exception (abort) -- see §6.

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: 5 official Google docs + 1 maintainer GitHub issue)
- [x] 10+ unique URLs total (24 collected: 6 full + ~18 snippet)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (§0-§2, §5-§6)
- [x] Internal exploration covered every enumerated module (+ 5 path drifts + 1 phantom + macro-parallel found)
- [x] Contradictions/consensus noted; all claims cited per-claim

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 18,
  "urls_collected": 24,
  "recency_scan_performed": true,
  "internal_files_inspected": 25,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Re-anchored all 75.9 sites; found material drift the executor must apply: migrations are 13 files/20 untimed .result() sites (not 12); 5 external file paths moved (sortino->metrics/, pead_signal->services/, sector_calendars->services/, cost_budget_watcher->slack_bot/jobs/ AND is a PHANTOM with no .result(), harness_autoresearch->api/); monthly_approval_api untimed line is :143 not :141; bigquery_client.py has 18 untimed .result() not 3; perf-11 '34' is the api+services subset (repo-wide 45 BigQueryClient/49 bigquery.Client); _get_existing_macro shares the fail-open bug but is FROZEN (queue separately). Templates confirmed: cache.py:108 UNNEST, get_recent_reports:278 params, get_paper_portfolio:533 timeout, skill_optimizer:172-176 logged-sibling, settings.py:612 lru. BOUNDARY: fail-closed dedup is safe on ALL callers (sole autonomous caller daily_price_refresh:86-87 already fail-open-wraps it; empty-result still inserts, only query-EXCEPTION aborts). External: params protect values not identifiers; result(timeout=) bounds client wait not job (issue #1922 = 6-day hang proves the risk); maximum_bytes_billed fails pre-execution free (5 GiB); lru singleton query-client sharing is safe. Prescribed a full vacuous-pass mutation matrix (crit 1 fixture must be NON-empty; crits 2/3/5 need absence-assertions + AST + missing-file hard-fail; crit 4 needs adoption not just constant; crit 6 needs identity test).",
  "brief_path": "handoff/current/research_brief_75.9.md",
  "gate_passed": true
}
```
