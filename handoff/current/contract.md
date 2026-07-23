# Contract -- Step 75.9: BigQuery fail-closed dedup, parameterization, 30s-timeout sweep, cost guard

- **Step id**: 75.9 (phase-75, Audit75 S9) -- P1, executor sonnet-tier
- **Date**: 2026-07-23
- **Author**: Main (contract + orchestration + review). **GENERATE delegated to a Sonnet-4.6 executor agent** per the step's `[executor: sonnet-4.6/high]` tag and the operator's session directive (recurring/mechanical work on cheaper models; Opus stays on the Researcher/Q-A gates; Main reviews the diff before Q/A).
- **BOUNDARY (from step text)**: paper-only, no schema/table changes, historical_macro frozen.

## Research-gate summary (gate PASSED)

Workflow `wf_d6469920-55f` (researcher role, opus/max, tier=moderate).
Envelope: `external_sources_read_in_full=6, snippet_only_sources=18, urls_collected=24, recency_scan_performed=true, internal_files_inspected=25, gate_passed=true`.
Brief: `handoff/current/research_brief_75.9.md`.

**Step-text corrections adopted (binding; the audit anchors had drifted):**
1. **Migrations**: 13 distinct files / 20 untimed `.result()` sites, not "12 files" (full list in the brief: add_efficiency_snapshots:96, add_external_flow_today_column:70+77, add_round_trip_schema:36, add_session_budget_to_llm_call_log:85, add_ticker_to_llm_call_log:81, create_alpha_velocity_table:81+96, create_data_source_events_table:93, create_directive_versions_table:77+92, create_historical_fx_rates_table:64, create_options_snapshots_table:104, create_promoted_strategies_table:96, create_strategy_deployments_view:114+116+118+143, phase_32_1_add_stop_advanced_at_R:71+78) -- all `timeout=60`. Scanning the 13 satisfies criterion 3's "the 12" a fortiori.
2. **Path drifts**: sortino is `backend/metrics/sortino.py:114`; pead_signal is `backend/services/pead_signal.py:342`; sector_calendars is `backend/services/sector_calendars.py:200`; harness_autoresearch is `backend/api/harness_autoresearch.py:196`. monthly_approval_api's untimed site is `:143`.
3. **Phantom site**: `backend/slack_bot/jobs/cost_budget_watcher.py` has NO `.result()`/`.query()` at all -- it stays in the criterion-3 scan list (vacuously satisfied at 0 sites) but gets no edit.
4. **Undercount**: `bigquery_client.py` has **18** untimed `.result()` sites (281, 295, 352, 374, 455, 470, 508, 540, 571, 580, 590, 632, 676, 704, 986, 1037, 1048, 1069), not the 3 the step implies; `:540/:571` are DML -> `timeout=60`.
5. **perf-11 count**: "34 construction sites" is correct only for `BigQueryClient(` under backend/api/ + backend/services/; repo-wide non-test is 45 `BigQueryClient(` / 49 `bigquery.Client(`. The step's "migrate at least these named ones" scope is unchanged.
6. **Discovered defect (queued, NOT fixed here)**: `_get_existing_macro` (data_ingestion.py:271-278) has the identical fail-open dedup bug but historical_macro is FROZEN -> queued as step **75.9.1**.

**Key research findings (load-bearing):**
- **Boundary cleared**: every caller of ingest_prices/ingest_fundamentals tolerates fail-closed (daily_price_refresh.py:82 wraps in try/except at :86-87 -> self-heals next day; backtest_engine.py:1303 guarded non-fatal; api/backtest.py:201 -> HTTPException 500). No caller relies on the fail-open `set()`.
- **The distinction the fix must preserve**: a SUCCESSFUL query returning 0 rows (empty/first-run table) keeps yielding an empty set -> insert-all (unchanged). Only the query-EXCEPTION path changes to log + re-raise. `_ensure_tables_exist()` runs before ingest in run_full_ingestion, so cold-start never throws table-not-found.
- `result(timeout=)` bounds the client-side transport wait (raises concurrent.futures.TimeoutError), NOT job cancellation; python-bigquery #1922 documents a >6-day hang -- the risk is real. Matches the existing compliant convention at bigquery_client.py:533. Do NOT add job_timeout_ms (scope creep).
- `maximum_bytes_billed`: bytes estimated BEFORE execution; over-cap queries FAIL WITHOUT charge (true pre-charge kill switch). 5 GiB = 5368709120. Cache hits bill 0 (cap inert on them).
- Parameters protect VALUES not identifiers -- agent_type + LIMIT parameterizable; table-name f-strings stay. In-repo templates: cache.py:108-109 (ArrayQueryParameter + IN UNNEST(@tickers)); bigquery_client.py:278-280 (ScalarQueryParameter).
- lru singleton is thread-safe for bigquery.Client (requests transport; the httplib2 caveat does not apply). `get_bq_client()` MUST be zero-arg (mirror settings.py:612) -- Pydantic Settings is not reliably hashable.
- Exclude `future.result()` at api/paper_trading.py:1164 (ThreadPool future, not a BQ job) from the sweep.

## Hypothesis

The BQ data plane can be made fail-closed (dedup), injection-clean (parameterized values), hang-proof (timeout on every job result), and cost-capped (maximum_bytes_billed factory) as a purely mechanical sweep with zero behavior change on the success path -- provable offline by a mocked-client test file whose scans hard-fail on missing paths and whose fixtures can represent the failure (non-empty ingest mock).

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.9)

verification.command:
```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_bq_discipline.py -q
```

1. "New backend/tests/test_phase_75_bq_discipline.py passes offline and asserts with a mocked client whose query() raises: ingest_prices performs ZERO insert_rows_json calls and surfaces the error (fail-closed dedup), with the exception logged"
2. "Source scan in the test: get_agent_memories and the data_ingestion ticker query build via QueryJobConfig parameters (ScalarQueryParameter/ArrayQueryParameter present; no f-string interpolation of agent_type/limit/ticker values into SQL text)"
3. "AST/text scan proves every .result( call in backend/db/bigquery_client.py, the 13 enumerated external files, and the 12 enumerated migration files carries a timeout= argument"
4. "bigquery_client exposes one shared QueryJobConfig factory setting maximum_bytes_billed (default 5 GiB documented) and its own query paths use it; test asserts the factory value and at least one call-path adoption"
5. "skill_optimizer outcomes-query failure logs a warning and returns []/degraded (no bare pass -- source scan), and slot_accounting default helpers reuse a module-level client with timeout=30"
6. "get_bq_client() is lru_cached and imported by api/paper_trading.py, api/performance_api.py, and api/reports.py (import scan); repeated calls return the identical instance (test)"

(Criterion-3 note: "the 13 enumerated external files" is read at the corrected paths above, minus none -- the phantom cost_budget_watcher stays in the scan and passes at 0 sites; "the 12 enumerated migration files" is covered by scanning the measured 13.)

verification.live_check: "handoff/current/live_check_75.9.md: verbatim output of this step's verification command (exit 0) + git diff --stat proving the change surface; for any flag-gated live-loop behavior an ON-vs-OFF $0 diff, and for UI-touching parts a Playwright/curl capture. Findings covered: data-bq-01, data-bq-02, data-bq-03, data-bq-06, py-core-03, gap3-08, gap6-09, perf-11"

## Plan steps

1. **(a) Fail-closed dedup**: `_get_existing_price_dates` (data_ingestion.py:91) + `_get_existing_fundamentals` (:189) -- log the exception, re-raise. `_get_existing_macro` UNTOUCHED (frozen; queued 75.9.1). Empty-success-result path byte-identical.
2. **(b) Parameterization**: get_agent_memories (:500,:506) -> ScalarQueryParameter (mirror :278-280); data_ingestion ticker lists (:82,:180) -> ArrayQueryParameter + IN UNNEST(@tickers) (mirror cache.py:108-109). Table names stay f-strings (identifiers are not parameterizable).
3. **(c) Timeout sweep**: `timeout=30` (DML/DDL `timeout=60`) on ALL 18 bigquery_client.py sites, the corrected external sites (paper_trader.py:1245, cycle_health.py:474, metrics/sortino.py:114, api/paper_trading.py:1127, api/performance_api.py:82, services/pead_signal.py:342, services/sector_calendars.py:200, skill_optimizer.py:173+185, slot_accounting.py:139 (+:118/:134 per module-client change), api/harness_autoresearch.py:196, monthly_approval_api.py:143), and the 13 migration files / 20 sites (timeout=60). Exclude ThreadPool future.result() sites.
4. **(d) Cost guard**: one shared QueryJobConfig factory in bigquery_client.py applying `maximum_bytes_billed` (default 5 GiB = 5368709120, documented) adopted on its query paths; behavioral test asserts the value reaches client.query's job_config.
5. **(e) skill_optimizer.py:188**: bare `except: pass` -> logger.warning + degraded return mirroring the sibling :172-176; timeout=30 on :173/:185. **slot_accounting**: module-level client reuse + timeout=30.
6. **(f) `get_bq_client()`**: zero-arg @lru_cache in bigquery_client.py mirroring get_settings (settings.py:612); adopted in api/paper_trading.py (8 inline construction sites), api/performance_api.py:59, api/reports.py:24-25.
7. **Tests** (backend/tests/test_phase_75_bq_discipline.py, offline, PYFINAGENT_TEST_NO_BQ conventions): crit-1 with a NON-EMPTY yf mock + query.side_effect (asserting BOTH the raise AND insert_rows_json.assert_not_called()); crit-3 AST scan that HARD-FAILS on any missing enumerated path and guards the empty-list case; crit-4 behavioral adoption assert; crit-6 identity assert + call-site (not just import) scan.
8. **Mutation matrix** (executor runs, Main spot-checks): revert dedup re-raise; blank the yf fixture (STUB mutation -- crit-1 must go vacuous-red); strip one params config; drop timeout= from one site in each scanned group (bigquery_client, external, migration); point the scan at a missing path (must ERROR not skip-green); zero out maximum_bytes_billed; break lru identity; restore the bare pass. Every mutation must flip red; matrix results in experiment_results with measured figures.
9. **Queue 75.9.1** in masterplan (pending): _get_existing_macro fail-open dedup on the frozen table (fix DARK/inert until historical_macro un-freeze token).
10. **live_check_75.9.md**: verbatim pytest (exit 0) + git diff --stat. No UI; no flag-gated live-loop behavior.

## Explicitly NOT in scope

- `_get_existing_macro` / anything touching historical_macro (frozen) -- queued as 75.9.1
- Schema/table changes of any kind; `job_timeout_ms`; repo-wide client-construction migration beyond the three named api files
- ThreadPool `future.result()` sites

## References

- `handoff/current/research_brief_75.9.md` (6 read-in-full incl. Google BQ parameterized-queries + best-practices-costs + QueryJob.result docs, python-bigquery GitHub #1922, google-api-python-client thread-safety doc; 24 URLs; recency scan)
- `handoff/current/audit_phase75/confirmed_findings.json` (data-bq-01/02/03/06, py-core-03, gap3-08, gap6-09, perf-11)
- CLAUDE.md Harness Protocol; `.claude/rules/research-gate.md`; feedback_queue_discovered_defects_in_masterplan; feedback_mutation_test_guards_and_fixtures
