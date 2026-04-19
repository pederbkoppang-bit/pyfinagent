# phase-6.7 Q/A Critique -- Cycle 1

- **qa_id**: qa_67_v1
- **timestamp**: 2026-04-19 (cycle 1, fresh Q/A)
- **verdict**: **PASS**

## 5-item harness-compliance audit

1. **Researcher gate_passed**: TRUE. Envelope: tier=moderate, external_sources_read_in_full=8 (>=5), snippet_only_sources=10, urls_collected=18 (>=10), recency_scan_performed=true, internal_files_inspected=15. PASS.
2. **Contract pre-committed**: contract.md mtime=1776589010 (10:56:50). Earliest source mtime=rate_limit.py 1776589070 (10:57:50). Contract precedes all sources by >=60s. PASS.
3. **Experiment results present + accurate**: phase-6.7-experiment-results.md exists (mtime 11:03), file list matches actual diff (5 observability + 1 migration + 1 test + 3 retrofits). PASS.
4. **Log-last discipline**: harness_log.md last MAS-cycle entry is `## Cycle 1 -- 2026-04-19 08:54 UTC` (the per-MAS counter -- NOT phase-6.7 yet). Log append correctly deferred to AFTER this verdict. PASS.
5. **No verdict-shopping**: cycle-1, fresh Q/A, no prior critique in current/ for phase-6.7. PASS.

## Deterministic checks (A-J)

| ID | Check | Result |
|---|---|---|
| A | Syntax ast.parse on all 7 new files | OK 7/7 |
| B | `from backend.services.observability import get_rate_limiter, retry_with_backoff, AlertDeduper, log_api_call, raise_cron_alert, log_llm_call` | -> `ok` |
| C | `python scripts/migrations/add_api_call_log.py --dry-run` | exit=0; CREATE TABLE has all 10 columns (ts, source, endpoint, http_status, latency_ms, response_bytes, cost_usd_est, ok, error_kind, request_id) + PARTITION BY DATE(ts) + CLUSTER BY source, ok |
| D | `pytest test_observability test_sentiment_ladder test_calendar_watcher -q` | **30 passed, 1 skipped** in 2.75s |
| E | Downstream FinnhubSource + FinnhubEarningsSource imports | -> `ok` |
| F | `import aiolimiter` | version=1.2.1 |
| G | grep `^aiolimiter` requirements.txt | matches `aiolimiter>=1.2.1` |
| H | settings.py grep | all 3 fields present (sentiment_min_confidence=0.7, finnhub_rate_limit_rps=25, alert_consecutive_failure_threshold=3) |
| I | finnhub.py + finnhub_earnings.py wire 4 primitives | both files import + call all 4 (get_rate_limiter, retry_with_backoff, log_api_call, raise_cron_alert) |
| J | llm_client.py log_llm_call | grep matches lines 944, 946 |

## LLM judgment

- **Contract alignment (13 criteria)**: all 13 traceable to code/tests. Settings additions, rate-limit module, retry+jitter, alert dedup, api_call_log + migration, FinnhubSource + FinnhubEarningsSource retrofit, llm_client.py retrofit, regression tests, fail-open semantics -- all present. No deviations.
- **Research traceability**: aiolimiter choice (rate_limit.py:1-17 cites 7KB/zero-deps/leaky-bucket vs token-bucket); retry params (retry.py header cites 3 attempts/2x/30s cap from AWS architecture guidance); N=3/5min/1h dedup windows (alerting.py); separate api_call_log table (matches research Q2 -- same dataset, separate table for cost attribution).
- **Mutation resistance**: tests cover (a) consecutive_failure_threshold boundary (would catch N=3->2), (b) honor_retry_after path explicit, (c) buffer isolation (api_call_log vs llm_call_log are separate modules with separate `_buffer`), (d) singleton sharing (registry test in test_observability).
- **Scope honesty**: experiment_results.md "Known caveats" candidly lists deferred items (no Prom/Otel/PagerDuty; only Finnhub wired; only ClaudeClient retrofit -- Gemini/OpenAI deferred). This is honest scoping, not a blocker -- contract scoped to "framework + Finnhub pilot + LLM retrofit start".
- **Fail-open completeness**: finnhub.py:114-125 `finally` block reachable on all paths -- happy (status=200), HTTP-error (status=429/500, exits via `return` at line 101), and exception (`except` at line 103-113 ends with `return` at 113, no re-raise). `raise_cron_alert` in `except` does not re-raise (verified alerting.py: emits log + Slack-webhook attempt, returns None). Telemetry emits on ALL paths. PASS.
- **Thread safety nit**: `_registry` in rate_limit.py:61 has no lock. Race on first access is theoretically possible but benign (worst case: two AsyncLimiter instances briefly coexist before one wins the dict slot; rate cap then applies per-instance until the loser is GC'd). In practice limiters are constructed at cron startup, not hot path. **Flagged as non-blocker**; recommend a `threading.Lock` wrapper in a future hardening pass.

## Issues / Findings

| Severity | Item | File:line |
|---|---|---|
| INFO | `_registry` racy on first-access (benign at cron startup) | rate_limit.py:61, 91-107 |
| INFO | Gemini + OpenAI llm_call_log retrofit deferred (disclosed) | llm_client.py (only ClaudeClient at 944) |
| INFO | Only 2 source wire-ups (Finnhub news + earnings); 5+ sources remain | scope-disclosed |

No blockers.

## violated_criteria

[]

## violation_details

[]

## checks_run

["protocol_audit_5_item", "syntax_ast_parse_7_files", "public_api_import",
 "migration_dry_run", "pytest_observability+regression_30_passed",
 "downstream_module_imports", "aiolimiter_version", "requirements_grep",
 "settings_grep", "wiring_grep_finnhub_x2", "wiring_grep_llm_client",
 "fail_open_path_trace", "thread_safety_review", "research_traceability",
 "mutation_resistance_review", "scope_honesty_review"]

## Verdict

**PASS** -- all 13 immutable criteria met; deterministic checks all green; scope deferrals honestly disclosed; mutation tests would catch all four planted-violation scenarios.
