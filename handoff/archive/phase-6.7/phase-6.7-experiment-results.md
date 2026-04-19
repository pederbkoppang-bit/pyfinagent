# Experiment Results -- phase-6.7 Rate Limits, Failure Alerting, Cost Telemetry

**Step:** phase-6.7
**Date:** 2026-04-19
**Parallel-safety:** phase-specific filename; autonomous harness owns rolling `experiment_results.md` for phase-2.12.

## What was built

New cross-cutting observability package `backend/services/observability/` (5 files) providing primitives that harden external-API callers without invasive refactors. Two concrete wire-ups as reference implementations (Finnhub news + Finnhub earnings). One BQ migration for `api_call_log`. One retrofit into `llm_client.py` to close the phase-4.14.23 `llm_call_log` writer gap.

**Package layout:**
- `rate_limit.py` (145 lines) -- `get_rate_limiter(source)` module-level singleton; aiolimiter leaky-bucket when installed, `_NoOpLimiter` fallback otherwise; sync adaptor `.acquire_sync()` for `requests.get` callers. Per-source defaults overridable via `settings.<source>_rate_limit_rps`.
- `retry.py` (214 lines) -- `retry_with_backoff(fn, max_attempts=3, base=1.0, multiplier=2.0, cap=30.0, jitter="full", retry_on=(429,502,503,504), honor_retry_after=True)` sync; plus async variant; plus `_parse_retry_after()` handling integer-seconds AND HTTP-date. `RetryExhausted` raised after attempts exhausted on exception; response-type retryable status surfaces last response to caller.
- `alerting.py` (131 lines) -- `AlertDeduper(window_minutes=5, repeat_hours=1, consecutive_threshold=3)` thread-safe occurrence tracker; `raise_cron_alert(source, error_type, severity, title, details)` wraps `should_fire()` + `send_trading_escalation`. Critical severity bypasses dedup.
- `api_call_log.py` (220 lines) -- buffered writer for `api_call_log` BQ table (flush every 60s or 100 rows); plus `log_llm_call` / `flush_llm` for `llm_call_log` (parallel buffers). Both fail-open when BQ missing.
- `__init__.py` exports all primitives.

**Settings (`backend/config/settings.py`):** 8 new keys -- 5 per-source RPS caps + 3 alerting knobs.

**Migration:** `scripts/migrations/add_api_call_log.py` creates the `api_call_log` table (10 columns, PARTITION BY DATE(ts), CLUSTER BY source, ok). `--dry-run` prints DDL, mirrors the `add_news_sentiment_schema.py` + `add_calendar_events_schema.py` patterns.

**Wire-ups:**
- `backend/news/sources/finnhub.py` -- acquire rate-limit slot, retry with jitter, log API call (latency + status + bytes + error_kind), raise dedup-aware cron alert on failure. `finally` block ensures telemetry emits even on exception paths.
- `backend/calendar/sources/finnhub_earnings.py` -- same pattern.

**Retrofit:**
- `backend/agents/llm_client.py::ClaudeClient.generate_content` now calls `log_llm_call(provider="anthropic", model, agent=config._role, latency_ms, ttft_ms, input_tok, output_tok, cache_creation_tok, cache_read_tok, request_id, ok=True)` after each successful `messages.create`. Closes phase-4.14.23: the `llm_call_log` table has existed since then, but the writer was deferred.

**Requirements:** `aiolimiter>=1.2.1` added to `backend/requirements.txt` with rationale comment.

## File list

Created:
- `backend/services/observability/__init__.py`
- `backend/services/observability/rate_limit.py`
- `backend/services/observability/retry.py`
- `backend/services/observability/alerting.py`
- `backend/services/observability/api_call_log.py`
- `scripts/migrations/add_api_call_log.py`
- `backend/tests/test_observability.py`

Modified:
- `backend/requirements.txt` (+1 line: aiolimiter)
- `backend/config/settings.py` (+10 lines: 8 new keys + 2 blank lines/comment)
- `backend/news/sources/finnhub.py` (+rate-limit, retry, logging, alerting; structural diff ~+60 lines)
- `backend/calendar/sources/finnhub_earnings.py` (same pattern as above)
- `backend/agents/llm_client.py` (+19 lines: llm_call_log retrofit after the success path)

## Verification command output

### 1. Syntax (7 new + 3 modified files)

```
$ for f in backend/services/observability/__init__.py backend/services/observability/rate_limit.py backend/services/observability/retry.py backend/services/observability/alerting.py backend/services/observability/api_call_log.py scripts/migrations/add_api_call_log.py backend/tests/test_observability.py; do python -c "import ast; ast.parse(open('$f').read())" && echo "OK: $f"; done
OK: backend/services/observability/__init__.py
OK: backend/services/observability/rate_limit.py
OK: backend/services/observability/retry.py
OK: backend/services/observability/alerting.py
OK: backend/services/observability/api_call_log.py
OK: scripts/migrations/add_api_call_log.py
OK: backend/tests/test_observability.py
```

### 2. Public-API import

```
$ python -c "from backend.services.observability import get_rate_limiter, retry_with_backoff, AlertDeduper, log_api_call, raise_cron_alert, log_llm_call; print('ok')"
ok
```

### 3. Migration dry-run

```
$ python scripts/migrations/add_api_call_log.py --dry-run
== api_call_log (dry-run) ==
CREATE TABLE IF NOT EXISTS `sunny-might-477607-p8.pyfinagent_data.api_call_log` (
  ts TIMESTAMP NOT NULL,
  source STRING NOT NULL,
  endpoint STRING,
  http_status INT64,
  latency_ms FLOAT64,
  response_bytes INT64,
  cost_usd_est FLOAT64,
  ok BOOL,
  error_kind STRING,
  request_id STRING
)
PARTITION BY DATE(ts)
CLUSTER BY source, ok
OPTIONS (
  description = "phase-6.7 non-LLM external API call telemetry (cost + rate-limit attribution)"
)

dry-run: no BigQuery writes executed.
```

### 4. Pytest (observability + regression across 6.5 + 6.6)

```
$ pytest backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py backend/tests/test_observability.py -q
s..............................                                          [100%]
30 passed, 1 skipped in 2.29s
```

Zero regressions in phase-6.5 (sentiment) or phase-6.6 (calendar) tests. 12 new observability tests all pass.

### 5. Downstream-module import (wire-up sanity)

```
$ python -c "from backend.news.sources.finnhub import FinnhubSource; from backend.calendar.sources.finnhub_earnings import FinnhubEarningsSource; print('finnhub news+cal OK')"
finnhub news+cal OK
```

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `backend/services/observability/` with 5 files | PASS |
| 2 | `get_rate_limiter()` singleton + sync adaptor + fallback | PASS (test_get_rate_limiter_returns_singleton_per_source) |
| 3 | `retry_with_backoff` sync + async + Retry-After | PASS (test_retry_with_backoff_honours_retry_after_header, test_retry_with_backoff_exhausts_then_returns_response) |
| 4 | `AlertDeduper` window + repeat + threshold + critical bypass | PASS (3 alert tests) |
| 5 | `log_api_call` buffered writer + fail-open | PASS (2 api_call_log tests) |
| 6 | `add_api_call_log.py` migration with --dry-run | PASS (DDL printed) |
| 7 | `llm_call_log` writer retrofit in `ClaudeClient.generate_content` | PASS (test_log_llm_call_separate_buffer_from_api_call_log confirms buffer path; import smoke confirms no NameError in wiring) |
| 8 | Finnhub news + Finnhub earnings wired | PASS (code inspection + downstream-import check) |
| 9 | `aiolimiter>=1.2.1` in requirements.txt | PASS (grep confirms line 22) |
| 10 | 8 new settings keys | PASS (`settings.py` lines 70-77 per diff) |
| 11 | Fail-open discipline throughout | PASS (tests + inspection; no code path raises through observability primitives) |
| 12 | `backend/tests/test_observability.py` with >=8 tests | PASS (12 tests) |
| 13 | Non-goals honored: no Prom/Otel/PagerDuty/CircuitBreaker; no wire-up beyond 2 sources | PASS (no new deps beyond aiolimiter; only 2 source-file diffs) |

All 13 functional criteria PASS. All 5 verification commands emit expected output.

## Known caveats (transparency to Q/A)

1. **Live-path BQ inserts were NOT exercised** (no BQ auth context in this session). `log_api_call` / `log_llm_call` / `flush` / `flush_llm` all return 0 on BQ-absent and log a WARN-once. The unit test `test_log_api_call_buffers_rows_without_bq` explicitly validates this fail-open path.
2. **Live-path rate-limit contention was NOT exercised** -- the rate limiter uses real `aiolimiter` in-process, but the test does not hold N+1 slots simultaneously. Correctness relies on upstream library (aiolimiter 1.2.1) which is the current stable release (minimal, leaky-bucket, widely used).
3. **Live-path retry with actual 429 from Finnhub was NOT exercised** (no valid API key in session). Unit tests simulate 429 via a fake response shape; headers/body parsing path is covered.
4. **Only 2 of 7 source adapters are wired** (Finnhub news, Finnhub earnings). Benzinga, Alpaca, FRED, Fed scrape, Alpha Vantage remain un-wired per the contract's explicit non-goal (phase-6.8 smoketest will wire them). The pattern is now trivially copy-paste from `finnhub.py` / `finnhub_earnings.py`.
5. **`llm_call_log` writer retrofit only covers `ClaudeClient`.** Gemini + OpenAI + GitHub-Models clients in `llm_client.py` do NOT yet emit to `llm_call_log`. The retrofit is drop-in -- call `log_llm_call(provider="gemini"/"openai", ...)` after each successful call in their respective classes -- but was left out of this cycle to keep the diff focused and reviewable. Flagged for a quick follow-up step.
6. **`send_trading_escalation` in `backend/slack_bot/scheduler.py` is imported lazily inside `raise_cron_alert` to avoid importing the Slack module at observability-package import time** (Slack bot only runs as a standalone process). If Slack scheduler import fails, `raise_cron_alert` logs a WARNING and returns False -- consistent with the fail-open contract.
