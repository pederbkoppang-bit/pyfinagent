# Sprint Contract -- phase-6.7 Rate Limits, Failure Alerting, Cost Telemetry

**Written:** 2026-04-19 (PRE-commit, before any generator work).
**Step id:** `phase-6.7` in `.claude/masterplan.json` phase-6 (News & Sentiment Cron).
**Parallel-safety:** phase-specific filename; autonomous harness owns rolling files for phase-2.12. Mirror to rolling before masterplan flip.

## Research-gate summary

Researcher spawned today. Envelope: `{tier: moderate, external_sources_read_in_full: 8, snippet_only_sources: 10, urls_collected: 18, recency_scan_performed: true, internal_files_inspected: 15, gate_passed: true}`. Brief at `handoff/current/phase-6.7-research-brief.md` (139 lines), 4 new 2024-2026 recency findings.

Key research decisions (empirical):
- **aiolimiter** for async leaky-bucket per-source rate limiting — minimal dep (6-7 KB, zero runtime deps), asyncio-native, Python 3.9+. Fits the existing httpx/asyncio stack.
- **Retry policy**: 3 attempts, base 1s, multiplier 2x, cap 30s, full-jitter; retry on 429+503+network; no-retry on 4xx non-429; honour `Retry-After` header on 429 (Anthropic sends it — confirmed in phase-4.14.11).
- **Alert thresholds**: N=3 consecutive failures over 5-min window before firing; repeat 1h for error severity, 4h for warnings; inhibition rule (critical suppresses lower-severity same-source).
- **Separate `api_call_log` BQ table**; do NOT extend `llm_call_log` (phase-4.14.23). AI telemetry schema differs from general API schema (OneUptime Apr 2026 guidance).
- **In-memory alert dedup** sufficient — single-process asyncio app. `dict[(source, error_type), datetime]` last-fired tracker with TTL.
- **Module-level singleton** rate limiters per-source (prevents thundering-herd on startup).
- **Existing primitives to reuse**: `scheduler.send_trading_escalation(severity, title, details)` at `backend/slack_bot/scheduler.py:201-250` for routing; `backend/services/cycle_health.py:68-112` for heartbeat/JSONL pattern.

## Hypothesis

A thin observability layer built from `aiolimiter` + a backoff helper + an alert dedupe class + an `api_call_log` BQ table can harden all existing + future external-API callers without invasive refactors, and retrofit the `llm_call_log` BQ writer at the same time to close the observability gap flagged in phase-4.14.23.

## Success criteria

NOTE: masterplan step has `verification: null`. Defining in-contract per discipline:

**Functional:**
1. New package `backend/services/observability/` with `__init__.py`, `rate_limit.py`, `retry.py`, `alerting.py`, `api_call_log.py`.
2. **`rate_limit.py`** — `get_rate_limiter(source: str) -> AsyncLimiter`. Module-level singleton per source. Reads per-source defaults from `backend/config/settings.py` (new keys: `finnhub_rate_limit_rps=25`, `benzinga_rate_limit_rps=2`, `alpaca_rate_limit_rps=30`, `fred_rate_limit_rps=5`, `alphavantage_rate_limit_rps=1`). Safe fall-through when `aiolimiter` isn't installed (no-op limiter that passes through). Exposes both async and sync adapter so existing `requests.get` callers can adopt without full async migration.
3. **`retry.py`** — `retry_with_backoff(fn, *, max_attempts=3, base=1.0, multiplier=2.0, cap=30.0, jitter="full", retry_on=(429, 503, 502, 504), honor_retry_after=True) -> Any`. Synchronous variant for `requests.get` callers; async variant for httpx/asyncio. Returns result or raises last exception after exhausting attempts. Correctly handles `Retry-After` as integer-seconds or HTTP-date.
4. **`alerting.py`** — `AlertDeduper(window_minutes=5, repeat_hours=1)` class with `.should_fire(source, error_type) -> bool` returning True iff `>=3` occurrences within window AND (never fired OR last-fired > repeat_hours ago). `raise_cron_alert(source, error_type, severity, title, details)` wraps `AlertDeduper.should_fire` + `send_trading_escalation` so callers see a single line. Critical severity bypasses dedup.
5. **`api_call_log.py`** — `log_api_call(source, endpoint, http_status, latency_ms, response_bytes, cost_usd_est=0.0, ok=True, error_kind=None, request_id=None) -> None`. Module-level buffered writer to BQ `api_call_log`; flushes every 60s or 100 rows (whichever first). Fail-open: BQ exception logs at WARNING, does not raise. Dataset resolved from `settings.bq_dataset_observability` with fallback to `pyfinagent_data`.
6. **New BQ migration** `scripts/migrations/add_api_call_log.py` — creates `api_call_log` table with columns: `ts TIMESTAMP NOT NULL`, `source STRING NOT NULL`, `endpoint STRING`, `http_status INT64`, `latency_ms FLOAT64`, `response_bytes INT64`, `cost_usd_est FLOAT64`, `ok BOOL`, `error_kind STRING`, `request_id STRING`. PARTITION BY DATE(ts); CLUSTER BY source, ok. `--dry-run` prints DDL.
7. **`llm_call_log` BQ writer retrofit** — `backend/agents/llm_client.py` `ClaudeClient.generate_content()` writes a row to `llm_call_log` after each call using the existing `UsageMeta` data. Fail-open. This closes the phase-4.14.23 gap (migration created the table but writer was deferred).
8. **Wire rate-limit + retry into one news source and one calendar source** as reference implementations: `backend/news/sources/finnhub.py` and `backend/calendar/sources/finnhub_earnings.py`. Both call `log_api_call` on success and error. Other adapters (benzinga, alpaca, fred, fed_scrape, alphavantage) documented as follow-up in experiment_results.md but not wired in this cycle.
9. **Requirement**: add `aiolimiter>=1.2.1` to `backend/requirements.txt` with a comment explaining the choice.
10. **Settings**: new keys in `backend/config/settings.py`: `finnhub_rate_limit_rps: int = 25`, `benzinga_rate_limit_rps: int = 2`, `alpaca_rate_limit_rps: int = 30`, `fred_rate_limit_rps: int = 5`, `alphavantage_rate_limit_rps: int = 1`, `alert_consecutive_failure_threshold: int = 3`, `alert_debounce_minutes: int = 5`, `alert_repeat_hours: int = 1`.
11. **Fail-open discipline**: every primitive catches at its boundary — retry exhaustion raises, but the caller's existing try/except still fires; rate-limit no-op when aiolimiter absent; alerting no-op when Slack not configured; BQ writer no-op when auth missing. Never block the caller's happy path on infra failures.
12. **Tests**: `backend/tests/test_observability.py` with >=8 tests covering: rate-limit singleton reuse; retry-with-jitter correctly caps max delay; retry honours `Retry-After`; alert dedup fires only at N=3; alert dedup respects repeat window; alert critical-bypass; api_call_log buffered writer batches; aiolimiter-absent graceful fallback.
13. **Non-goals**: no Prometheus/Grafana export; no OpenTelemetry; no PagerDuty integration (Slack-only per existing infra); no wiring into all 5 remaining adapters (deferred to phase-6.8 smoketest); no CircuitBreaker abstraction (3-retry + fail-open is sufficient per research).

**Correctness verification commands:**
- `python -c "import ast; ast.parse(open('backend/services/observability/rate_limit.py').read())"` -> exit 0 (plus retry, alerting, api_call_log)
- `python -c "from backend.services.observability import get_rate_limiter, retry_with_backoff, AlertDeduper, log_api_call, raise_cron_alert; print('ok')"` -> `ok`
- `python scripts/migrations/add_api_call_log.py --dry-run` -> prints CREATE TABLE DDL, exit 0
- `pytest backend/tests/test_observability.py -x -q` -> all pass

## Plan steps

1. Add `aiolimiter` to `backend/requirements.txt`. Install via `pip install aiolimiter` in the active venv.
2. Create `backend/services/observability/` with 4 module files + `__init__.py`.
3. Add 8 settings keys to `backend/config/settings.py`.
4. Write `scripts/migrations/add_api_call_log.py`.
5. Retrofit `llm_call_log` writer into `backend/agents/llm_client.py::ClaudeClient.generate_content`.
6. Wire rate-limit + retry + logging into `backend/news/sources/finnhub.py` and `backend/calendar/sources/finnhub_earnings.py`.
7. Write tests.
8. Run verification, capture into `phase-6.7-experiment-results.md`.

## References

- `handoff/current/phase-6.7-research-brief.md` (139 lines, 8 read-in-full, 4 recency findings, 15 internal files)
- `scripts/migrations/add_llm_call_log.py` (DDL pattern to parallel)
- `backend/agents/llm_client.py:225-280` (UsageMeta), `:620-690` (generate_content cache block)
- `backend/services/cycle_health.py:68-112` (JSONL heartbeat pattern)
- `backend/slack_bot/scheduler.py:201-250` (send_trading_escalation)
- `backend/news/sources/finnhub.py`, `backend/calendar/sources/finnhub_earnings.py` (wire-up targets)
- External read-in-full: AWS builders-library retries/backoff, AWS prescriptive guidance, aiolimiter GH + PyPI, PyrateLimiter GH, eraser.io token-vs-leaky, betterstack alert-fatigue, dataops Medium severity routing.

## Researcher agent id

`a922c2638e6355907`
