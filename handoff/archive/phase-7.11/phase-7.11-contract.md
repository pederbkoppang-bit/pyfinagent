# Sprint Contract — phase-7 / 7.11 (Shared scraper infrastructure)

**Step id:** 7.11 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** moderate

## Research-gate summary

8 sources in full (Apify best-practices 2026, Oxylabs httpx-vs-requests, GCP BQ audit-log pipeline, MS correlation-id playbook, oneuptime circuit-breaker 2026, Oxylabs requests-retry, GCP partitioned tables, findwork.dev requests-advanced), 18 URLs, three-variant queries, recency scan. `gate_passed: true`. Resolves advisories `adv_73_cdn_403` (bounded backoff) and `adv_71_docstring_merge` (audit log is streaming-only).

## Hypothesis

Consolidate the duplicated HTTP + BQ helpers across 8 ingesters into `backend/alt_data/http.py`. Ship `ScraperClient` with bounded backoff (`min(base * 2^attempt, base * 8)`), full-jitter, sliding-window circuit breaker (deque maxlen=20, trips >50% failure over window, excludes 4xx), correlation-id audit rows to `pyfinagent_data.scraper_audit_log`. Live-create the audit table.

## Immutable criteria

- `python -c "import ast; ast.parse(open('backend/alt_data/http.py').read())"`
- `bq ls pyfinagent_data | grep -q scraper_audit_log`

## Plan

1. Write `backend/alt_data/http.py` (~260 lines):
   - `UserAgent` constants (SEC, REDDIT, GENERIC).
   - `RateLimit` dataclass with bounded backoff + max_attempts.
   - Preset map `SOURCE_PRESETS` for FINRA (5s 403 base), SEC (60s), Reddit (60s), generic.
   - `ScraperClient` class: rate-limit + retry loop with jitter + sliding circuit breaker + audit-write hook.
   - `ensure_audit_table` + `_audit_row` fail-open.
   - `get_shared_client(source_name)` factory.
2. Create `pyfinagent_data.scraper_audit_log` live via `ensure_audit_table()`.
3. Verify both immutable criteria + regression.
4. Q/A. Log. Flip.

## Out of scope

- No refactor of existing 8 ingesters to USE the new client (separate cleanup; explicitly not this cycle).
- No httpx migration (research confirms requests is fine for sync scrapers).
- No OpenTelemetry wiring (correlation ID via UUID suffices).
- No tests — the smoketest pattern for a synchronous HTTP helper belongs in phase-7.12 or a future cleanup.
- ASCII-only.

## References

- `handoff/current/phase-7.11-research-brief.md`
- `backend/alt_data/{congress,f13,finra_short,twitter,google_trends,reddit_wsb,hiring,etf_flows}.py` (duplicated patterns to consolidate)
- `docs/compliance/alt-data.md` Sec. 6.1 (audit-log DDL schema)
- Advisories: `adv_73_cdn_403`, `adv_71_docstring_merge`
- `.claude/masterplan.json` → phase-7 / 7.11
