# Experiment Results — phase-7 / 7.11 (Shared scraper infrastructure)

**Step:** 7.11 **Date:** 2026-04-20 **Cycle:** 1.

## What was built

One new file + one new BQ table.

1. `backend/alt_data/http.py` (~340 lines): `UserAgent` constants (SEC/REDDIT/GENERIC); `RateLimit` dataclass with bounded backoff + max_attempts=3 + backoff_max_multiplier=8 (fixes `adv_73_cdn_403`); `SOURCE_PRESETS` map (sec.edgar, finra.cdn, reddit, x.api, google.trends, github.raw, linkup.api, generic); `ScraperClient` class with sliding-window (deque maxlen=20) circuit breaker, full-jitter backoff, correlation-id audit rows, 4xx-not-counted-as-failure discipline; `ensure_audit_table` + `_audit_write` fail-open; `get_shared_client(source_name)` factory.

2. `pyfinagent_data.scraper_audit_log` BQ table created live. 11 columns matching compliance doc Sec. 6.1. Partition `DATE(ts)`, cluster `source, status_code`.

## Verification

```
$ python -c "import ast; ast.parse(open('backend/alt_data/http.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -c "from backend.alt_data.http import ensure_audit_table; print('ensure_audit_table ->', ensure_audit_table())"
ensure_audit_table -> True

$ bq ls --project_id=sunny-might-477607-p8 pyfinagent_data | grep scraper_audit_log
  scraper_audit_log        TABLE    DAY (field: ts)    source, status_code

$ bq ls --project_id=sunny-might-477607-p8 pyfinagent_data | grep -q scraper_audit_log && echo "GREP EXIT=0"
GREP EXIT=0

$ python -c "open('backend/alt_data/http.py','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `ast.parse(http.py)` | PASS | SYNTAX OK |
| 2 | `bq ls pyfinagent_data | grep -q scraper_audit_log` | PASS | Table listed with correct partition/cluster |

## Advisories resolved

- **adv_73_cdn_403** — `_jittered_backoff(base, attempt, max_mult)` returns `min(base * 2^attempt, base * 8) + random(0,1)`. At `max_mult=8` and base=60s (SEC), worst-case single sleep is 480s + 1s jitter, capped. FINRA preset uses `backoff_403_base_s=5.0` for a max 40s sleep. No more 60·2^attempt-unbounded hang.
- **adv_71_docstring_merge** — `_audit_write` docstring explicitly says "Streaming insert only; never MERGE." Audit-log is append-only by design.

## Known caveats

1. **Existing 8 ingesters NOT migrated** to use the new client. The contract's out-of-scope list says the refactor is separate; this cycle ships the infrastructure only.
2. **No tests** — HTTP clients with real network paths are typically tested via monkeypatched `requests.get` or `responses` library. Deferred.
3. **Egress IP hashing disabled** — `ip_hash` always None in the audit row. Enabling requires reading the egress IP from an external service (STUN or ipify.org). Kept as a future enhancement.
4. **Circuit breaker cool-down is fixed 60s.** Not tunable per source. Adequate for the current volumes; tune later if a source trips frequently.
5. **Python 3.14 typing nit** — `collections.deque[bool]` annotation works at runtime. No runtime cost.
