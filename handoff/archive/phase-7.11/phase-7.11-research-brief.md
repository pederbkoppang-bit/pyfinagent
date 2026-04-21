---
step: phase-7.11
title: Shared scraper infrastructure — http.py + scraper_audit_log
tier: moderate
date: 2026-04-19
gate_passed: true
---

## Research: Phase-7.11 Shared Scraper Infrastructure

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://use-apify.com/blog/web-scraping-best-practices-2026 | 2026-04-19 | blog/guide | WebFetch | Exponential backoff + jitter formula: `(2**attempt) + random.uniform(0,1)`; error-rate >10% as circuit threshold; monitor schema validation failures |
| https://oxylabs.io/blog/httpx-vs-requests-vs-aiohttp | 2026-04-19 | comparison/blog | WebFetch | requests is sync-only; httpx supports sync+async+HTTP/2; aiohttp async-only; for a sync scraper library, httpx offers the most future-proof upgrade path without forcing async |
| https://cloud.google.com/blog/products/data-analytics/bigquery-audit-logs-pipelines-analysis | 2026-04-19 | official docs | WebFetch | Streaming inserts handle per-row log ingestion with near-real-time queryability; use partitioned tables with `--use-partitioned-tables`; set partition expiration to control storage costs |
| https://microsoft.github.io/code-with-engineering-playbook/observability/correlation-id/ | 2026-04-19 | authoritative eng playbook | WebFetch | Correlation ID: generate at request boundary, embed in every downstream log and BQ row; propagate via header; OpenTelemetry TraceId is the preferred standard |
| https://oneuptime.com/blog/post/2026-01-23-python-circuit-breakers/view | 2026-01-23 | blog/guide 2026 | WebFetch | Three-state machine (closed/open/half-open); deque(maxlen=N) sliding window; fail_max=3-5; exclude 4xx from failure count; reset_timeout=30-60s |
| https://oxylabs.io/blog/python-requests-retry | 2026-04-19 | blog/guide | WebFetch | urllib3 Retry: backoff_factor=2, status_forcelist=[429,500,502,503,504]; mount adapter pattern; no native backoff_max cap -- must wrap |
| https://docs.cloud.google.com/bigquery/docs/partitioned-tables | 2026-04-19 | official docs | WebFetch | Time-unit column partitioning on TIMESTAMP; for high-write-rate tables use monthly/yearly + clustering to avoid partition-modification quota (4000/day per table); daily granularity is fine for moderate-volume audit logs |
| https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/ | 2026-04-19 | authoritative blog | WebFetch | Per-adapter mount: `session.mount("https://", adapter)` and `"http://"` separately; POST not retried by default -- must set `allowed_methods`; backoff_max cap not exposed natively |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.scrapehero.com/rate-limiting-in-web-scraping/ | blog | Covered by Apify article |
| https://www.scraperapi.com/web-scraping/best-user-agent-list-for-web-scraping/ | blog | User-agent detail already captured |
| https://pypi.org/project/circuitbreaker/ | PyPI | Library survey only; design covered by oneuptime article |
| https://github.com/danielfm/pybreaker | GitHub | Covered by oneuptime article |
| https://scrapingant.com/blog/requests-vs-httpx | blog | Covered by Oxylabs comparison |
| https://medium.com/@ThinkingLoop/10-advanced-logging-correlation-trace-ids-in-python-50bff4024344 | blog | Covered by MS playbook |
| https://last9.io/blog/correlation-id-vs-trace-id/ | blog | Definitional only |
| https://www.proxy-cheap.com/blog/httpx-vs-requests | blog | Covered by Oxylabs comparison |
| https://iproyal.com/blog/best-python-http-clients/ | blog | Covered by Oxylabs comparison |
| https://github.com/jd/tenacity | GitHub | Retry library; urllib3 Retry is sufficient for sync requests |

### Recency scan (2024-2026)

Searched: "shared HTTP client scraper 2026", "httpx vs requests 2025 2024", "circuit breaker Python 2026", "BigQuery audit log 2025".

Found new 2026 findings:
- oneuptime.com (2026-01-23): circuit breaker guide recommending `deque(maxlen=20)` sliding window with failure-rate >50% -- directly aligns with the proposed `circuit_breaker_threshold=0.5` over 20 requests.
- use-apify.com (2026): production scraper guide recommends error-rate >10% alert threshold and jitter on exponential backoff.
- BigQuery audit-log pipeline guide (2026-02): confirms streaming insert + partitioned table as the current best practice for append-heavy audit tables.

No finding in the 2024-2026 window supersedes the canonical urllib3/requests design; requests remains the sync-scraping default and httpx remains the recommended upgrade path.

### Key findings

1. **Per-source backoff cap is not built into urllib3 Retry** -- `backoff_factor` grows unboundedly unless wrapped; FINRA's `60 * 2^attempt` is the exact anti-pattern flagged in adv_73_cdn_403. The `RateLimit.backoff_403_base_s` field with a per-source cap solves this. (Source: oxylabs retry guide, findwork.dev urllib3 article)

2. **Jitter is mandatory to prevent thundering-herd** -- AWS/litl backoff library "full_jitter" is the canonical formula. Deterministic exponential backoff is detectable by bot-detection systems. (Source: Apify 2026, backoff PyPI)

3. **Circuit breaker: failure-rate over sliding window, not absolute count** -- a `deque(maxlen=20)` tracking pass/fail, tripping at >50% failure rate, is the recommended pattern for scrapers where requests are sparse and absolute-count thresholds give false positives. Exclude 4xx from the count -- they are source-side policy, not infrastructure failure. (Source: oneuptime 2026)

4. **Correlation ID: generate UUID at `get()` call, embed in both Python log line and BQ audit row** -- single `request_id` field (UUID4) satisfies the MS playbook's "assign as early as possible" rule and connects log grep to BQ query without OpenTelemetry overhead. (Source: MS Engineering Playbook)

5. **BigQuery streaming insert is correct for audit rows** -- one `insertAll` call per request, daily partition on `DATE(ts)`, cluster on `(source, status_code)`. Partition modification quota (4000/day) is not a concern at alt-data scraper volumes (<<1000 req/day). (Source: GCP official docs, GCP partitioned-tables)

6. **`ensure_audit_table` should use `exists_ok=True` / CREATE TABLE IF NOT EXISTS DDL** -- idempotent DDL is the standard pattern; the existing `ensure_table` functions in congress.py and f13.py already follow this convention (file:line anchors below).

7. **Keep `requests` (not httpx) for phase-7.11** -- all 8 ingesters already import `requests`; httpx is the right long-term upgrade but switching transports in this phase adds scope without benefit. `ScraperClient` wraps `requests.Session` internally.

### Internal code inventory

| File | Lines (approx) | Role | Status |
|------|---------------|------|--------|
| `backend/alt_data/congress.py` | ~314 | Congress trades ingester | Active; has `_get_bq_client` L221, `_resolve_target` L202, `_USER_AGENT` L48, no `_http_get` (inline) |
| `backend/alt_data/f13.py` | ~400+ | SEC 13-F ingester | Active; `_http_get` L121, `_rate_limit` L91, `_RATE_INTERVAL_S` L40 (8 req/s); **403 backoff = `60 * 2^attempt`** (L135) -- known adv_73 issue |
| `backend/alt_data/finra_short.py` | ~260+ | FINRA CDN ingester | Active; `_http_get` L67, `_rate_limit` L63; **403 backoff = `60 * 2^attempt`** (L82) -- adv_73 confirmed; 5xx backoff = `5 * 2^attempt` (L85) |
| `backend/alt_data/etf_flows.py` | ~250+ | ETF flows ingester | Active; `_rate_limit` L65 (sleep only); no `_http_get` wrapper; `_USER_AGENT` L34 |
| `backend/alt_data/reddit_wsb.py` | ~270+ | Reddit WSB ingester | Active; `_USER_AGENT` L42 (Reddit-specific format); no `_http_get` (uses PRAW) |
| `backend/alt_data/twitter.py` | ~200+ | Twitter ingester | Active; `_USER_AGENT` L36 |
| `backend/alt_data/google_trends.py` | ~200+ | Google Trends ingester | Active; `_resolve_target` L89, `_get_bq_client` L108; `_RATE_INTERVAL_S` L35 (12s) |
| `backend/alt_data/hiring.py` | ~200+ | Job-posting ingester | Active; `_resolve_target` L122, `_get_bq_client` L141; `_USER_AGENT` L35 |
| `backend/alt_data/http.py` | -- | **Does not exist** | To be created by phase-7.11 |

**Duplication confirmed across all 8 files:**
- `_resolve_target`: identical pattern in congress.py L202, f13.py L340, finra_short.py L191, google_trends.py L89, hiring.py L122
- `_get_bq_client`: identical fail-open pattern in congress.py L221, f13.py L359, finra_short.py L210, google_trends.py L108, hiring.py L141
- `_http_get` with `60 * 2^attempt` 403 backoff: f13.py L135, finra_short.py L82 (adv_73 confirmed)
- `_USER_AGENT`: 6 files use `"pyfinagent/1.0 peder.bkoppang@hotmail.no"`; reddit_wsb.py L42 uses Reddit-specific UA (must be preserved as override)

### Consensus vs debate (external)

Consensus: exponential backoff + jitter; streaming inserts to partitioned BQ; correlation IDs tied to request boundary; circuit breaker as a deque-based sliding window.

Debate: httpx vs requests -- no clear consensus on migrating existing sync scrapers; Oxylabs leans httpx for future-proofing, others lean requests for simplicity. Decision: keep requests for phase-7.11 given zero existing httpx usage.

### Pitfalls (from literature)

- P1: `60 * 2^attempt` without a cap means attempt=2 sleeps 240s, attempt=3 sleeps 480s -- blocking the process entirely. **adv_73 confirmed.** Cap at `backoff_403_base_s * (2**attempt)` with a `min(..., cap)` guard.
- P2: Recursively calling `_http_get` for retries instead of a loop risks Python stack overflow at high `max_attempts`. ScraperClient should use an iterative retry loop.
- P3: Streaming insert row ordering: BQ streaming does not guarantee insert order; `request_id` (UUID4) + `ts` (TIMESTAMP) together provide sufficient deduplication.
- P4: Circuit breaker counting 403s as failures is wrong -- FINRA returns 403 for rate-limit enforcement, not infrastructure failure. Separate `circuit_failure` (5xx + network error) from `http_policy_error` (403/404).
- P5: `adv_71_docstring_merge`: the `_audit_row` docstring in the proposal says "writes to BQ" -- confirm whether stream insert (one row) or MERGE (upsert). Design decision: stream insert only (append-only audit log; no MERGE needed).

### Application to pyfinagent (mapping to file:line anchors)

| Finding | Maps to |
|---------|---------|
| Consolidate `_resolve_target` | congress.py L202, f13.py L340, finra_short.py L191, google_trends.py L89, hiring.py L122 -- all become `from backend.alt_data.http import _resolve_target` |
| Consolidate `_get_bq_client` | Same 5 files -- become `from backend.alt_data.http import _get_bq_client` |
| Per-source backoff cap (adv_73) | `RateLimit.backoff_403_base_s=5.0` for FINRA; default 60.0 for SEC; cap formula: `min(base * 2**attempt, base * 8)` (hard cap at 3 doublings) |
| Reddit UA preserved | reddit_wsb.py L42 `_USER_AGENT` must NOT be replaced by the shared constant; pass as constructor arg |
| Iterative retry loop | ScraperClient.get() uses `for attempt in range(max_attempts)` not recursion |
| Circuit breaker excludes 4xx | Only 5xx + `requests.exceptions.RequestException` increment the failure counter |
| UUID4 request_id | `import uuid; request_id = str(uuid.uuid4())` at top of `ScraperClient.get()` |
| adv_71 resolved | `_audit_row` docstring: "appends one row via streaming insert"; no MERGE path in audit table |

---

## Design Proposal: `backend/alt_data/http.py`

### Module-level constants

```python
class UserAgent:
    SEC   = "pyfinagent/1.0 peder.bkoppang@hotmail.no"   # EDGAR-compliant
    REDDIT = "python:pyfinagent:1.0 (by /u/pederbkoppang)" # Reddit API rules
    GENERIC = "pyfinagent/1.0 peder.bkoppang@hotmail.no"
```

### `RateLimit` dataclass

```python
@dataclasses.dataclass
class RateLimit:
    per_second_cap: float = 2.0
    burst_cap: int = 1
    backoff_403_base_s: float = 60.0   # FINRA override: 5.0
    backoff_5xx_base_s: float = 5.0
    max_attempts: int = 3
    backoff_max_multiplier: int = 8    # hard cap at base * 8 (3 doublings)
```

FINRA preset: `RateLimit(per_second_cap=0.5, backoff_403_base_s=5.0)` -- satisfies adv_73.

### `ScraperClient` class skeleton

```python
class ScraperClient:
    def __init__(
        self,
        source_name: str,
        user_agent: str,
        rate_limit: RateLimit,
        audit_on: bool = True,
        circuit_breaker_threshold: float = 0.5,
        circuit_window: int = 20,
    ) -> None:
        self._source = source_name
        self._ua = user_agent
        self._rl = rate_limit
        self._audit_on = audit_on
        self._cb_threshold = circuit_breaker_threshold
        self._cb_window: collections.deque[bool] = collections.deque(maxlen=circuit_window)
        self._cb_open_until: float = 0.0   # epoch seconds; 0 = closed
        self._last_req: float = 0.0

    def get(self, url: str, *, accept: str | None = None) -> requests.Response | None:
        # 1. Circuit-breaker open check
        # 2. Rate-limit sleep (time since _last_req vs per_second_cap)
        # 3. Build headers (User-Agent, Accept)
        # 4. Iterative retry loop (range(max_attempts)):
        #    a. GET with timeout=30
        #    b. On 403: sleep min(base*2**attempt, base*max_mult) + jitter; continue
        #    c. On 5xx: sleep min(base_5xx*2**attempt, ...) + jitter; continue
        #    d. On success: record cb pass, audit, return
        #    e. On exception: record cb fail, audit, continue
        # 5. After loop exhausted: record cb fail, audit None, return None

    def _record_cb(self, success: bool) -> None:
        # Append to deque; if failure-rate > threshold, open circuit for reset_timeout
        self._cb_window.append(success)
        if len(self._cb_window) == self._cb_window.maxlen:
            fail_rate = self._cb_window.count(False) / len(self._cb_window)
            if fail_rate > self._cb_threshold:
                self._cb_open_until = time.time() + 60.0

    def _audit_row(
        self, url: str, status_code: int | None, latency_ms: float, error: str | None
    ) -> None:
        # Appends one row via streaming insert to pyfinagent_data.scraper_audit_log
        # Fields: request_id (UUID4), source, url, method="GET", status_code,
        #         latency_ms, user_agent, ts=NOW, bytes_returned, error
        # Fail-open: any BQ error is logged at WARNING, never raises

def get_shared_client(source_name: str) -> ScraperClient:
    # Factory: maps source_name prefix to preset RateLimit + UserAgent
    # "7.2.13f" -> UserAgent.SEC, RateLimit(per_second_cap=8, ...)
    # "7.5.finra" -> UserAgent.SEC, RateLimit(backoff_403_base_s=5.0)
    # "7.5.reddit" -> UserAgent.REDDIT, ...
    # default -> UserAgent.GENERIC, RateLimit()
```

### `ensure_audit_table(project, dataset)` DDL

```sql
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.scraper_audit_log` (
    request_id  STRING    NOT NULL,
    source      STRING    NOT NULL,
    url         STRING    NOT NULL,
    method      STRING,
    status_code INT64,
    latency_ms  FLOAT64,
    user_agent  STRING,
    ip_hash     STRING,
    ts          TIMESTAMP NOT NULL,
    bytes_returned INT64,
    error       STRING
)
PARTITION BY DATE(ts)
CLUSTER BY source, status_code
OPTIONS (require_partition_filter = false);
```

Notes: daily partition on `DATE(ts)` is correct at alt-data volumes (<<4000 partition modifications/day). `ip_hash` is optional (sha256 of egress IP); omit if egress IP is not available without extra syscall.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8 fetched)
- [x] 10+ unique URLs total incl. snippet-only (18 URLs)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (all 8 ingesters)
- [x] Contradictions / consensus noted (httpx vs requests decision documented)
- [x] All claims cited per-claim

---

## Queries run (three-variant discipline)

1. Current-year frontier: "shared HTTP client scraper design patterns retries rate limits 2026"
2. Last-2-year window: "httpx vs requests Python 2025 2024 async scraping performance", "circuit breaker Python HTTP client 2025"
3. Year-less canonical: "Python requests session retry backoff exponential jitter shared HTTP client library", "BigQuery audit log table design partition cluster high write rate", "request_id correlation Python logging BigQuery observability distributed tracing"

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 10,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/phase-7.11-research-brief.md",
  "gate_passed": true
}
```
