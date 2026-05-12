# Research Brief: phase-25.B3 -- Daily loop reads latest promoted strategy via load_promoted_params()

Tier assumption: **moderate** (stated by caller).

---

## Read in full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://docs.cloud.google.com/bigquery/docs/parameterized-queries | 2026-05-12 | Official doc | WebFetch | Full Python samples for ScalarQueryParameter and ArrayQueryParameter with UNNEST for IN-clause; `WHERE state IN UNNEST(@states)` pattern confirmed |
| https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax | 2026-05-12 | Official doc | WebFetch | ORDER BY DESC + LIMIT 1 syntax; WHERE IN for multiple status values confirmed for standard SQL |
| https://platform.claude.com/docs/en/build-with-claude/prompt-caching | 2026-05-12 | Official doc | WebFetch | TTL guidance: 5-min default ephemeral, 1-hour extended; freshness analogy for once-daily cycles favors no in-process TTL cache |
| https://badia-kharroubi.gitbooks.io/microservices-architecture/content/patterns/communication-patterns/fallback-pattern.html | 2026-05-12 | Architecture doc | WebFetch | Fallback pattern: circuit opens -> local-cache fallback; "fallback logic must have little chance of failing"; fail-silent vs fail-fast split |
| https://geeksforgeeks.org/system-design/microservices-resilience-patterns/ | 2026-05-12 | Practitioner blog | WebFetch | Multi-tier fallback: "static responses, cached data, or entirely different service"; Spotify example: personalized -> cached popular -> default |
| https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.client.Client | 2026-05-12 | Official doc | WebFetch | Client.query() signature; ArrayQueryParameter type confirmed; result(timeout=) float parameter |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.quantconnect.com/forum/discussion/18213/automating-quantconnect-strategy-execution-with-dynamic-parameters-via-lean-and-python-backend/ | Practitioner forum | Fetched but contained no hot-swap implementation details; QuantConnect requires full-redeploy for parameter changes |
| https://appmaster.io/blog/microservices-architecture-resiliency-patterns | Blog | Fetched; no multi-tier chain implementation detail |
| https://www.jrebel.com/blog/microservices-resilience-patterns | Blog | Fetched; no Python examples for DB-unavailable fallback |
| https://medium.com/@glee8804/selecting-the-latest-records-in-bigquery-by-partition-by-39c396e8489d | Blog | Fetched; ROW_NUMBER() PARTITION BY pattern confirmed but only for multi-table UNIONs |
| https://dev.to/whoffagents/claude-prompt-caching-in-2026-the-5-minute-ttl-change-thats-costing-you-money-4363 | Blog | Snippet only; TTL change confirmed in platform.claude.com primary source |
| https://docs.cloud.google.com/bigquery/docs/samples/bigquery-query-params-named | Official doc | Fetched; scalar + array param samples confirmed (redundant after parameterized-queries fetch) |
| https://hedgenordic.com/2025/06/report-systematic-strategies-and-quant-trading-2025/ | Industry report | Snippet only; no state-machine content relevant to status field |
| https://github.com/googleapis/python-bigquery/issues/2310 | GitHub issue | Snippet only; confirms BQ client hangs indefinitely on network drop -- reinforces 30s timeout rule |
| https://oneuptime.com/blog/post/2026-01-24-cascading-failures-microservices/view | Blog | Snippet only; cascading failure context |
| https://hevodata.com/learn/bigquery-parameterized-queries/ | Blog | Snippet only; redundant with official GCP docs |

---

## Recency scan (2024-2026)

Searched for:
1. `BigQuery parameterized query Python client latest row status filter 2026`
2. `three-tier fallback pattern database unavailable local file defaults Python 2025`
3. `Anthropic prompt caching TTL 5 minute cache_control ephemeral best practices 2026`

**Findings:**
- **Anthropic prompt caching TTL change (March 6, 2026):** The default ephemeral TTL dropped from 1 hour to 5 minutes. This is directly applicable by analogy: `load_promoted_params()` is called once per daily cycle (24-hour cadence), so no in-process result caching is warranted -- the function should always do a fresh BQ read at cycle start and fall back immediately if BQ is unavailable.
- **BigQuery Python client hang (GitHub issue #2310, known):** If network drops during a BQ query, the client can hang indefinitely. Reinforces the mandatory `result(timeout=30)` on every query (already established in CLAUDE.md and mirrored in all sibling methods in `bigquery_client.py`).
- **No superseding literature** on strategy-registry "latest active" query patterns or multi-tier fallback for once-daily loops found in 2024-2026 window beyond what is documented in the official GCP docs.

---

## Key findings

1. **BigQuery `WHERE status IN UNNEST(@statuses)` with `ArrayQueryParameter` is the idiomatic parameterized multi-value filter.** The parameterized-queries docs show: `WHERE state IN UNNEST(@states)` with `bigquery.ArrayQueryParameter("states", "STRING", ["WA","WI"])`. This avoids string interpolation and SQL injection. (Source: Google BigQuery parameterized queries doc, https://docs.cloud.google.com/bigquery/docs/parameterized-queries, accessed 2026-05-12)

2. **"Latest active row" pattern: `ORDER BY promoted_at DESC, dsr DESC LIMIT 1`.** `ROW_NUMBER() OVER (PARTITION BY ...)` is the right pattern when you need latest-per-group across multiple partitions; for a single-row query from a global table (one active strategy at a time), `ORDER BY ... DESC LIMIT 1` is simpler and cheaper. Tie-break on `dsr DESC` is essential: if two rows share identical `promoted_at` (batch promotion in the same second), `LIMIT 1` without secondary sort is non-deterministic. (Source: BigQuery standard SQL query-syntax doc + glee8804 Medium article, 2026-05-12)

3. **Three-tier fallback chain: BQ -> local JSON -> in-code `{}`.** Fallback pattern literature: "fallback logic must have little chance of failing." Tier-2 (`optimizer_best.json`) is a pure file read with no network dependency; Tier-3 (`{}`) is always safe. The existing `load_best_params()` already implements Tier-2 -> Tier-3. `load_promoted_params()` wraps BQ as Tier-1 and delegates to `load_best_params()` on any exception. (Source: microservices fallback pattern, badia-kharroubi.gitbooks.io + GeeksforGeeks resilience patterns, 2026-05-12)

4. **Status-filter posture for first pass (research call): pass `["pending", "active"]`.** Friday promotion writes rows with `status="pending"` (confirmed at `friday_promotion.py:162`). There is no 25.C3 flipper yet. If the query filters for `status="active"` only, zero rows will match until 25.C3 exists. The recommended first-pass filter is `["pending", "active"]`; when 25.C3 lands and flips rows to `"active"`, the filter can be narrowed to `["active"]` without breaking anything. (Internal audit: `friday_promotion.py:162`)

5. **No in-process TTL cache needed.** `run_daily_cycle` is called once per APScheduler cron tick (daily). There is no benefit to caching the BQ result in module-level state. A fresh BQ read per cycle is correct; the 30s timeout is the only ceiling needed. (Anthropic caching TTL docs + autonomous_loop.py line 53)

6. **`result(timeout=30)` is mandatory on all BQ reads.** Confirmed by CLAUDE.md Critical Rules, mirrored in every sibling method in `bigquery_client.py` (lines 503, 718, 735). BQ Python client can hang indefinitely on network drop (GitHub issue #2310).

7. **`params` column is JSON type in BQ; read requires `TO_JSON_STRING(params)`.** The `save_promoted_strategy` writer uses `PARSE_JSON(@v_params)` to insert (bigquery_client.py:694). On read, `TO_JSON_STRING(params) AS params_json` in the SELECT is needed to recover a serializable string; otherwise `dict(row)` will have a non-JSON-serializable object for the `params` key.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/autonomous_loop.py` | ~850 | Daily cycle orchestrator | Active; `load_best_params` at lines 33-43, caller at line 100 |
| `backend/db/bigquery_client.py` | 812 | BQ wrapper | Active; `save_promoted_strategy` at lines 659-718; no `get_latest_promoted_strategy` yet |
| `backend/autoresearch/friday_promotion.py` | ~185 | Friday strategy promotion | Active; writes `status="pending"` rows at line 162 |
| `backend/services/__init__.py` | 1 line (empty) | Package init | No imports to update |
| `tests/verify_phase_25_A3.py` | 271 | Phase-25.A3 verifier | Reference pattern: `MagicMock()` for `BigQueryClient`, static text analysis + behavioral round-trip |
| `tests/verify_phase_25_B3.py` | N/A | NOT YET CREATED | Must be created by this step |

---

## Consensus vs debate (external)

**Consensus:**
- `WHERE col IN UNNEST(@param)` with `ArrayQueryParameter` is the correct parameterized multi-value filter in BigQuery Python client.
- `ORDER BY timestamp DESC LIMIT 1` is the standard "get latest" pattern; add secondary sort for determinism.
- Three-tier fallback (remote -> local file -> in-code default) is well-established in microservices resilience literature.
- `result(timeout=30)` mandatory given known client hang-on-network-drop behavior.

**Open question resolved as research call:**
- Status filter scope: `["pending","active"]` for first pass; narrows to `["active"]` when 25.C3 lands.

---

## Pitfalls (from literature + code audit)

1. **`params` JSON column serialization trap:** BQ `params` column is `JSON` type. Reading it naively via `dict(row)` yields a non-serializable Python object. Must use `TO_JSON_STRING(params) AS params_json` in the SELECT and `json.loads(row["params_json"])` in Python. (`bigquery_client.py:694` shows the write-side `PARSE_JSON`.)
2. **Missing `result(timeout=30)` causes silent hangs** (GitHub issue #2310). Every BQ call must include it.
3. **Module-level `_running` guard in `autonomous_loop.py:47`.** `load_promoted_params()` must not touch `_running` -- it is a pure loader.
4. **`load_best_params()` has no external callers** (grep confirms only lines 33 and 100 in `autonomous_loop.py`). Safe to add sibling without breaking callers.
5. **Tie-break on `dsr DESC` is essential.** Batch promotion can write multiple rows at the same timestamp.
6. **Do not replace `load_best_params()`.** Step spec: ADD `load_promoted_params()` as a sibling that wraps `load_best_params()` as its Tier-2 fallback.

---

## Application to pyfinagent (mapping to file:line anchors)

| External finding | Maps to | File:line anchor |
|-----------------|---------|-----------------|
| `ArrayQueryParameter` for IN UNNEST | `get_latest_promoted_strategy` SQL | new method in `bigquery_client.py` after line 718 |
| `ORDER BY promoted_at DESC, dsr DESC LIMIT 1` | Same new method | same |
| `TO_JSON_STRING(params) AS params_json` | SELECT clause | same |
| `result(timeout=30)` mandatory | New method call | same |
| Three-tier fallback chain | `load_promoted_params()` | new function in `autonomous_loop.py` after line 43 |
| Caller update: prefer promoted over optimizer | `run_daily_cycle` call site | `autonomous_loop.py:100` |
| `status_filter=["pending","active"]` first pass | `load_promoted_params()` default arg | new function |
| Logging line shapes | `load_promoted_params()` | new function (see below) |

---

## Verbatim SQL for `get_latest_promoted_strategy`

```sql
SELECT
    strategy_id,
    week_iso,
    TO_JSON_STRING(params) AS params_json,
    dsr,
    pbo,
    status,
    allocation_pct,
    promoted_at,
    sortino_monthly
FROM `{table}`
WHERE status IN UNNEST(@statuses)
ORDER BY promoted_at DESC, dsr DESC
LIMIT 1
```

`{table}` resolves to `settings.gcp_project_id + ".pyfinagent_data.promoted_strategies"`.

---

## Verbatim Python signature: `BigQueryClient.get_latest_promoted_strategy`

```python
def get_latest_promoted_strategy(
    self,
    status_filter: list[str] | None = None,
) -> dict | None:
    """phase-25.B3: return the single highest-quality promoted strategy row
    that matches the given status values, or None if the table is empty or
    the status filter matches nothing.

    status_filter defaults to ["pending", "active"] for first-pass: 25.A3
    writes rows with status="pending"; 25.C3 will flip them to "active".
    Using both values here means the daily loop can pick up rows immediately
    after Friday promotion without waiting for 25.C3.

    Returns a dict with keys:
        strategy_id, week_iso, params (dict), dsr, pbo, status,
        allocation_pct, promoted_at, sortino_monthly.
    Returns None on empty result or any BQ error (callers must handle None).
    30s BQ timeout per CLAUDE.md rule.
    """
    if status_filter is None:
        status_filter = ["pending", "active"]
    table = f"{self.settings.gcp_project_id}.pyfinagent_data.promoted_strategies"
    query = f"""
        SELECT
            strategy_id,
            week_iso,
            TO_JSON_STRING(params) AS params_json,
            dsr,
            pbo,
            status,
            allocation_pct,
            promoted_at,
            sortino_monthly
        FROM `{table}`
        WHERE status IN UNNEST(@statuses)
        ORDER BY promoted_at DESC, dsr DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("statuses", "STRING", status_filter),
    ])
    rows = list(self.client.query(query, job_config=job_config).result(timeout=30))
    if not rows:
        return None
    row = dict(rows[0])
    params_raw = row.pop("params_json", None)
    try:
        row["params"] = json.loads(params_raw) if params_raw else {}
    except (json.JSONDecodeError, TypeError):
        row["params"] = {}
    return row
```

## Verbatim Python signature: `load_promoted_params` in `autonomous_loop.py`

```python
def load_promoted_params(bq: BigQueryClient) -> dict:
    """phase-25.B3: load the latest promoted strategy params from BQ.

    Fallback chain:
        1. BQ promoted_strategies (status in ['pending','active'])
        2. optimizer_best.json  (load_best_params fallback)
        3. {}                   (load_best_params in-code default)
    """
    try:
        row = bq.get_latest_promoted_strategy()
        if row and row.get("params"):
            dsr = row.get("dsr", "?")
            params = row["params"]
            logger.info(
                "Loaded promoted params (DSR %s week=%s): %s",
                dsr,
                row.get("week_iso", "?"),
                list(params.keys()),
            )
            return params
        logger.info(
            "No active promoted strategy in BQ, falling back to optimizer_best"
        )
    except Exception as exc:
        logger.warning(
            "Promoted strategy BQ unavailable, falling back to optimizer_best: %s", exc
        )
    return load_best_params()
```

---

## Fallback chain spec

```
load_promoted_params(bq)
  |
  +-- BQ query succeeds AND row returned AND row["params"] non-empty
  |       -> return row["params"]
  |       -> log: "Loaded promoted params (DSR {dsr} week={week_iso}): {list(params.keys())}"
  |
  +-- BQ query succeeds BUT no matching row (empty result)
  |       -> log: "No active promoted strategy in BQ, falling back to optimizer_best"
  |       -> delegate to load_best_params()
  |
  +-- BQ query raises ANY exception (network, timeout, auth, table-not-found)
          -> log: "Promoted strategy BQ unavailable, falling back to optimizer_best: {exc}"
          -> delegate to load_best_params()
                |
                +-- optimizer_best.json exists -> return params dict
                |       -> log: "Loaded best params (Sharpe ...)"  [existing line 42]
                +-- optimizer_best.json missing -> return {}
                        -> log: "optimizer_best.json not found, using defaults"  [existing line 36]
```

---

## Recommended status-filter list (first pass)

**`["pending", "active"]`** -- rationale: 25.A3 writes `status="pending"` at `friday_promotion.py:162`. There is no 25.C3 status-flipper as of this brief. Using `["pending","active"]` ensures rows are immediately visible to the daily loop post-promotion. When 25.C3 lands and flips rows to `"active"`, the filter can be narrowed to `["active"]` only without breaking anything.

---

## Logging line shapes (grep targets for verifier)

```
# BQ path (success):
"Loaded promoted params (DSR"

# BQ path (no rows):
"No active promoted strategy in BQ, falling back to optimizer_best"

# BQ unavailable path:
"Promoted strategy BQ unavailable, falling back to optimizer_best"
```

The verifier's criterion `autonomous_cycle_logs_show_promoted_strategy_loaded` should assert that the success-path log line appears when a mock BQ client returns a valid row.

---

## Files to modify

| File | Change | Notes |
|------|--------|-------|
| `backend/db/bigquery_client.py` | ADD `get_latest_promoted_strategy(self, status_filter: list[str] | None = None) -> dict | None` | Insert after `save_promoted_strategy` method (line 718). Mirror `get_paper_trades_in_window` query structure: `QueryJobConfig` + `result(timeout=30)`. |
| `backend/services/autonomous_loop.py` | ADD `load_promoted_params(bq: BigQueryClient) -> dict` | Insert after `load_best_params` (line 43). No new imports needed (`BigQueryClient` already imported at line 23). |
| `backend/services/autonomous_loop.py` | CHANGE line 100 | `best_params = load_best_params()` -> `best_params = load_promoted_params(bq)`. `bq` is already in scope at line 85. |
| `tests/verify_phase_25_B3.py` | CREATE new verifier | Mirror `verify_phase_25_A3.py` style: static text analysis + `MagicMock` behavioral round-trip. Three criteria matching immutable success criteria verbatim. |

---

## Search queries run (three-variant discipline)

1. **Current-year frontier:** `"runtime parameter hot-swap daily trading loop fallback chain BQ unavailable local JSON 2026"` -- no relevant results  
2. **Last-2-year window:** `"three-tier fallback pattern database unavailable local file in-code defaults Python microservice 2025"` -- fallback pattern architecture confirmed  
3. **Year-less canonical:** `"BigQuery parameterized query Python client latest row status filter strategy registry pattern"` -- canonical GCP docs confirmed; `ArrayQueryParameter` + `ScalarQueryParameter` patterns  
4. **Year-less canonical (BQ syntax):** `"BigQuery latest active row query pattern WHERE status ORDER BY timestamp DESC LIMIT 1 strategy registry"` -- window function alternative confirmed  
5. **Recency scan 2026:** `"Anthropic prompt caching TTL 5 minute cache_control ephemeral best practices 2026"` -- TTL change March 2026 confirmed; once-daily cycle needs no in-process caching  
6. **Strategy state machine:** `"strategy state machine pending active superseded deployment status transition quant trading system 2025"` -- no direct match; resolved by internal code audit of `friday_promotion.py:162`

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched in full)
- [x] 10+ unique URLs total (16 unique URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (`autonomous_loop.py`, `bigquery_client.py`, `friday_promotion.py`, `tests/verify_phase_25_A3.py`, `services/__init__.py`)
- [x] Contradictions / consensus noted (status-filter debate resolved as research call)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
