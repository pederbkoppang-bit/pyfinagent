# Research Brief — Step 61.4: Learnings + Reports history restoration

**Tier:** moderate (caller-specified)
**Date:** 2026-07-08
**Status:** COMPLETE (gate_passed: true). Durable pre-pay artifact for the future 61.4 contract; step itself starts after 61.3.

## Scope (from caller)

1. SAFE_CAST divergences fix — 74 swallowed BQ 400s since 2026-05-12
2. Error-vs-empty distinction in the learnings API
3. Repo-wide STRING-vs-TIMESTAMP audit on `paper_trades.created_at`
4. Per-ticker score history for the 30D TREND sparkline (ARRAY_AGG vs dedicated endpoint)
5. Sprint-tile WIRE|PRUNE decision

## Internal code inventory

### (1) The failing predicate + swallow site

- `backend/db/bigquery_client.py:955-964` — `get_paper_trades_in_window`.
  Failing predicate at :957:
  `WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @window_days DAY)`.
  `paper_trades.created_at` is **STRING** (schema verified live 2026-07-08:
  `created_at:STRING`; written as `datetime.now(timezone.utc).isoformat()` at
  `bigquery_client.py:489`). STRING >= TIMESTAMP has no operator signature in
  GoogleSQL -> BQ 400.
- Exact error, from rotated log `handoff/logs/backend.log.20260612T104931Z.gz`
  (38 occurrences in that file alone; 0 in `backend.log.20260706T225648Z.gz` —
  endpoint simply not hit during the away window):
  `get_learnings: divergence computation failed: 400 No matching signature for
  operator >= for argument types: STRING, TIMESTAMP`
- Swallow site: `backend/api/paper_trading.py:822-823` — bare
  `except Exception as e: logger.warning(...)` inside `_compute_learnings`;
  falls through to `divergences=[]`, indistinguishable from genuinely-empty.
- Correct adjacent pattern: `get_paper_trades_for_ticker_since`
  (`bigquery_client.py:966-986`) compares `created_at >= @since_iso` with a
  STRING param — ISO-8601 strings sort lexicographically, so string-vs-string
  comparison is correct. Same pattern in `get_paper_trades`
  (`bigquery_client.py:678-704`, `WHERE created_at >= @since` STRING param).
- Cache: `paper:learnings:{window_days}` TTL 300s (`api_cache.py:132`,
  endpoint at `paper_trading.py:878-895`) — 5 min self-expiry; "cache busted"
  in the criterion just means wait/restart.

### (2) Repo-wide STRING-vs-TIMESTAMP audit on paper_trades.created_at

| Site | Predicate | Verdict |
|---|---|---|
| `bigquery_client.py:957` `get_paper_trades_in_window` | `created_at >= TIMESTAMP_SUB(...)` | **BROKEN** (STRING vs TIMESTAMP, the 61.4 target) |
| `bigquery_client.py:693` `get_paper_trades` | `created_at >= @since` (STRING param) | clean |
| `bigquery_client.py:978` `get_paper_trades_for_ticker_since` | `created_at >= @since_iso` (STRING param) | clean |
| `backend/slack_bot/jobs/_production_fns.py:219-227` `make_ledger_fetch_fn` (nightly_outcome_rebuild) | `TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP_SUB(...)` + selects `realized_pnl` | **BROKEN twice over**: paper_trades has NO `timestamp` column and NO `realized_pnl` column (schema has `created_at:STRING`, `realized_pnl_pct:FLOAT`) -> "Unrecognized name" 400, swallowed fail-open at :229-231 (`nightly_outcome_rebuild: BQ fetch fail-open`, returns []) |
| `backend/services/reconciliation.py:70`, `backend/services/paper_round_trips.py:55` | Python-side `_parse_ts(created_at)` sort keys | clean (client-side parse) |
| `backend/db/tickets_db.py:99,397` | SQLite, not BQ | out of scope |

The slack-bot site is a NEW finding beyond the audit basis — same defect
class (broken predicate + swallowed 400 -> silent empty), should be fixed or
explicitly recorded in the 61.4 audit deliverable.

### (3) Learnings endpoint response shape

`_compute_learnings` (`paper_trading.py:783-875`) returns
`{reconciliation_divergences, kill_switch_triggers, regime_buckets,
window_days, collected_at}`. Three sections computed independently, each with
its own try/except swallow (:822, :852). A `divergences_error: str|null` field
(and symmetric `kill_switch_error`) slots naturally into this dict; the
frontend consumer is `VirtualFundLearnings.tsx` (empty-arrays handled
gracefully today — an error field lets it render an error banner instead of
the false empty state). TypeScript shape: `VirtualFundLearningsData` in
`frontend/src/lib/types.ts`.

### (4) 30D TREND sparkline

- Dedup kill site: `get_recent_reports` (`bigquery_client.py:257-282`) —
  `ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analysis_date DESC) ... WHERE rk = 1`
  -> exactly 1 row/ticker.
- Frontend already has the full client-side machinery:
  `reports-columns.tsx:110-124` (trend column, renders em-dash when
  `hist.length < 2`) + `buildTickerHistory` (:132-150, groups by ticker, sorts
  by `analysis_date` asc, takes last 30 `final_score`s). It never gets >=2
  rows/ticker because the API dedups.
- Live table sizes (BQ, 2026-07-08): `analysis_results` 389 rows / 44.86 MB;
  8+ tickers have >=2 analyses (SNDK 37, MU 36, INTC 26, STX 25, AMD 24,
  NVDA 24, DELL 21, 009150.KS 20). ARRAY_AGG cost is negligible at this
  scale — the scan touches only ticker/analysis_date/final_score columns
  (columnar), well under any cost threshold.
- `analysis_date` is TIMESTAMP in `analysis_results` (verified) — no
  STRING-comparison hazard on this table.

### (5) Sprint tile

- Display: `frontend/src/components/HarnessSprintTile.tsx` (read-only,
  fetches nothing; renders permanent "No sprint activity yet" empty state).
- Data chain: `HarnessDashboard.tsx:198-249` -> `GET /api/harness/sprint-state`
  (`backend/api/harness_autoresearch.py:227-229`) -> reads
  `pyfinagent_data.harness_learning_log` (`_BQ_TABLE` at :27) -> table does
  NOT exist -> fail-open at :199 -> returns null -> tile shows empty state
  forever.
- Writer with zero production callers: `log_slot_usage` at
  `backend/autoresearch/slot_accounting.py:30` (NOT `backend/services/` as the
  audit basis says — path drift). Only caller:
  `scripts/harness/phase10_slot_accounting_test.py` (test script).
- Schema creator: `backend/backtest/learning_schema.py:33`
  `create_learning_log_table(dataset_id="trading", ...)` — note default
  dataset is `trading` (doesn't exist); a WIRE decision must pass
  `dataset_id="pyfinagent_data"` to match the reader.
- Legacy dead writers pinned at the nonexistent `trading` dataset:
  `backend/backtest/learning_logger.py:70`
  (`f"{project_id}.trading.harness_learning_log"`) and
  `backend/autonomous_loop.py:73,86` (`dataset_id: str = "trading"`,
  `self.learning_table = f"{project_id}.{dataset_id}.harness_learning_log"`).
- Goal-file recommendation (goal_phase61_churn_integrity.md:181-182):
  **PRUNE** — never-wired scaffolding; stress-test doctrine.

### (6) CommandPalette + learnings route

- `frontend/src/components/CommandPalette.tsx:55` links
  `/paper-trading/learnings` (the legacy path); the real page moved to root
  `/learnings` in phase-73 (2026-05-26) — see header comment in
  `frontend/src/app/learnings/page.tsx:3-11`. `Sidebar.tsx:44` already links
  `/learnings`. A `frontend/src/app/paper-trading/learnings/` directory still
  exists (the redirect). Fix = one-line href change.

### (7) Timing dependency on the >=10-divergences criterion (NEW, contract-relevant)

Live BQ 2026-07-08: paper_trades has 58 rows all-time; only **14 trades and 8
round-trip SELLs fall in the current 30-day window** (newest trade
2026-07-03T18:15Z — trading halted by the credential death; see
return-day memory). `pair_round_trips` FIFO-matches BUY->SELL within the
fetched window, so an in-window SELL whose BUY predates the window does NOT
pair — the fixed query would yield **at most ~8 divergences today, below the
immutable >=10 floor**. 61.4 depends on 61.1-61.3 anyway; the contract should
note the criterion becomes satisfiable only after reactivation (phase-66)
accrues enough in-window round trips, or verify with a wider window during
development while keeping the immutable criterion for the final live_check.

## External sources

Query variants run (per research-gate rules): (1) year-less canonical —
"BigQuery SAFE_CAST STRING TIMESTAMP comparison no matching signature
pitfalls", "sparkline data shape API design time series per row table",
"Google AIP-193 error model partial failure"; (2) last-2-year — "API design
error vs empty result RFC 9457 problem details 2025", "BigQuery timestamp
comparison implicit coercion literal 2024 2025 breaking change"; (3)
current-year — "BigQuery ARRAY_AGG ORDER BY LIMIT per-group latest N rows
pattern cost 2026".

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://cloud.google.com/bigquery/docs/reference/standard-sql/conversion_rules | 2026-07-08 | official doc | curl + text extraction (WebFetch got nav-only; page is JS-heavy) | Coercion table: STRING expressions (columns) have NO coerce-to targets — "The Coerce to column applies to all expressions of a given data type... but literals and parameters can also be coerced." STRING **literal** -> DATE/DATETIME/TIME/TIMESTAMP; STRING **parameter** -> DATE/DATETIME/TIME/TIMESTAMP. A STRING *column* never implicitly coerces to TIMESTAMP -> the :957 predicate has no operator signature -> 400. |
| https://cloud.google.com/bigquery/docs/reference/standard-sql/conversion_functions | 2026-07-08 | official doc | curl + text extraction | "SAFE_CAST replaces runtime errors with NULLs. However, during static analysis, impossible casts between two non-castable types still produce an error." Pitfall: NULL result in a WHERE predicate silently DROPS the row — malformed created_at strings vanish without trace. Page last updated 2026-07-07. |
| https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions | 2026-07-08 | official doc | curl + text extraction | `ARRAY_AGG([DISTINCT] expression [{IGNORE|RESPECT} NULLS] [ORDER BY key [{ASC|DESC}][,...]] [LIMIT n])` — ORDER BY + LIMIT live INSIDE the aggregate; "An error is raised if an array in the final query result contains a NULL element" -> must use IGNORE NULLS on final_score. LIMIT n keeps the first n by the ORDER BY -> `ORDER BY analysis_date DESC LIMIT 30` yields newest-30 in DESC order (must reverse for chronological sparkline). |
| https://www.rfc-editor.org/rfc/rfc9457.html | 2026-07-08 | standard (obsoletes RFC 7807) | WebFetch full | All five members optional; "truly generic problems... are usually better expressed as plain status codes"; extension members: "Clients consuming problem details MUST ignore any such extensions that they don't recognize". Applies to full-request failure (4xx/5xx), not partial-section failure inside a 200. |
| https://google.aip.dev/193 | 2026-07-08 | official design guide | WebFetch full | "APIs should not support partial errors... [but] occasionally partial errors are necessary, particularly in bulk operations where it would be hostile to users to fail an entire large request because of a problem with a single entry" — the learnings endpoint (3 independent sections) is exactly this aggregate case; per-section failure info in the response body is the sanctioned exception. |
| https://swagger.io/blog/problem-details-rfc9457-doing-api-errors-well/ | 2026-07-08 | authoritative blog | WebFetch full | Anti-pattern #3: "Hiding Errors in Successful Responses... will mislead consumers into believing a request was successful when it wasn't" — the current swallowed-400 behavior is this anti-pattern verbatim; the fix must make section failure EXPLICIT in the payload (or fail the request). "Using extensions would be recommended over asking a client to parse the detail property." |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/metabase/metabase/issues/66880 + /48010 + /8932, https://discourse.metabase.com/t/21241 | community | Confirm the exact "No matching signature" error class in the wild; official doc supersedes |
| https://datatracker.ietf.org/doc/html/rfc7807 | standard (obsoleted) | Superseded by RFC 9457 (read in full) |
| https://www.codecentric.de/.../rfc-7807-and-rfc-9457, https://zuplo.com/learning-center/best-practices-for-api-error-handling, https://www.speakeasy.com/api-design/errors, https://www.eke.li/dotnet/2025/09/26/problem-details.html | blogs | RFC + swagger blog cover the same ground |
| https://medium.com/data-engineers-notes/using-array-agg-in-bigquery-8f178031bd8b, https://datawise.dev/using-array-agg-in-bigquery, https://count.co/sql-resources/bigquery-standard-sql/array-agg, https://www.owox.com/blog/articles/bigquery-aggregate-functions, hevodata.com, getorchestra.io | blogs | Official aggregate_functions doc read in full instead; snippets note 10,000-element array cap |
| https://www.owox.com/blog/articles/bigquery-cast-and-safe-cast (2025), /bigquery-timestamp-functions (2025), https://yukidata.com/bigquery-data-types/, https://www.secoda.co/learn/type-casting-in-bigquery, https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/data-types, /lexical | doc/blog | Coercion semantics fully covered by conversion_rules |
| https://www.highcharts.com/docs/grid/sparklines, https://dash.plotly.com/dash-ag-grid/enterprise-sparklines, https://learn.microsoft.com/en-us/power-bi/create-reports/power-bi-sparklines-tables, https://www.domo.com/learn/charts/sparkline-chart, https://www.visualizing.org/sparkline | vendor docs | Data-shape consensus visible in snippets: one ordered numeric array per row, 5-10+ points for a recognizable shape, delivered as a per-row array field (AG Grid/Highcharts cellRenderer pattern) — matches the existing `buildTickerHistory` contract |
| https://google.aip.dev/151, /234, https://aep.dev/193, https://github.com/aip-dev/google.aip.dev/issues/1615, https://developers.google.com/google-ads/api/docs/best-practices/understand-api-errors | official | AIP-193 read in full; these corroborate |
| https://github.com/cube-js/cube/issues/10643, https://community.powerbi.com/t5/.../2118367, https://count.co/sql-resources/bigquery-standard-sql/dates-and-times, https://github.com/duckdb/duckdb/discussions/8151 | community | Same error class corroboration |

URLs collected: 35+ unique.

### Recency scan (2024-2026)

Performed. Findings: (1) RFC 9457 (2023) obsoletes RFC 7807 — any new
error-shape work should cite 9457, not 7807 (the caller's prompt said 7807;
9457 is the current number, semantics near-identical plus a problem-type
registry). (2) BigQuery conversion/coercion rules show NO 2024-2026 breaking
change — the conversion_rules and conversion_functions pages are current as
of 2026-07-07 and the STRING-column-never-coerces rule is longstanding and
stable. (3) ARRAY_AGG signature unchanged; 2025-2026 vendor guides (OWOX)
add nothing beyond the official doc. (4) 2025 practitioner writing (Swagger,
Speakeasy, eke.li Sept-2025) converges on the same anti-pattern language:
never mask failures inside 200-with-empty-body responses.

## Key findings

1. **Mechanism of the 400 is a coercion-scope rule.** GoogleSQL coerces
   STRING *literals* and STRING *query parameters* to TIMESTAMP, but never
   STRING *column expressions* (conversion_rules coercion table). So
   `created_at >= TIMESTAMP_SUB(...)` (column vs TIMESTAMP) has no operator
   signature -> `400 No matching signature for operator >= for argument
   types: STRING, TIMESTAMP` — exactly the logged error. The adjacent
   "clean" helpers work because both sides are STRING (column vs STRING
   param, no coercion involved). (Source: conversion_rules, 2026-07-08.)
2. **SAFE_CAST's failure mode is silent row-drop, not error.** SAFE_CAST
   returns NULL on runtime conversion failure; NULL in a comparison is
   NULL -> row filtered out. Acceptable here (all 58 rows share one writer,
   `bigquery_client.py:489` isoformat) but the fix should carry a test with
   a malformed created_at row documenting the drop semantics. (Source:
   conversion_functions.)
3. **Per-section error fields are the sanctioned design for aggregate
   endpoints.** AIP-193's bulk-operation exception + Swagger's
   "hiding errors in successful responses" anti-pattern both point to:
   200 + explicit `divergences_error` field when one of three independent
   sections fails; reserve RFC 9457 problem+json for whole-request failures.
4. **ARRAY_AGG with in-aggregate ORDER BY/LIMIT is the right per-ticker
   history tool, with two traps:** NULL elements raise an error (use IGNORE
   NULLS) and `ORDER BY analysis_date DESC LIMIT 30` returns newest-first
   (reverse before feeding the sparkline — cf. the project's known
   DESC-order trap in metric paths). Cost at 389 rows / 3 columns is
   negligible.
5. **Sparkline data-shape consensus:** one ordered numeric array per row
   (oldest -> newest), >=5-10 points for a recognizable shape; per-row array
   field in the existing list response (AG Grid / Highcharts pattern), not a
   separate per-ticker fetch.

## Recommended fix design per criterion

### C1 — predicate fix (get_paper_trades_in_window)

Use the step-name-pinned SAFE_CAST form, keeping the TIMESTAMP param:
`WHERE SAFE_CAST(created_at AS TIMESTAMP) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @window_days DAY)`
(plus `ORDER BY created_at DESC` unchanged — lexicographic = chronological
for uniform ISO strings). Alternative satisfying the same immutable
criterion: compute the cutoff ISO in Python and compare STRING-to-STRING
param, matching the :693/:978 idiom. Either way, add a unit test with a
malformed created_at row (SAFE_CAST drop semantics) and name the test file
`test_phase_61_4_learnings.py` so the immutable `-k 'learnings or
paper_trades_window or 61_4'` selector actually matches (cf. the phase-59.1
false-green `-k` lesson).

### C2 — error-vs-empty

Add `divergences_error: str | None` (and symmetric `kill_switch_error`) to
the `_compute_learnings` return dict; set from the except arms at
`paper_trading.py:822/:852`. **Do NOT cache a result with a non-null error
field** — today a failed compute is cached 300s (`paper:learnings:*`,
`api_cache.py:132`); skip `cache.set` on error so recovery is immediate.
Frontend: extend `VirtualFundLearningsData` in `types.ts`; render the
standard rose error banner in `VirtualFundLearnings.tsx` when non-null
(distinct from the existing empty state). Design basis: AIP-193 aggregate
exception + Swagger anti-pattern #3; RFC 9457 stays the shape for
whole-request 5xx.

### C3 — repo-wide audit

Table in Internal (2). One additional broken site found beyond the audit
basis: `_production_fns.py:219-227` (nightly_outcome_rebuild) references
nonexistent `timestamp` and `realized_pnl` columns -> permanent swallowed
400 -> outcome_tracking never rebuilt. Fix = `created_at` via SAFE_CAST +
`realized_pnl_pct`, or record an explicit clean-bill/defer decision in the
step's audit deliverable. All other sites clean (verdicts in the table).

### C4 — per-ticker score history

Extend `get_recent_reports` (single query, no new endpoint — 389 rows makes
a dedicated endpoint over-engineering):

```sql
WITH ranked AS (... existing rk=1 CTE ...),
hist AS (
  SELECT ticker,
         ARRAY_AGG(final_score IGNORE NULLS ORDER BY analysis_date DESC LIMIT 30) AS score_history_desc
  FROM `{reports_table}` GROUP BY ticker
)
SELECT r.*, h.score_history_desc FROM ranked r JOIN hist h USING (ticker)
WHERE rk = 1 ORDER BY analysis_date DESC LIMIT @limit
```

Reverse to chronological in Python (`list(reversed(...))`) before returning
`score_history: list[float]`. Add the field to `ReportSummary` (backend
model + `types.ts`); in `reports-columns.tsx` build `tickerHistory` from
`r.score_history` instead of the 1-row-per-ticker list (`buildTickerHistory`
:132-150 becomes a trivial map or is retired). 8+ tickers already have >=2
analyses (SNDK 37, MU 36, ...) so sparklines render immediately on live data.

### C5 — sprint tile WIRE|PRUNE

Recommendation: **PRUNE** (matches goal-file recommendation :181-182 —
never-wired scaffolding, stress-test doctrine). PRUNE removal set:
`HarnessSprintTile.tsx` + `.test.tsx`, `HarnessDashboard.tsx` import/fetch/
render (:18, :198, :211, :217, :249), `backend/api/harness_autoresearch.py`
sprint-state endpoint + router registration, `backend/autoresearch/
slot_accounting.py` writer, `scripts/harness/phase10_slot_accounting_test.py`,
legacy dead writers (`backend/backtest/learning_logger.py:70`,
`backend/autonomous_loop.py:73,86`, `backend/backtest/learning_schema.py` if
nothing else imports it). If WIRE: `create_learning_log_table` MUST be
called with `dataset_id="pyfinagent_data"` (its default `"trading"` at
learning_schema.py:35 points at a nonexistent dataset and would recreate the
split-brain). Operator token verbatim, recorded in live_check_61.4.md.

### C6 — CommandPalette

One-line fix: `CommandPalette.tsx:55` href `/paper-trading/learnings` ->
`/learnings` (page moved in phase-73, 2026-05-26; Sidebar already correct).

### C7 — Playwright captures + timing

Standard :3100 skip-auth workflow (frontend.md "Live-UI verification").
**Timing dependency (flag in the contract):** live BQ today shows only 14
trades / 8 round-trip SELLs in the 30-day window (halt since 2026-07-03) —
the immutable ">=10 divergences" check needs post-reactivation (phase-66)
trades to accrue, or the step must run late enough in the window. Do not
soften the criterion; sequence the step.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (6: 3 official BQ docs
  via curl full-text extraction after WebFetch returned nav-only, RFC 9457,
  AIP-193, Swagger practice blog)
- [x] 10+ unique URLs total (35+)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered all five scope areas + live BQ evidence
- [x] Contradictions noted (RFC 9457 whole-request vs AIP-193 per-section —
  resolved: aggregate endpoint qualifies for AIP's bulk exception)
- [x] Per-claim citations inline

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 29,
  "urls_collected": 35,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "report_md": "handoff/current/research_brief_61.4.md",
  "gate_passed": true
}
```
