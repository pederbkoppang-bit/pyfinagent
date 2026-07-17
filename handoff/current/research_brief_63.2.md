# Research Brief — Step 63.2: BQ cross-check of displayed numbers

**Tier:** moderate | **Gate:** research (pre-contract) | **Started:** 2026-07-17
**Objective (verbatim):** (post-66.2) BQ cross-check of displayed numbers -- for
every number-bearing page (cockpit /, /paper-trading/*, /performance, /learnings,
/sovereign...)

**Deliverable:** `handoff/away_ops/defect_register.md` with displayed-vs-API-vs-BQ
triples (criterion 1) + `| DEF-...` rows for mismatches (criterion 2), all at $0
(BQ + curl only, criterion 3).

---

## STATUS: COMPLETE — gate_passed: true (5 external read-in-full, recency done)

---

## Framing confirmation (the triple)

The page renders API JSON, so **displayed == API is definitional** (frontend just
formats). The meaningful cross-check is **API vs BQ (source of truth)**. Audit design:
for each number-bearing page → API endpoint(s) it calls → raw API JSON value (curl)
→ BQ query that should reproduce it → compare beyond rounding.

---

## Internal code inventory

### $0 curl path (VERIFIED LIVE, non-disruptive)
Auth is enforced at the **middleware** level (`backend/main.py:426-460`
`auth_and_security_middleware`), NOT per-route. Two facts make curl-at-:8000 the
reliable $0 path:
1. `_PUBLIC_PATHS` (main.py:406-423) exempt `/api/sovereign`, `/api/signals`,
   `/api/observability`, `/api/cost-budget`, `/api/health` from auth entirely.
2. **The running backend answers 200 on ALL gated endpoints from a plain
   localhost curl with NO auth header** — verified live 2026-07-17:
   `200 /api/paper-trading/portfolio`, `200 /api/paper-trading/status`,
   `200 /api/reports/performance`, `200 /api/observability/data-freshness`.
   => `DEV_LOCALHOST_BYPASS=1` is active (auth.py:150: env flag + client host in
   127.0.0.1/::1/localhost returns a dev user). No token, no :3100 needed.
   (:3000 returned 302 = healthy authed instance, undisturbed; :3100 down.)

**Recommended $0 path:** `curl -s http://localhost:8000<endpoint> | python3 -m json.tool`
(or `| jq '.path.to.number'`). Read-only GETs only; do NOT hit POST/PUT/DELETE
routes (start/stop/run-now/deposit/risk-limits/kill-switch) — those mutate.

### BQ source-of-truth (dataset `financial_reports`, us-central1)
`backend/db/bigquery_client.py::_pt_table` (:516) => every paper table is
`sunny-might-477607-p8.financial_reports.<name>`:
| BQ table | Read method | Order | Nature |
|---|---|---|---|
| `paper_portfolio` | get_paper_portfolio :521 | single row WHERE portfolio_id='default' | **STORED** `SELECT *` |
| `paper_positions` | get_paper_positions :575 | ORDER BY entry_date DESC | **STORED** `SELECT *` |
| `paper_trades` | get_paper_trades :678 | (DESC created_at) | **STORED** rows |
| `paper_portfolio_snapshots` | get_paper_snapshots :1039 | **ORDER BY snapshot_date DESC** | STORED rows; API RE-COMPUTES metrics from them |

Note the snapshots table is `paper_portfolio_snapshots`, NOT `paper_snapshots`.
The DESC order is the known phase-47.4 trap: any metric computed from snapshots
(Sharpe, maxDD) MUST re-sort chronologically or the sign flips.

### STORED vs COMPUTED (drives the check type)
- **STORED** (API value must EQUAL a single BQ cell — direct compare):
  NAV=`paper_portfolio.total_nav`, cash=`.current_cash`,
  pnl_pct=`.total_pnl_pct`, benchmark=`.benchmark_return_pct`,
  starting_capital=`.starting_capital`; per-position qty/avg_entry_price/
  market_value/unrealized_pnl/current_price/cost_basis/stop_loss_price =
  `paper_positions.<col>`; trade rows = `paper_trades.<col>`.
- **COMPUTED** (API derives from BQ rows — check by RE-DERIVING in SQL OR by
  internal-consistency identity, NOT a single-cell compare):
  - `/portfolio` `sharpe_ratio` (3.56) = `compute_sharpe_from_snapshots(snapshots)`
    — OVERWRITES any stored col (paper_trading.py:226-233).
  - `/portfolio` `sector_breakdown[*].weight_pct` = market_value/total_nav*100.
  - `/performance` `alpha_pct` (14.19) = pnl_pct - benchmark_pct;
    `total_sell_trades`/`total_buy_trades` (30/31) = COUNT(paper_trades by action);
    `days_active` (60) = len(snapshots) = COUNT(paper_portfolio_snapshots);
    `total_analysis_cost` (17.7) = SUM(snapshots.analysis_cost_today);
    `round_trip_summary.*` = pair_round_trips(trades).
  - `/metrics-v2` psr/dsr/sortino/calmar/rolling_sharpe(+CI)/n_obs(59)/
    n_strategies_tested(5) = paper_metrics_v2 from snapshots.
  - Cross-metric consistency identities worth a triple:
    (a) NAV == cash + SUM(positions.market_value) [live: 23214.43+660.13=23874.56 OK];
    (b) pnl_pct == (NAV-starting)/starting*100 [live: 19.37 OK];
    (c) `/portfolio.sharpe_ratio` (3.56) vs `/metrics-v2.rolling_sharpe` (3.0168)
        — DIFFERENT formulas; record both, flag only if same-window mismatch.

### Number-bearing page -> API endpoint -> BQ mapping (the audit worklist)
See the "Application to pyfinagent" table below (also sent to Main).

### Files inspected (anchors)
- frontend/src/lib/api.ts (all fetchers)
- frontend/src/app/page.tsx (cockpit KPIs :387-437; navValue/pnl/benchmark/alpha :278-282)
- frontend/src/app/performance/page.tsx (:115-116 getPerformanceStats+getCostHistory)
- frontend/src/app/learnings/page.tsx (:30 getPaperLearnings)
- frontend/src/app/sovereign/page.tsx (:64/:89/:108 redline/compute-cost/leaderboard)
- backend/main.py:406-460 (auth middleware + _PUBLIC_PATHS)
- backend/api/auth.py:150 (DEV_LOCALHOST_BYPASS)
- backend/api/paper_trading.py:116/171/244/278/294/994/878 (status/portfolio/trades/snapshots/performance/metrics-v2/learnings)
- backend/api/reports.py (/performance recommendation stats — all-zero live)
- backend/db/bigquery_client.py:516/521/575/678/1039 (table refs + read methods)

---

## External research (read in full via WebFetch)

| # | Source | Kind | Key finding |
|---|---|---|---|
| 1 | dqops.com — table comparison checks | vendor doc | Reconcile via row-count, column SUM-vs-reference, min/max/mean, null-count, distinct-count. Uses **tiered severity thresholds NOT exact match**: Warning 0.0%, Error 1.0% relative discrepancy. "Group by common identity columns" before aggregating a detail table vs a summary. Color-code cells by severity. |
| 2 | datafold.com — reconciliation best practices | vendor eng blog | **Value-level** field-by-field beats count-level ("highly misleading") and aggregate-level ("overlooks data-quality issues"). Value-level "pinpoints exactly where the two tables are not matching". PK/uniqueness alignment first. (No explicit float tolerance given.) |
| 3 | e6data.com — BigQuery cost optimization 2025 | practitioner | Avoid `SELECT *` (scan only needed cols); `require_partition_filter=true`; **LIMIT alone does NOT reduce bytes scanned** (engine scans then limits); use `--dry_run` to estimate bytes pre-run; `INFORMATION_SCHEMA` + `TABLESAMPLE` preview for near-zero cost. |
| 4 | oliviac.dev — jq with curl | tutorial | `curl -s URL \| jq '.a.b'` nested dot; `.[0]`/`.[-1]`/`.[2:5]` arrays; `.results \| map({name,url})` project fields; `-r` raw output for a bare scalar. `-s` suppresses curl progress meter (avoids corrupting the pipe). |
| 5 | montecarlo.ai — data reconciliation guide | vendor eng blog | 6 stages: extract -> match (deterministic/probabilistic/fuzzy) -> discrepancy ID (field-by-field) -> correct -> validate (re-run) -> **document**. Distinguish systematic transform diffs (timestamps off 8h = TZ; "decimal places disappear = formatting") from genuine breaks. Some exceptions need human judgment. |

### Key findings applied to this audit
1. **"Beyond rounding" (criterion 2) = a tolerance threshold, not exact equality.**
   DQOps default Error tolerance is 1.0% relative. For a displayed number the
   correct tolerance is the display's own rounding unit (a value shown to 2dp =>
   +/-0.005 absolute for displayed-vs-API; a small relative band ~0.5-1% for
   API-vs-BQ since the API rounds for display while BQ stores full precision).
2. **Value-level is the right granularity** (datafold) — our API-vs-BQ compare
   is a single scalar per number, exactly the recommended level.
3. **$0 is trivially satisfied** (e6data): the paper_* tables are tiny (1
   portfolio row, ~60 snapshots, ~60 trades, 1 position) so even `SELECT *`
   scans << the 1 TB/month free tier. Use the **BQ MCP describe-table** (reads
   metadata, scans 0 bytes) for schema, and SELECT only the needed column for
   values. LIMIT is for row-capping, not cost.
4. **Not every diff is a break** (montecarlo): timezone offsets, display
   formatting, live-price vs stored-snapshot lag are EXPECTED differences —
   record them in the triple but do NOT raise a DEF- row.

## Application to pyfinagent — the audit worklist (page -> number -> API -> BQ)

Legend: **S**=stored (direct single-cell compare) | **C**=computed (re-derive in
SQL or check via identity) | **L**=live-priced (no stored SoT; check identity or
skip). Dataset = `financial_reports` unless noted.

| Page | Number (displayed) | API endpoint -> JSON path | BQ source-of-truth | Type |
|---|---|---|---|---|
| cockpit `/` | NAV | `/api/paper-trading/status` -> `.portfolio.nav` (also `/portfolio` -> `.portfolio.total_nav`) | `paper_portfolio.total_nav` | S |
| `/` | Cash | `/status` -> `.portfolio.cash` | `paper_portfolio.current_cash` | S |
| `/` | Total P&L % | `/portfolio` -> `.portfolio.total_pnl_pct` | `paper_portfolio.total_pnl_pct`; identity (NAV-start)/start*100 | S |
| `/` | vs SPY (benchmark) | `/status` -> `.portfolio.benchmark_return_pct` | `paper_portfolio.benchmark_return_pct` | S |
| `/` | alpha (vs SPY tile) | derived `pnl - benchmark` (page.tsx:282) | identity from the two stored cells | C |
| `/` | Sharpe (90d) | `/portfolio` -> `.portfolio.sharpe_ratio` | `compute_sharpe_from_snapshots(paper_portfolio_snapshots)` | C |
| `/` | Positions count | `/status` -> `.position_count` | `COUNT(*) paper_positions` | C |
| `/` | P&L (today) $/% | LivePortfolio overlay (`lp.pnlTodayDollars`) | live vs yesterday snapshot close — LIVE, not a stored SoT | L |
| `/` | NAV (KPI, live) | LivePortfolio `lp.liveNav` (live-priced) | differs from stored NAV BY DESIGN — cross-check the STORED path via `/status`, not the live tile | L |
| `/paper-trading/positions` | qty, avg_entry_price, cost_basis, stop_loss_price, entry_date, sector | `/portfolio` -> `.positions[]` | `paper_positions.<col>` | S |
| `/positions` | current_price, market_value, unrealized_pnl(_pct) | `/portfolio` -> `.positions[]` | live-priced; identities: cost_basis==qty*avg_entry; market_value==qty*current_price; upnl==mv-cost_basis; upnl_pct==upnl/cost_basis*100 | L/C |
| `/paper-trading/nav` | NAV series points | `/snapshots` -> `.snapshots[]` | `paper_portfolio_snapshots` (DESC — resort) | S |
| `/paper-trading/trades` | trade rows (ticker/action/qty/price/cost/date) | `/trades` -> `.trades[]` | `paper_trades.<col>` | S |
| `/trades` | trade count | `/trades` -> `.count` | `COUNT(*) paper_trades` | C |
| `/paper-trading` (+manage) | psr, dsr, sortino, calmar, rolling_sharpe(+CI), n_obs, n_strategies_tested | `/metrics-v2` | `paper_metrics_v2` recomputed from snapshots | C |
| `/paper-trading` perf | nav, pnl_pct, alpha_pct, sharpe, sell/buy counts, days_active, round_trip_summary.* | `/performance` | portfolio (S) + COUNT/SUM(paper_trades, snapshots) (C) | S/C |
| `/performance` | total_recommendations, wins, losses, avg_return, win_rate, benchmark_beat_rate | `/api/reports/performance` | `outcome_tracking` (dataset `pyfinagent_data`) — **all 0 live** | C |
| `/performance` | cost history points | `/api/reports/cost-history` | llm_call_log / cost rollup | C |
| `/learnings` | reconciliation_divergences ct, kill_switch_triggers ct, regime_buckets | `/api/paper-trading/learnings` | reconciliation + kill-switch audit + snapshots regime — **all empty live** | C |
| `/sovereign` | compute cost daily (anthropic/vertex/bigquery/openai/altdata) + grand_total | `/api/sovereign/compute-cost` (PUBLIC) | llm_call_log / cost aggregation | C |
| `/sovereign` | leaderboard sharpe/dsr/pbo/max_dd/allocation | `/api/sovereign/leaderboard` (PUBLIC) | `strategy_deployments_view` | S(view) |
| `/sovereign` | red-line NAV series | `/api/sovereign/red-line` (PUBLIC) | `paper_portfolio_snapshots` + gap-fill | C |

Live-observed consistency (all held 2026-07-17): NAV 23874.56 == cash 23214.43 +
AMD mv 660.13; pnl 19.37 == (23874.56-20000)/20000; AMD cost_basis 719.93 ==
1.319955*545.42; mv 660.13 == 1.319955*500.11; upnl -59.8 == 660.13-719.93.
Watch item (record both, not an auto-DEF): `/portfolio.sharpe_ratio` 3.56 vs
`/metrics-v2.rolling_sharpe` 3.0168 — different formulas/windows.

## defect_register.md — proposed format (satisfies verification cmd)

The immutable cmd does `grep -c '^| DEF-'`. So DEF rows MUST start `| DEF-`.
Put the criterion-1 TRIPLE TABLE first (rows start `| /route` — NOT matched by
the grep), then the criterion-2 DEF rows. `grep -c` counts only DEF rows (0 is a
valid pass if every triple matches).

```markdown
# Defect Register — step 63.2 displayed-vs-API-vs-BQ cross-check
Generated <date>. $0 method: curl :8000 (localhost, DEV_LOCALHOST_BYPASS) + BQ MCP read.

## Triple table (criterion 1 — every number, matches included)
| route | number | displayed | API | BQ | verdict |
| /paper-trading/positions | AMD.market_value | 660.13 | 660.13 | 660.13 | MATCH |
| / | NAV | 23,874.56 | 23874.56 | 23874.56 | MATCH |
...

## Defects (criterion 2 — mismatches beyond rounding only)
| id | route | number | displayed | API | BQ | severity | repro |
| DEF-001 | /performance | win_rate | 0.0% | 0.0 | 0.63 | HIGH | curl .../reports/performance \| jq .win_rate ; BQ: SELECT ... outcome_tracking |
```

Severity rubric: **CRITICAL** money/risk cell (NAV, cash, stop_loss, P&L) an
operator acts on; **HIGH** perf metric feeding promotion gates (Sharpe, DSR,
alpha, win_rate); **MEDIUM** secondary stat (counts); **LOW** chrome/rounding-
adjacent. Tolerance "beyond rounding" = display rounding unit (2dp => +/-0.005)
for displayed-vs-API; ~0.5-1% relative for API-vs-BQ. TZ/formatting/live-lag
differences are recorded in the triple but are NOT DEF rows (montecarlo).

---

## Recency scan (last 2 years)

Searched 2025-2026 literature on data reconciliation, golden-source verification,
and BigQuery cost. Findings that COMPLEMENT the canonical methodology:
- **e6data BigQuery Cost Optimization 2025** + nicheelab "50 tips (2026)": the
  `SELECT *` / partition-filter / LIMIT-doesn't-cut-bytes guidance is current;
  no change that affects a $0 tiny-table read.
- **Gresham "A golden source of data: has the industry changed its mind?"
  (2025-2026)** + Infoverity "Moving past the golden source": the 2026 debate is
  whether ONE universal golden source is still right when multiple upstreams
  disagree. Relevant framing for us: the paper_* BQ tables ARE the golden source
  for displayed numbers; the audit's job is to prove the API layer doesn't drift
  from them. No methodology supersession — value-level field compare (datafold,
  montecarlo) remains the state of the art.
- **No new finding overturns the plan.** The three-query-variant search surfaced
  no 2025-2026 tool or technique that changes a curl-plus-BQ, value-level,
  tolerance-based reconciliation. The approach is stable canonical practice.

## Search queries run (three-variant discipline)
- Current-year frontier: "data reconciliation methodology golden source ... 2026";
  "BigQuery cost optimization ... 2025".
- Last-2-year window: "reconciliation testing UI displayed value vs database
  source of truth"; "data reconciliation aggregate count checksum field-to-field".
- Year-less canonical: "jq extract nested JSON field from curl API response
  tutorial" (surfaced the canonical jq/curl prior art).

---

## Research Gate Checklist

Hard blockers (all satisfied):
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5)
- [x] 10+ unique URLs total (~40 across 5 searches)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (WebFetch), not abstracts
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every number-bearing page + its API + BQ SoT
- [x] Consistency identities verified live (NAV/cost_basis/mv/upnl all held)
- [x] Claims cited per-claim with URL / file:line

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 35,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Auth is middleware-level; running backend answers 200 on all gated endpoints from a plain localhost curl (DEV_LOCALHOST_BYPASS active, auth.py:150) and /api/sovereign|signals|observability are in _PUBLIC_PATHS -- so curl :8000 with no token is the reliable $0 API leg (:3100 not needed). BQ SoT = financial_reports.paper_portfolio/paper_positions/paper_trades/paper_portfolio_snapshots. get_paper_portfolio is a stored SELECT*; snapshots read DESC (phase-47.4 resort trap). Numbers split STORED (direct cell compare: NAV/cash/pnl/benchmark/qty/avg_entry) vs COMPUTED (re-derive/identity: sharpe, alpha, counts, metrics-v2, sector wts). External sources: value-level field compare (datafold/montecarlo) with a tolerance threshold not exact match (dqops 1% default); 'beyond rounding' = display rounding unit; $0 trivial on tiny tables (e6data: SELECT only needed col, LIMIT doesn't cut bytes, MCP describe=0 bytes). defect_register.md = triple table (rows start '| /route') + DEF rows (start '| DEF-' so grep -c matches only those). gate_passed true.",
  "brief_path": "handoff/current/research_brief_63.2.md",
  "gate_passed": true
}
```

## STATUS: COMPLETE — gate_passed: true
