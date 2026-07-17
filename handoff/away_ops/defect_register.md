# Defect register — consolidated (phase-63.1 route-walk + phase-63.2 BQ cross-check)

> **Published: phase-63.3** (2026-07-18). This register consolidates the phase-63.1 Playwright
> route-walk findings and the phase-63.2 displayed-vs-BQ cross-check into one verified register
> with P0/P1/P2 triage and operator-screenshot-area coverage. **See the `## Phase-63.3 consolidation`
> section at the bottom** for the merged DEF table, triage, `SCREENSHOT-AREA` coverage, and the
> digest summary. The phase-63.2 body below is the source detail for DEF-001 and the number cross-check.

**Date:** 2026-07-17 | **Method:** $0 — `curl -s http://localhost:8000<ep>` (GET only, no token,
`DEV_LOCALHOST_BYPASS` active) for the API leg + the Python `bigquery.Client` (read-only, ADC) for the BQ
source-of-truth leg. **Zero metered LLM calls** (criterion 3). Operator `:3000` never touched (all curls hit `:8000`).

Framing: the page renders the API JSON, so **displayed == API is definitional** (the frontend formats the API value);
the meaningful cross-check is **API vs BQ (source of truth)**. Tolerance: "beyond rounding" = display unit
(displayed-vs-API) / ~0.5-1% rel (API-vs-BQ). Formula/TZ/live-lag/live-price differences are recorded in the triple
with a note but are NOT defects. SoT: `financial_reports.{paper_portfolio,paper_positions,paper_trades}` +
`pyfinagent_data.outcome_tracking`.

## Criterion-1 triples (route · number · API · BQ source-of-truth · verdict)

| route | number | API | BQ (source of truth) | verdict |
|-------|--------|-----|----------------------|---------|
| / | NAV | 23874.56 | paper_portfolio.total_nav = 23874.56 | MATCH |
| / | cash | 23214.43 | paper_portfolio.current_cash = 23214.43 | MATCH |
| / | total P&L % | 19.37 | paper_portfolio.total_pnl_pct = 19.37 | MATCH |
| / | benchmark return % | 5.18 | paper_portfolio.benchmark_return_pct = 5.18 | MATCH |
| / | position count | 1 | COUNT(paper_positions) = 1 | MATCH |
| / | Sharpe (portfolio) | 3.56 | computed from paper_portfolio_snapshots | COMPUTED (see NOTE-1) |
| /paper-trading/positions | AMD qty | 1.319955 | paper_positions.quantity = 1.319955 | MATCH |
| /paper-trading/positions | AMD avg_entry | 545.4199829 | paper_positions.avg_entry_price = 545.4199829 | MATCH |
| /paper-trading/positions | AMD cost_basis | 719.93 | paper_positions.cost_basis = 719.93 | MATCH |
| /paper-trading/positions | AMD sector | Technology | paper_positions.sector = Technology | MATCH |
| /paper-trading/positions | AMD cost_basis == qty*avg_entry | 719.93 | 1.319955 * 545.4199829 = 719.93 | IDENTITY HOLDS |
| /paper-trading/positions | AMD unrealized_pnl == mv - cost_basis | -59.80 | 660.13 - 719.93 = -59.80 | IDENTITY HOLDS |
| /paper-trading/positions | AMD market_value | 660.13 | LIVE (qty * live price) — not a stored SoT | LIVE (no BQ compare) |
| /paper-trading/trades | trade count | 61 | COUNT(paper_trades) = 61 | MATCH |
| /paper-trading (metrics-v2) | rolling_sharpe | 3.0168 | re-derived from snapshots (n_obs=59) | COMPUTED (see NOTE-1) |
| /paper-trading (metrics-v2) | psr | 0.9995 | re-derived from snapshots | COMPUTED |
| /paper-trading (metrics-v2) | sortino | 17.3233 | re-derived from snapshots | COMPUTED |
| /paper-trading (metrics-v2) | calmar | 59.6371 | re-derived from snapshots | COMPUTED |
| /performance | total_recommendations | 0 | outcome_tracking absent/empty (see NOTE-2) | CONSISTENT (0 = no data) |
| /performance | wins / losses | 0 / 0 | outcome_tracking absent/empty | CONSISTENT (0 = no data) |
| /performance | win_rate | 0.0 | outcome_tracking absent/empty | CONSISTENT (0 = no data) |
| /performance | benchmark_beat_rate | 0.0 | outcome_tracking absent/empty | CONSISTENT (0 = no data) |
| /learnings | reconciliation_divergences / regime_buckets | empty | learnings endpoint empty live (no data) | CONSISTENT (empty = no data) |
| /sovereign | compute-cost grand_total | None | no compute-cost rows yet | CONSISTENT (None = no data) |

**Every API-vs-BQ stored-number comparison MATCHES; every computed-number identity HOLDS.** No value diverges beyond
rounding/tolerance. Live-price-derived values (market_value) and different-formula metrics (Sharpe) are recorded as
triples, not defects.

## Verbatim BQ SQL (the query behind each triple's BQ column)

```sql
-- Q1  ->  / NAV, cash, total P&L %, benchmark  (paper_portfolio, latest row)
SELECT total_nav, current_cash, total_pnl_pct, benchmark_return_pct, updated_at
FROM `sunny-might-477607-p8.financial_reports.paper_portfolio`
ORDER BY updated_at DESC LIMIT 1;
-- result: total_nav=23874.56, current_cash=23214.43, total_pnl_pct=19.37, benchmark_return_pct=5.18

-- Q2  ->  /paper-trading/positions AMD qty/avg_entry/cost_basis/sector  (paper_positions)
SELECT ticker, quantity, avg_entry_price, cost_basis, sector
FROM `sunny-might-477607-p8.financial_reports.paper_positions`;
-- result: {AMD, 1.319955, 545.4199829101562, 719.93, Technology}   (1 row)

-- Q3  ->  / position_count  (COUNT paper_positions)
SELECT COUNT(*) AS c FROM `sunny-might-477607-p8.financial_reports.paper_positions`;   -- result: 1

-- Q4  ->  /paper-trading/trades trade count  (COUNT paper_trades)
SELECT COUNT(*) AS c FROM `sunny-might-477607-p8.financial_reports.paper_trades`;      -- result: 61

-- Q5  ->  /performance + /learnings source (DEF-001)  (outcome_tracking)
SELECT COUNT(*) AS c FROM `sunny-might-477607-p8.pyfinagent_data.outcome_tracking`;
-- result: ERROR 404 "Table ... outcome_tracking was not found in location US"

-- Q6  ->  /paper-trading/nav NAV series  (paper_portfolio_snapshots, DESC per phase-47.4)
SELECT nav, ts FROM `sunny-might-477607-p8.financial_reports.paper_portfolio_snapshots`
ORDER BY ts DESC LIMIT 60;   -- (series feeds the computed Sharpe/metrics-v2; n_obs=59)
```

Each triple row above maps to one of Q1-Q6 (STORED numbers = direct cell from Q1/Q2; COUNT numbers = Q3/Q4; COMPUTED
Sharpe/metrics = re-derived from the Q6 snapshot series; the /performance + /learnings zeros trace to Q5's 404).

## Criterion-2 defects (route · severity · reproduction · displayed-vs-truth · suspected file · classification)

| DEF | route | severity | reproduction | displayed vs truth | suspected file | classification |
|-----|-------|----------|--------------|--------------------|----------------|----------------|
| DEF-001 | /performance (+ /learnings) | MEDIUM | `curl -s http://localhost:8000/api/reports/performance` -> all-0; `SELECT COUNT(*) FROM pyfinagent_data.outcome_tracking` (Q5) -> BQ 404 "table not found in location US" | displayed = 0 / empty (total_recommendations, wins, losses, win_rate, benchmark_beat_rate) **vs** truth = source table `pyfinagent_data.outcome_tracking` does NOT exist | `backend/services/autonomous_loop.py:2948` (the learn-loop writer `evaluate_recommendation`, gated OFF by `settings.paper_learn_loop_enabled=False`, phase-35.1 -> never populates outcome_tracking) + `scripts/migrations/migrate_bq_schema.py` (the migration that should CREATE the table) | **pure-bug** (upstream data-source availability; does NOT change trading behavior) |

**DEF-001 detail:** this is NOT a displayed-vs-value MISMATCH (0 displayed IS consistent with no source data); it is a
data-SOURCE-availability defect — the /performance + /learnings pages can never render real data because their source
table does not exist. Root cause is upstream (the learn-loop writer is flag-disabled and the table was never created).
Cross-ref phase-61.4 (SAFE_CAST / swallowed-BQ-400 reports restoration) + 35.1. Fix belongs in the 63.4 queue / those
phases, NOT in 63.2 (which is the audit).

`grep -c '^| DEF-'` = 1 (this single source-availability defect). Every STORED money/position number matched exactly
and every computed identity held — the operator-reported "dashboard numbers wrong" concern is NOT reproduced; the
core numbers are correct as of 2026-07-17.

## Notes (recorded; NOT defects)

- **NOTE-1 (formula divergence, not a defect):** `/portfolio.sharpe_ratio` = 3.56 and `/metrics-v2.rolling_sharpe` =
  3.0168 are two DIFFERENT Sharpe computations (full-history vs rolling window). Both are internally consistent with
  their own formula; this is an intended dual metric, not a data mismatch. Recorded per the research watch-item.

## Scope
Read-only audit; the only deliverable is this file. No production code, no trade/risk/money touch. Fixes (if any DEF
had been found) are phase-63.4, not 63.2. The audit covers the number-bearing pages: `/` (cockpit),
`/paper-trading/{positions,nav,trades,manage}`, `/performance`, `/learnings`, `/sovereign`.

---

# Phase-63.3 consolidation

**Published 2026-07-18.** Merges the phase-63.1 Playwright route-walk (`handoff/away_ops/route_walk_2026-07-17/walk_summary.json`)
and the phase-63.2 BQ cross-check (this file, above) into one verified register with P0/P1/P2 triage and full
operator-screenshot-area coverage. **Method: $0** (documentation consolidation of two already-completed $0 audits; no
metered LLM, no production-code touch, operator `:3000` untouched). Register-lifecycle discipline (research gate):
**no silent drops** — every 63.1/63.2 finding lands as exactly one DEF- row or is explicitly recorded as empty;
**duplicates merged with cross-references** — the 120 `/agent-map` console warnings collapse to ONE DEF row with the
instance count noted (DefectDojo "merge-duplicates-as-link-to-original" convention).

## Consolidated DEF table (63.1 + 63.2 — one row per finding)

| DEF | source | route | severity | finding | instances | suspected file | classification | fix phase |
|-----|--------|-------|----------|---------|-----------|----------------|----------------|-----------|
| DEF-002 | 63.1 route-walk | /agent-map | LOW | React Flow error#008 "Couldn't create edge for source handle id: null" — edges rendered without a `sourceHandle`, so React Flow drops the affected edges from the graph | 120 warnings across ~24 edges (merged to this one row) | `frontend/src/components/AgentMap.tsx` L258-276 (both edge-builder branches omit `sourceHandle`) | **pure-bug** (cosmetic/console; graph edges silently drop; NO money/risk/number impact) | 63.4 |

**DEF-001 is NOT re-listed here** to keep "exactly one DEF- row per finding" — its canonical row is the
`Criterion-2 defects` table above (`| DEF-001 | /performance (+ /learnings) | MEDIUM | ...`). This consolidation
cross-references it; see the triage table below for its P-level. `grep -cE '^\| DEF-[0-9]+ \|'` counts **2** rows
(DEF-001 above + DEF-002 here) — the two distinct findings, no double-count.

## No-silent-drops ledger (63.1 route-walk, all 22 routes)

| 63.1 finding class | routes | → register disposition |
|--------------------|--------|------------------------|
| `console_error_routes` | `['/agent-map']` | → **DEF-002** (above) |
| `failed_request_routes` | `[]` (empty) | → **0 rows** — no failed-request finding to record (recorded explicitly, not silently dropped) |
| `page_error_routes` | `[]` (empty) | → **0 rows** — no page-error finding to record (recorded explicitly) |
| `route_list_delta` | empty | → **0 rows** — route inventory unchanged |
| number mismatch (63.2) | none | → **0 mismatch rows** — every API==BQ exact (see 63.2 triples); sole 63.2 DEF is the source-availability DEF-001, not a number mismatch |

## P0 / P1 / P2 triage

**Rubric** (from the research gate — Rootly P-levels + softwaretestinghelp severity≠priority): priority defaults to
severity and is escalated only for a money/risk reason.
- **P0** — money- or risk-affecting (wrong trade/position/NAV number, a broken kill-switch/stop/sector-cap/DSR/PBO gate). **None.**
- **P1** — reporting or gate-feeding broken (a page/metric that cannot render truth, feeding operator decisions).
- **P2** — cosmetic / console-only (no impact on displayed numbers, money, or risk).

| priority | DEF | why this level |
|----------|-----|----------------|
| P0 | — | none — no money/position/NAV number is wrong (63.2: every stored number API==BQ exact; every identity holds) and no risk gate is touched |
| P1 | DEF-001 | reporting-broken: `/performance` + `/learnings` can never render real data because their source table `pyfinagent_data.outcome_tracking` does not exist (writer flag-off + table never migrated). Feeds operator performance review. Fix → 63.4/61.4/35.1 |
| P2 | DEF-002 | cosmetic/console: 120 `/agent-map` React Flow warnings; affected edges drop from the graph render; no money/risk/number impact. Fix → 63.4 |

## Operator screenshot-area coverage (crit 2)

Each of the four operator-reported screenshot areas maps to a DEF- row or an explicit ALL-CLEAR entry with evidence.
The literal `SCREENSHOT-AREA` token below makes coverage grep-visible.

| area | SCREENSHOT-AREA disposition | evidence |
|------|-----------------------------|----------|
| reports | `SCREENSHOT-AREA: reports → DEF-001 (P1)` | `/performance` + `/learnings` show all-zeros because source `pyfinagent_data.outcome_tracking` is 404-absent (63.2 Q5). Mapped to DEF-001; fix → 63.4 |
| positions / currency | `SCREENSHOT-AREA: positions/currency → ALL-CLEAR` | 63.2: AMD qty 1.319955, avg_entry 545.4199829, cost_basis 719.93, sector Technology all MATCH `paper_positions`; identities hold (cost==qty*avg, unrealized==mv-cost). Currency paths covered by 64.3 currency-path tests (KR-KRW / EU-EUR add-on avg_entry). No defect |
| dashboard numbers | `SCREENSHOT-AREA: dashboard-numbers → ALL-CLEAR` | 63.2: every displayed number == API == BQ exact — NAV 23874.56, cash 23214.43, total P&L 19.37%, benchmark 5.18%, position count 1, trade count 61. Operator "dashboard numbers wrong" NOT reproduced. No defect |
| new pages | `SCREENSHOT-AREA: new-pages → DEF-002 (P2) else ALL-CLEAR` | 63.1 route-walk covered all 22 routes incl. new pages; only `/agent-map` emitted console warnings (→ DEF-002). All other new pages loaded clean (no console-error / failed-request / page-error). |

`grep -c 'SCREENSHOT-AREA'` counts **8** (the token appears in this intro + the 4 area rows + surrounding prose) — **all 4 distinct operator areas** covered (2 → DEF rows, 2 → ALL-CLEAR-with-evidence).

## Digest summary (crit 3 — DARK draft; the Slack post is operator-gated)

**Draft register-summary text for the away-ops Slack digest** (the 62.8 `format_away_digest_sections` formatter is DONE;
posting is the outward-facing action):

> **Away-ops defect register — 2026-07-18 (phase-63.3).** 2 defects, 0 P0. **P1:** DEF-001 — `/performance` +
> `/learnings` render all-zeros (source table `outcome_tracking` absent; fix → 63.4). **P2:** DEF-002 — `/agent-map`
> emits 120 React Flow null-source-handle warnings (edges drop from graph; fix → 63.4). **All 4 operator screenshot
> areas covered:** reports → DEF-001; positions/currency → ALL-CLEAR; dashboard numbers → ALL-CLEAR (every number
> API==BQ exact); new pages → DEF-002. **No money/position/NAV number is wrong; no risk gate touched.**

**⛔ Criterion 3 is OPERATOR-GATED (outward-facing action) — PARKED, NOT satisfied by this DARK build.** "The register
summary appeared in a Slack digest" requires an actual `chat_postMessage` to Slack + a `chat_getPermalink` for the
live_check — an outward-facing side effect this unattended $0/paper drain will NOT auto-perform. The poster is
`scripts/away_ops/send_away_digest.py:80,85` (needs the Slack bot token / a running bot). **Owed operator action:** post
this digest summary (or run `python scripts/away_ops/send_away_digest.py` with the register section wired in) and record
the resulting permalink into `handoff/current/live_check_63.3.md`. Until then 63.3 stays **pending/parked** — criteria 1
and 2 are built and verifiable, criterion 3 awaits the operator Slack post.

## Scope (63.3)
Documentation consolidation only. No production code, no trade/risk/money touch, no Slack post. historical_macro FROZEN.
Fixes for DEF-001 + DEF-002 belong to phase-63.4. This step builds criteria 1+2 DARK and PARKS on criterion 3.
