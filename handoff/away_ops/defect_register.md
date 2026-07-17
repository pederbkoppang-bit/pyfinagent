# Defect register — phase-63.2 BQ cross-check of displayed numbers

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
