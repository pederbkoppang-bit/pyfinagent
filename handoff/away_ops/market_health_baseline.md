# US + KR per-market health baseline (phase-65.3)

**Date:** 2026-07-18 | **Window:** trades since 2026-06-01 | **Source:** `financial_reports.paper_trades` (read-only,
$0). **NAV reference:** $23,874.56. paper_trades has NO `market` column and `created_at` is a STRING → market derived
from the ticker suffix (`market_for_symbol`); dates filtered `created_at >= "2026-06-01"` (lexical).

**Statistical caveat (binding for 65.4):** US has 17 and KR has 5 CLOSED (SELL) trades; the minimum for an
inferential win-rate / profit-factor is ~30 trades/metric. So **win-rate and PF here are DESCRIPTIVE, not
inferential** — 65.4 should judge primarily against the STRUCTURAL thresholds (holding-days, churn-swap-hold,
fee-drag, liveness), and treat win-rate/PF as trend indicators until n≥30.

## Per-market aggregates since 2026-06-01

| market | buys | sells | win rate | total fees (USD) | median hold (d) | fees % of NAV |
|--------|------|-------|----------|------------------|------------------|---------------|
| US | 11 | 17 | 70.6% | $20.30 | 3 | 0.085% |
| KR | 5 | 5 | 20.0% | $4.82 | 1 | 0.020% |
| EU | 0 | 0 | — (no trades yet) | $0.00 | — | 0% |

### Holding-day distribution (SELL rows)

| market | ≤1d | 2-5d | 6-20d | ≥21d |
|--------|-----|------|-------|------|
| US | 6 | 4 | 1 | 6 |
| KR | 4 | 0 | 0 | 1 |

### Exit-reason mix (SELL rows)

| market | reason | n | avg hold (d) | avg realized P&L % |
|--------|--------|---|--------------|--------------------|
| US | swap_for_higher_conviction | 10 | 4.7 | +11.95% |
| US | stop_loss_trigger | 7 | 29.3 | +32.19% |
| KR | stop_loss_trigger | 3 | 8.0 | -0.40% |
| KR | swap_for_higher_conviction | 2 | 0.5 | -3.26% |

## Post-churn-fix (61.1 flags ON) trend, noted SEPARATELY from the pre-fix baseline

`paper_swap_churn_fix_enabled` was operator-activated ON **2026-06-12** (harness_log.md:27097). Segments are presented
separately and NEVER merged:

| phase (churn flag) | market | buys | sells | swap-exits | ≤1d sells |
|--------------------|--------|------|-------|------------|-----------|
| PRE-FIX (06-01→06-11, flag OFF) | US | 9 | 15 | **10** | 6 |
| PRE-FIX (06-01→06-11, flag OFF) | KR | 5 | 4 | **2** | 4 |
| POST-FIX (06-12+, flag ON) | US | 2 | 2 | **0** | 0 |
| POST-FIX (06-12+, flag ON) | KR | 0 | 1 | **0** | 0 |

**Read:** the PRE-FIX window carries the entire swap-churn cluster (12 swap-exits, many ≤1d holds — the churn the
phase-61 audit flagged, e.g. KR swap-exits avg 0.5d hold / -3.26% realized). The POST-FIX window has **0 swap-exits**
(the fix holds) BUT the sample is thin (US 2 sells, KR 1 sell) and CONFOUNDED by the away-ops engine-quiet period — so
the post-fix improvement is DIRECTIONALLY confirmed (0 churn swaps) but the trend is **PENDING more cycles** before it
is statistically meaningful. Do not over-read the post-fix win rate at n≤2.

## HEALTHY-THRESHOLD lines (65.4 will be judged against these)

Structural / robust (primary — valid at any n):
- HEALTHY-THRESHOLD: no market > 0.50% of NAV in cumulative fees over any rolling 30-trade window (0.50% of
  $23,874.56 = **$119.37**). Current: US $20.30 (0.085%), KR $4.82 (0.020%) — PASS.
- HEALTHY-THRESHOLD: per-market median holding_days ≥ 5 for a NON-churn regime (median ≤ 1d signals churn). Current:
  US median 3d (BORDERLINE — driven by the pre-fix swap cluster), KR median 1d (below — pre-fix churn) → the pre-fix
  window is the reason; re-judge on post-fix cycles.
- HEALTHY-THRESHOLD: ≤1d exits < 40% of all SELLs per market. Current: US 6/17 = 35% (PASS), KR 4/5 = 80% (FAIL —
  pre-fix churn).
- HEALTHY-THRESHOLD: swap-exit (`swap_for_higher_conviction`) average holding_days ≥ 3 (a swap should replace a
  conviction, not day-trade). Current: US 4.7d (PASS), KR 0.5d (FAIL — pre-fix). POST-FIX: 0 swap-exits.
- HEALTHY-THRESHOLD: ≥ 1 filled trade per ACTIVE market in the 65.4 proof window (liveness). Current: EU has 0 trades
  (not yet active — a separate zero-trades diagnosis is 65.1/65.2).

Secondary / descriptive-until-n≥30 (trend indicators, NOT pass/fail below 30 closed trades):
- HEALTHY-THRESHOLD: profit_factor ≥ 1.5 per market (consensus floor) — compute via
  `paper_round_trips.summarize`; descriptive at US 17 / KR 5 closed.
- HEALTHY-THRESHOLD: win_rate ≥ 40% per market on ≥ 30 CLOSED trades. Current: US 70.6% (n=17, descriptive), KR 20.0%
  (n=5, descriptive — driven by the pre-fix churn losses).
- HEALTHY-THRESHOLD: POST-FIX swap-exit share ≈ 0 sustained (the churn fix does not regress). Current POST-FIX: 0/4
  sells are swap-exits — PASS directionally, pending sample.

## Verbatim BQ SQL (criterion 1)

The market suffix + date filter used by every query:
```sql
-- market (no market column; derive from the yfinance suffix, market_for_symbol):
CASE WHEN ticker LIKE '%.KS' OR ticker LIKE '%.KQ' THEN 'KR'
     WHEN ticker LIKE '%.DE' OR ticker LIKE '%.PA' OR ticker LIKE '%.AS' OR ticker LIKE '%.F' THEN 'EU'
     ELSE 'US' END AS market
-- date filter (created_at is STRING; lexical compare works for ISO-8601):  WHERE created_at >= '2026-06-01'
```

```sql
-- B1/B2/fees/median  (per-market aggregates)
SELECT <market> AS market,
  COUNTIF(action='BUY') AS buys, COUNTIF(action='SELL') AS sells,
  ROUND(SAFE_DIVIDE(COUNTIF(action='SELL' AND realized_pnl_pct>0), COUNTIF(action='SELL'))*100,1) AS win_rate_pct,
  ROUND(SUM(transaction_cost),2) AS total_fees_usd,
  APPROX_QUANTILES(IF(action='SELL', holding_days, NULL), 2)[OFFSET(1)] AS median_hold_days
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE created_at >= '2026-06-01' GROUP BY market ORDER BY market;

-- B4  holding-day distribution (SELL rows)
SELECT <market> AS market,
  COUNTIF(holding_days<=1) AS le1d, COUNTIF(holding_days BETWEEN 2 AND 5) AS d2_5,
  COUNTIF(holding_days BETWEEN 6 AND 20) AS d6_20, COUNTIF(holding_days>=21) AS ge21
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE created_at >= '2026-06-01' AND action='SELL' GROUP BY market ORDER BY market;

-- B3  exit-reason mix (SELL rows)
SELECT <market> AS market, reason, COUNT(*) AS n,
  ROUND(AVG(holding_days),1) AS avg_hold, ROUND(AVG(realized_pnl_pct),2) AS avg_pnl_pct
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE created_at >= '2026-06-01' AND action='SELL' GROUP BY market, reason ORDER BY market, n DESC;

-- churn split (pre-fix 06-01..06-11 vs post-fix 06-12+)
SELECT CASE WHEN created_at < '2026-06-12' THEN 'PRE_FIX' ELSE 'POST_FIX' END AS phase, <market> AS market,
  COUNTIF(action='BUY') AS buys, COUNTIF(action='SELL') AS sells,
  COUNTIF(action='SELL' AND reason LIKE '%swap%') AS swap_exits,
  COUNTIF(action='SELL' AND holding_days<=1) AS le1d_sells
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE created_at >= '2026-06-01' GROUP BY phase, market ORDER BY phase DESC, market;
```
(`<market>` = the CASE block above; definitions: win = `realized_pnl_pct > 0` on SELL rows, paper_round_trips.py:145;
`holding_days`/`realized_pnl_pct` precomputed on SELL rows; fee = `transaction_cost` = notional × 0.1%,
settings.py:371.)

## Scope
Read-only $0 BQ baseline; the only deliverable is this file. No production code, no trade/risk/money touch. The
HEALTHY-THRESHOLD lines are BASELINE targets for phase-65.4 to judge against (not enforced live). historical_macro
FROZEN; live book untouched.
