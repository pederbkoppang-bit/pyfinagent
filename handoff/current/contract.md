# Contract — step 65.3 (US+KR per-market health baseline)

**Phase:** phase-65 | **Step:** 65.3 | **Priority:** P1 | harness_required: true | depends_on: none; post-66.2 (done)
**Cycle:** 1 | Date: 2026-07-18 | **Type:** $0 BQ baseline AUDIT (read-only; produces a baseline doc). live book
untouched; historical_macro FROZEN; NO trade/risk/money touch.

## Research-gate summary (gate PASSED)

Researcher subagent (Agent tool, Opus 4.8 effort:max, $0), brief `research_brief_65.3.md`. Envelope:
**gate_passed=true**, tier=moderate, **7 external sources read in full**, 11 snippet-only, 25 URLs, recency scan, 6
internal files. KEY:
- **Schema trap**: `financial_reports.paper_trades` (61 rows) has **NO `market` column** and `created_at` is a
  **STRING** → derive market from the ticker suffix (`market_for_symbol`, markets.py:142: bare=US, .KS/.KQ=KR,
  .DE/.PA=EU); filter `created_at >= "2026-06-01"` (lexical, works) — NOT `>= TIMESTAMP(...)` (that failed).
- **Definitions**: win = `realized_pnl_pct > 0` on SELL rows (paper_round_trips.py:145; break-even=loss);
  `holding_days` + `realized_pnl_pct` precomputed on SELL rows (no re-pairing); fee = `transaction_cost` = notional ×
  `paper_transaction_cost_pct`(0.1, settings.py:371)/100 (USD).
- **Live dry-run** (to re-run): US 28 trades / 70.6% win / median-hold 3d / $20.30 fees; KR 10 / 20% / 1d / $4.82;
  EU 0.
- **Churn split** (criterion 3): `paper_swap_churn_fix_enabled` operator-activated ON **2026-06-12** (harness_log
  :27097). PRE-FIX = 06-01→06-11 (swap-churn cluster, dominates the aggregates); POST-FIX = 06-12+ (0 churn swaps,
  thin sample confounded by the away-ops quiet period — disclose both causes; post-fix trend PENDING more cycles).
- **Statistical caveat**: min ~30 trades/metric → at US 17 / KR 5 closed, win-rate/PF are DESCRIPTIVE not inferential
  → lean the thresholds on robust STRUCTURAL gates (holding-days, churn-swap-hold, fee-drag, liveness).

## Plan
1. Run the 4 aggregate BQ queries ($0, read-only, Python bigquery client, us-central1) with the market-from-suffix
   CASE + `WHERE created_at >= "2026-06-01"`: (B1) per-market trade counts (buys/sells); (B2) per-market win rate
   (COUNTIF(SELL & realized_pnl_pct>0)/COUNTIF(SELL)); (B3) per-market exit-reason mix (GROUP BY reason + avg
   holding_days + avg realized_pnl_pct); (B4) per-market holding-day distribution (<=1d / 2-5 / 6-20 / >=21 + median).
2. Run the same split by the churn flag date (pre-fix 06-01→06-11 vs post-fix 06-12+).
3. Write **`handoff/away_ops/market_health_baseline.md`**: per-market aggregate tables + **the verbatim SQL pasted**
   (criterion 1) + **≥1 explicit `HEALTHY-THRESHOLD:` lines** (criterion 2, e.g. "no market > X% of NAV in fees",
   "stop-out rate < Y%", median holding-days ≥ Z, churn-swap-hold ≥ 3d) + the **pre/post-61.1-fix split noted
   separately** (criterion 3) + the low-n descriptive caveat.

## Immutable success criteria (VERBATIM from masterplan.json 65.3)
1. "per-market aggregates (trades, win rate, exit reasons, holding days) since 2026-06-01 with the SQL pasted verbatim"
2. "explicit HEALTHY-THRESHOLD lines that 65.4 will be judged against (e.g. no market >X% of NAV in fees, stop-out rate <Y%)"
3. "post-churn-fix (61.1 flags ON) trend noted separately from the pre-fix baseline"

**Verification command (immutable):**
`cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/away_ops/market_health_baseline.md && grep -c 'HEALTHY-THRESHOLD' handoff/away_ops/market_health_baseline.md`

## Boundaries (binding)
$0 — read-only BQ SELECT + Python aggregation. READ-ONLY baseline AUDIT; the only deliverable is
`market_health_baseline.md` (+ live_check). NO production code change; NO trade/risk/money touch;
kill-switch/stops/caps/DSR/PBO untouched; historical_macro FROZEN; live book untouched. The HEALTHY-THRESHOLD lines
are BASELINE targets that 65.4 will judge against (not enforced live). Low-n honesty: win-rate/PF at US 17 / KR 5
closed trades are DESCRIPTIVE, not inferential (disclosed). The post-61.1-fix trend is noted separately + flagged as
pending more cycles (away-ops quiet). SQL pasted verbatim per criterion 1 (lesson from 63.2: copy criteria verbatim).

## References
research_brief_65.3.md; backend/backtest/markets.py:142 (market_for_symbol); backend/services/paper_round_trips.py:145
(win def); backend/config/settings.py:345 (paper_swap_churn_fix_enabled), :371 (paper_transaction_cost_pct); the
financial_reports.paper_trades schema; harness_log.md:27097 (flag ON 2026-06-12).
