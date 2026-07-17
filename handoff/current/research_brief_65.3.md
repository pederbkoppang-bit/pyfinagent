# Research Brief — Step 65.3 (US+KR health baseline)

Tier: **moderate**. Research gate for the `market_health_baseline.md` deliverable.
Write-first; appended incrementally as sources/queries land.

## Step objective (verbatim from masterplan.json 65.3)
"(post-66.2) US+KR health baseline -- BQ per-market trade counts, win rate,
exit-reason mix, holding-day distribution since 2026-06-01, written to
handoff/away_ops/market_health_baseline.md with explicit 'healthy' thresholds
for the 65.4 proof."

## Immutable success criteria (VERBATIM — masterplan.json 65.3 verification.success_criteria)
1. "per-market aggregates (trades, win rate, exit reasons, holding days) since 2026-06-01 with the SQL pasted verbatim"
2. "explicit HEALTHY-THRESHOLD lines that 65.4 will be judged against (e.g. no market >X% of NAV in fees, stop-out rate <Y%)"
3. "post-churn-fix (61.1 flags ON) trend noted separately from the pre-fix baseline"

Immutable verification command:
`cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/away_ops/market_health_baseline.md && grep -c 'HEALTHY-THRESHOLD' handoff/away_ops/market_health_baseline.md`
Deliverable = `handoff/away_ops/market_health_baseline.md` with >=1 `HEALTHY-THRESHOLD` line.

---

## A. paper_trades schema (financial_reports.paper_trades) — LIVE, 61 rows

| Column | Type | Mode | Notes |
|---|---|---|---|
| trade_id | STRING | REQUIRED | PK |
| ticker | STRING | REQUIRED | **suffix encodes market** (bare=US, .KS/.KQ=KR, .DE/.PA=EU). NO `market` column. |
| action | STRING | REQUIRED | BUY / SELL |
| quantity | FLOAT | REQUIRED | |
| price | FLOAT | REQUIRED | LOCAL currency (EUR for .DE, KRW for .KS) |
| total_value | FLOAT | NULLABLE | |
| transaction_cost | FLOAT | NULLABLE | **fee/commission per trade** (source for %-of-NAV-in-fees threshold) |
| reason | STRING | NULLABLE | **exit-reason** on SELL rows (stop_loss/take_profit/signal/swap...) |
| analysis_id | STRING | NULLABLE | |
| risk_judge_decision | STRING | NULLABLE | |
| created_at | STRING | REQUIRED | **STRING, not TIMESTAMP** — ISO-8601 text. Filter with string compare or SAFE_CAST. |
| round_trip_id | STRING | NULLABLE | pairs BUY→SELL; set on the closing (SELL) row |
| holding_days | INTEGER | NULLABLE | **precomputed** entry→exit days on the SELL/close row |
| realized_pnl_pct | FLOAT | NULLABLE | **precomputed** realized P&L % on the SELL/close row → win = >0 |
| mfe_pct | FLOAT | NULLABLE | max favorable excursion |
| mae_pct | FLOAT | NULLABLE | max adverse excursion |
| capture_ratio | FLOAT | NULLABLE | |
| signals | STRING | NULLABLE | JSON blob |

### created_at date filter (the STRING trap)
`created_at >= TIMESTAMP("2026-06-01")` FAILS ("No matching signature for
operator >= for argument types: STRING, TIMESTAMP"). created_at is STRING.
Two safe options:
- **String compare** (works because ISO-8601 sorts lexically): `WHERE created_at >= "2026-06-01"`
- **Explicit cast** (robust if format varies): `WHERE SAFE_CAST(created_at AS TIMESTAMP) >= TIMESTAMP("2026-06-01")`
Prefer SAFE_CAST for aggregation math; string compare is fine for the >= floor.

### per-market derivation (NO market column)
Derive from ticker suffix in SQL:
```sql
CASE
  WHEN ENDS_WITH(ticker, '.KS') OR ENDS_WITH(ticker, '.KQ') THEN 'KR'
  WHEN ENDS_WITH(ticker, '.DE') OR ENDS_WITH(ticker, '.PA') THEN 'EU'
  ELSE 'US'
END AS market
```
Verified against `backend/backtest/markets.py:142 market_for_symbol(symbol)` —
"the suffix IS the source of truth": bare→US, `.KS/.KQ`→KR, `.DE/.PA/.AS/.F`→EU,
`.OL`→NO, `.ST`→SE, `.TO`→CA. The SQL CASE above reproduces US/KR/EU exactly
(the only markets with live trades). NB: `market` column on `paper_portfolio` is
a separate hardcoded field ('US') — do NOT use it for per-trade market; derive
from the ticker suffix.

---

## B. The 4 aggregate SQL queries (criterion 1 — "SQL pasted verbatim")

All queries derive `market` from the ticker suffix and filter
`created_at >= "2026-06-01"` (string compare; ISO-8601 sorts lexically).
Live results shown are from a $0 dry-run 2026-07-18 (61-row table).

### B1. Per-market trade COUNT
```sql
SELECT
  CASE WHEN ENDS_WITH(ticker,'.KS') OR ENDS_WITH(ticker,'.KQ') THEN 'KR'
       WHEN ENDS_WITH(ticker,'.DE') OR ENDS_WITH(ticker,'.PA') THEN 'EU'
       ELSE 'US' END AS market,
  COUNTIF(action='BUY')  AS buys,
  COUNTIF(action='SELL') AS sells,
  COUNT(*)               AS total_trades
FROM `financial_reports.paper_trades`
WHERE created_at >= '2026-06-01'
GROUP BY market ORDER BY market;
```
Result: US buys=11 sells=17 total=28 | KR buys=5 sells=5 total=10 | EU 0 (no EU trades yet).

### B2. Per-market WIN RATE (win = realized_pnl_pct > 0 on the closing SELL row)
```sql
SELECT
  CASE WHEN ENDS_WITH(ticker,'.KS') OR ENDS_WITH(ticker,'.KQ') THEN 'KR'
       WHEN ENDS_WITH(ticker,'.DE') OR ENDS_WITH(ticker,'.PA') THEN 'EU'
       ELSE 'US' END AS market,
  COUNTIF(action='SELL')                            AS closed_trades,
  COUNTIF(action='SELL' AND realized_pnl_pct > 0)   AS wins,
  ROUND(SAFE_DIVIDE(
    COUNTIF(action='SELL' AND realized_pnl_pct > 0),
    COUNTIF(action='SELL')), 4)                      AS win_rate
FROM `financial_reports.paper_trades`
WHERE created_at >= '2026-06-01'
GROUP BY market ORDER BY market;
```
Result: US 12/17 = **0.7059** | KR 1/5 = **0.20** | EU n/a.
NB: closed-since-06-01 includes round-trips whose BUY predates 06-01 (April
US winners closed 06-01/06-05) — this is "realized since 06-01", the natural
P&L-booking read. Win definition mirrors `paper_round_trips.summarize()`:145
(`realized_pnl_pct > 0` is a win; exactly 0 counts as a loss).

### B3. Per-market EXIT-REASON mix (SELL rows)
```sql
SELECT
  CASE WHEN ENDS_WITH(ticker,'.KS') OR ENDS_WITH(ticker,'.KQ') THEN 'KR'
       WHEN ENDS_WITH(ticker,'.DE') OR ENDS_WITH(ticker,'.PA') THEN 'EU'
       ELSE 'US' END AS market,
  reason,
  COUNT(*)                       AS n,
  ROUND(AVG(holding_days),1)     AS avg_holding_days,
  ROUND(AVG(realized_pnl_pct),2) AS avg_pnl_pct
FROM `financial_reports.paper_trades`
WHERE created_at >= '2026-06-01' AND action='SELL'
GROUP BY market, reason ORDER BY market, n DESC;
```
Result:
- US: swap_for_higher_conviction 10 (avg_hd 4.7, avg_pnl +11.95%); stop_loss_trigger 7 (avg_hd 29.3, avg_pnl **+32.19%** — trailing-stop profit-taking, NOT losses)
- KR: stop_loss_trigger 3 (avg_hd 8.0, avg_pnl -0.4%); swap_for_higher_conviction 2 (avg_hd 0.5, avg_pnl -3.26%)
CRITICAL nuance: `stop_loss_trigger` is a TRAILING stop → US stop exits are at
big gains. A high stop-out rate is NOT unhealthy here; short-hold SWAP exits are
the churn signal.

### B4. Per-market HOLDING-DAY distribution (`holding_days` precomputed on SELL rows)
```sql
SELECT
  CASE WHEN ENDS_WITH(ticker,'.KS') OR ENDS_WITH(ticker,'.KQ') THEN 'KR'
       WHEN ENDS_WITH(ticker,'.DE') OR ENDS_WITH(ticker,'.PA') THEN 'EU'
       ELSE 'US' END AS market,
  COUNTIF(holding_days <= 1)              AS d_0_1,
  COUNTIF(holding_days BETWEEN 2 AND 5)   AS d_2_5,
  COUNTIF(holding_days BETWEEN 6 AND 20)  AS d_6_20,
  COUNTIF(holding_days >= 21)             AS d_21plus,
  ROUND(AVG(holding_days),1)              AS avg_holding_days,
  APPROX_QUANTILES(holding_days, 2)[OFFSET(1)] AS median_holding_days
FROM `financial_reports.paper_trades`
WHERE created_at >= '2026-06-01' AND action='SELL'
GROUP BY market ORDER BY market;
```
Result: US <=1d:6 / 2-5d:4 / 6-20d:1 / 21+d:6 (avg 14.8, median 3) — BIMODAL
(churn cluster + long winners). KR <=1d:4 / 21+d:1 (avg 5.0, median 1) — churn-dominated.

---

## C. Win-rate & holding-day definitions (which columns)

- **WIN**: `realized_pnl_pct > 0` on the closing SELL row. Source of truth =
  `backend/services/paper_round_trips.py:144-146`: `win_rate = wins/n`, `wins =
  [rt for rt if realized_pnl_pct > 0]`, `losses = <= 0` (break-even is a loss).
  `realized_pnl_pct` is precomputed on the SELL row (FLOAT).
- **HOLDING DAYS**: `holding_days` INTEGER, precomputed on the SELL row
  (entry→exit `.days`). The canonical pairing is FIFO in
  `paper_round_trips.pair_round_trips()` (recomputes from `created_at`); the
  SELL-row `holding_days` column is the same quantity, so the $0 SQL above can
  read it directly. `median_holding_days` uses the sorted-middle convention
  (paper_round_trips.py:153-154). No round-trip re-pairing needed for the
  baseline — the precomputed columns suffice.
- **FEES**: `transaction_cost` FLOAT per row = `notional * paper_transaction_cost_pct/100`
  (paper_trader.py:208 BUY / :443 SELL), `paper_transaction_cost_pct = 0.1`
  (settings.py:371 → 0.1% of notional). Stored in **USD** (SELL side multiplies
  by `_l2u` local→USD FX, :472) so fees are cross-market comparable.

---

## D. Proposed HEALTHY-THRESHOLD lines (criterion 2) — with justification

NAV = **$23,874.56** (paper_portfolio 2026-07-16; start $20k, +19.37%).
Fees since 06-01: US $20.30 + KR $4.82 = **$25.12 = 0.105% of NAV**
(US 0.085%, KR 0.020%). Cumulative all-time fees ~$50 / 61 trades.

Proposed lines (baseline defaults — 65.4 judged against these):
- `HEALTHY-THRESHOLD: no single market > 0.50% of NAV in cumulative fees over any rolling 30-trading-day window` (current US 0.085%, KR 0.020% — comfortably inside; 0.5% ≈ ~5x current, catches a churn blow-out without flagging normal activity).
- `HEALTHY-THRESHOLD: per-market median holding_days >= 5 (NOT churn)` — anchors on the churn finding (short <=2-day holds realized -$139.83 vs +$1355 on long holds). US median 3 / KR median 1 currently FAIL → flags the pre-fix churn baseline, which is the point.
- `HEALTHY-THRESHOLD: per-market share of exits at holding_days <= 1 < 40%` — US 6/17=35% (pass), KR 4/5=80% (fail) → KR churn flagged.
- `HEALTHY-THRESHOLD: per-market win_rate >= 40%` on >= 5 closed trades (small-n caveat) — US 70.6% pass, KR 20% fail. 40% is defensible for a trend/momentum system where profit-factor (avg_win >> avg_loss) matters more than hit-rate; below 40% with negative expectancy is the alarm.
- `HEALTHY-THRESHOLD: per-market swap_for_higher_conviction exits average holding_days >= 3` (the anti-churn bar) — US 4.7 pass, KR 0.5 fail.
- `HEALTHY-THRESHOLD: at least 1 filled trade per active market within the proof window` (65.4 liveness — EU currently 0 trades).
All thresholds are BASELINE defaults; 65.4 measures the post-token proof window
against them. Where current data fails (KR churn), that documents the pre-fix
state honestly rather than rigging a pass.

---

## E. The 61.1 / churn-fix flag split (criterion 3)

Flag = **`paper_swap_churn_fix_enabled`** (settings.py:345, phase-60.2, default
OFF). NOTE the criterion says "61.1 flags" but the swap-churn flag is phase-60.2;
`paper_data_integrity_enabled` (settings.py:45) is the sibling. Report both names.

- **State + EXACT date: operator-activated ON 2026-06-12.** The operator gave the
  token "60.2 FLAG: ON (Recommended)" via AskUserQuestion and appended `.env`
  (`handoff/harness_log.md:26954`). The **first post-flag cycle was `5f15fdbe`,
  2026-06-12 18:00 UTC** (`harness_log.md:27097`); re-confirmed still ON at
  phase-70.3 (`:27420`, "PAPER_SWAP_CHURN_FIX_ENABLED=true"). The 61.1 audit
  independently verified "post-flag swap_for_higher_conviction SELLs = 0".
  (Main can re-confirm from `.env`; the researcher sandbox is denied it, but the
  harness log pins the date.)
- **Behavioral cut agrees**: last `swap_for_higher_conviction` exit = **2026-06-10**
  (flag OFF); ZERO swap exits on/after 2026-06-12 (flag ON). Confound to disclose:
  the engine was ALSO quiet mid-June→early-July (away-ops credential expiry, sparse
  cycles 06-23/07-03/07-13), so the thin post-fix sample is partly liveness, not
  only the flag — state both causes honestly.
- **Recommended split for the doc (use these exact dates)**:
  - PRE-FIX baseline = **2026-06-01 → 2026-06-11** (flag OFF; swap-churn cluster;
    last swap 06-10). DOMINATES the "since 06-01" aggregates (KR median-hold 1d,
    US <=1d churn leg).
  - POST-FIX = **2026-06-12 onward** (flag ON; first cycle 5f15fdbe 06-12 18:00 UTC;
    0 churn swaps since). Live post-fix sample is THIN → note the post-fix trend as
    PENDING more cycles (the 65.4 proof window accrues it). Present both segments; never merge.

---

## F. External research

### Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://www.backtestbase.com/education/win-rate-vs-profit-factor | 2026-07-18 | blog(quant-edu) | WebFetch full | Profit factor <1.0 losing, 1.0-1.5 fragile, 1.5-2.0 solid, 2.0+ robust; PF>=1.5 = min acceptable; trend systems profitable at 30-40% win rate if PF>1.5; small samples give misleading PF |
| https://www.luxalgo.com/blog/top-5-metrics-for-evaluating-trading-strategies/ | 2026-07-18 | blog(tooling) | WebFetch full | PF>1.75 healthy; MDD<20%; Sharpe>1.0; win rate strategy-dependent (trend 30-50%, mean-rev 60-80%); review expectancy over >=100 trades |
| https://stockio.ai/blog/metrics-algorithmic-trading-success | 2026-07-18 | blog(algo) | WebFetch full | PF 1.5+ min / 2-3 ideal (>3-4 = overfit); win 50-55% baseline (trend 30-40% OK); MDD<15-20%, red flag if realtime MDD>150% historical; 50-100+ trades for confidence; early-warning = PF decline across D/W/M |
| https://www.tradezella.com/blog/analyze-trading-performance | 2026-07-18 | blog(journal) | WebFetch full | 2026: win day 40-60%/swing 35-45%; PF>1.3 solid/>1.5 strong; MDD<15%; **MIN 30 trades for a meaningful single-metric conclusion, <20 = noise**; segment by setup/time/ticker/HOLDING-TIME/market |
| https://iongroup.com/blog/markets/algorithmic-trading-monitoring-and-management/ | 2026-07-18 | industry(institutional) | WebFetch full | Institutional monitoring = alert on spread deviation from historical avg, slice/fill thresholds, exec unable-to-proceed; pause/intervene tooling; health = params within limits |
| https://www.owox.com/blog/articles/bigquery-cast-and-safe-cast | 2026-07-18 | doc(vendor) | WebFetch full | SAFE_CAST returns NULL (not error) on bad cast; CAST AS TIMESTAMP needs <=microsecond precision; DATE cast needs 'YYYY-MM-DD'; `WHERE SAFE_CAST(str AS DATE) >= '2024-01-01'` pattern |
| https://www.wrightresearch.in/blog/portfolio-churn-and-its-effect-on-long-term-returns/ | 2026-07-18 | industry(asset-mgr) | WebFetch full | Churn erodes returns via cost+tax+timing friction; long holds/patience recommended; QUALITATIVE only (no bp figures) |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.schwab.com/learn/story/how-overtrading-can-undercut-after-tax-returns | industry | auth/error page returned (nav-only) |
| http://fastercapital.com/content/Cost-of-Trading--Minimizing-the-Hidden-Cost... | blog | HTTP 403 (snippet: 5-15bp large-cap, 15-30bp mid/small, 10-50bp market impact per trade) |
| https://www.babypips.com/trading/trading-performance-metrics | blog(edu) | HTTP 403 |
| https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/conversion_functions | doc(official) | JS-rendered nav-only (known: feedback_gcloud_docs_fetch); owox blog used instead |
| https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions | doc(official) | JS-rendered nav-only |
| https://www.edgewonk.com/blog/the-ultimate-guide-to-the-10-most-important-trading-metrics | blog | redundant w/ read set |
| https://capitalcompanion.ai/tools/win-rate-calculator/ | tool | calculator, not prose |
| https://journalplus.co/blog/how-to-calculate-trading-performance/ | blog | redundant |
| https://www.paytmmoney.com/blog/portfolio-turnover-ratio-mutual-funds-strategy/ | blog | mutual-fund framing, less applicable |
| https://financialmodelslab.com/blogs/blog/portfolio-turnover-rate | blog | redundant on turnover |
| https://www.utradealgos.com/blog/analysing-the-performance-of-algorithmic-trading-strategies... | blog | redundant |

### Recency scan (last 2 years: 2024-2026)
Searched 2026-scoped ("...dashboard 2026...", "...transaction costs 2025") AND
year-less canonical. Result: **the 2026 sources (tradezella 2026, stockio 2025-26,
ION) CONFIRM and slightly TIGHTEN the canonical thresholds** rather than supersede
them: PF>=1.5, MDD<15-20%, win-rate strategy-dependent are stable across 2024-2026.
The one materially useful NEW finding is the explicit **minimum-sample-size rule**
(tradezella 2026: 30 trades min per metric, <20 = noise; stockio: 50-100+), which
directly bears on this baseline (US 17 / KR 5 closed trades → below the floor →
metrics are DESCRIPTIVE, not inferential). No newer source contradicts the churn/
short-hold-drag thesis; the 2026 algo-monitoring sources add the "early-warning =
PF decline across D/W/M + realtime MDD>150% historical" pattern for 65.4 framing.

### Key findings (external → application)
1. **Profit factor is the #1 health metric; >=1.5 is the floor.** (backtestbase, luxalgo, stockio consensus) → primary HEALTHY-THRESHOLD, sourced via `paper_round_trips.summarize().profit_factor` (the /round-trips endpoint), since paper_trades SQL lacks realized_pnl_usd.
2. **Win rate is strategy-dependent and NOT the primary indicator.** Trend/momentum systems (pyfinagent is momentum-tilted) are healthy at 30-40% win rate IF PF>1.5. (backtestbase, stockio) → win-rate threshold set low (>=40%) and flagged descriptive.
3. **Minimum 30 trades per metric for a meaningful conclusion; <20 = noise.** (tradezella 2026; stockio 50-100+) → US 17 / KR 5 are BELOW the floor. The baseline's inferential weight is on STRUCTURAL/behavioral lines (holding-days, churn-swap-hold, fee-drag), not win-rate/PF.
4. **Churn/short holds erode net returns via cost+tax+timing friction; segment by holding time.** (Wright, tradezella segmentation) + snippet: 5-30bp/trade + 10-50bp impact → the holding-days + fee thresholds are the anti-churn gates; corroborates the internal churn finding (-$139.83 on <=2-day holds).
5. **Early-warning framing for 65.4:** PF decline across daily/weekly/monthly + realtime MDD>150% of historical. (stockio, ION) → 65.4 "judged against thresholds" = compare proof-window aggregates to these baseline lines.
6. **BigQuery: SAFE_CAST is NULL-safe; ISO-8601 `created_at` casts cleanly.** (owox) → `SAFE_CAST(created_at AS TIMESTAMP)` valid (the 'T' separator + microsecond precision are fine); string `>= "2026-06-01"` also valid (lexical sort). Both documented in §A.

### Consensus vs debate (external)
CONSENSUS: PF>=1.5 floor; MDD<15-20%; win-rate is secondary + strategy-dependent;
segment analysis by holding time; min ~30 trades. DEBATE/variance: exact win-rate
bands differ by source (day 40-60% vs swing 35-45% vs trend 30-40%) — resolve by
strategy type, not a universal number. No source disputes that short-hold churn is
a net drag.

### Pitfalls (from literature)
- Judging a strategy on win-rate alone (high win-rate + poor R:R still loses).
- Trusting PF/win-rate at n<30 (this baseline's exact trap — disclose it).
- PF>3-4 as a red flag for overfit/small-sample, not just "great".
- CAST (not SAFE_CAST) aborts the whole query on one bad string row.

---

## G. HEALTHY-THRESHOLD lines — FINAL (externally grounded, criterion 2)

Global caveat line for the doc: *at current n (US 17 / KR 5 closed, EU 0), win-rate
& profit-factor are DESCRIPTIVE not inferential (external min ~30 trades; matches the
project's own 60.2 "n=1 regime, descriptive not inferential" ruling). The structural
lines below (holding-days, churn-swap-hold, fee-drag, liveness) are the robust 65.4 gates.*

- `HEALTHY-THRESHOLD: per-market profit_factor >= 1.5 (via paper_round_trips.summarize; descriptive at n<30)` — external #1 metric (backtestbase/luxalgo/stockio).
- `HEALTHY-THRESHOLD: per-market median holding_days >= 5` — anti-churn; internal churn = <=2-day holds net loss; segment-by-holding-time (tradezella). Current US 3 / KR 1 → flags pre-fix churn (intended).
- `HEALTHY-THRESHOLD: per-market share of exits at holding_days <= 1 is < 40%` — US 35% pass / KR 80% fail. Direct churn gauge.
- `HEALTHY-THRESHOLD: per-market swap_for_higher_conviction exits average holding_days >= 3` — the specific churn mechanism (60.2); US 4.7 pass / KR 0.5 fail.
- `HEALTHY-THRESHOLD: no market > 0.50% of NAV in cumulative fees per rolling 30-trading-day window` — current US 0.085% / KR 0.020% of $23,874.56 NAV; 0.5% ≈ 5x headroom, catches a churn blow-out (external 5-30bp/trade drag).
- `HEALTHY-THRESHOLD: per-market win_rate >= 40% on >= 30 closed trades (descriptive below 30)` — momentum-system floor (trend 30-40% OK if PF>1.5); US 70.6% / KR 20% (KR sub-floor + sub-n).
- `HEALTHY-THRESHOLD: >= 1 filled trade per ACTIVE market within the 65.4 proof window` — liveness (EU currently 0 trades).
- Early-warning (not a pass/fail line): flag if proof-window profit_factor drops across weekly buckets OR realtime max-drawdown exceeds 150% of the baseline MDD.

---

## H. Research Gate Checklist
Hard blockers (gate_passed=false if any unchecked):
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 read)
- [x] 10+ unique URLs total (25+ collected; 7 read, 11 snippet-only tabled)
- [x] Recency scan (last 2 years) performed + reported (§F)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (§A-E cite settings.py/paper_trader.py/paper_round_trips.py/markets.py/harness_log.md lines)
Soft checks:
- [x] Internal exploration covered schema, aggregates, win/hold defs, fee model, flag timing
- [x] Contradictions/consensus noted (§F consensus-vs-debate)
- [x] Per-claim citation (URLs + access dates in tables)

## JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 11,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0, "K_required": 2, "new_findings_last_round": 0, "dry": false},
  "summary": "US+KR health baseline audited $0. paper_trades: created_at is STRING (filter via >= string compare or SAFE_CAST), NO market column (derive via ticker suffix = markets.py:142 market_for_symbol). Win=realized_pnl_pct>0 (paper_round_trips.py:145); holding_days+realized_pnl_pct precomputed on SELL rows; fee=notional*0.1% USD (settings.py:371). 4 aggregate SQLs written + dry-run: US 28 trades/70.6% win/median-hold 3d/$20.30 fees; KR 10/20%/1d/$4.82; EU 0. Churn flag paper_swap_churn_fix_enabled ON since 2026-06-12 (harness_log:27097); split pre-fix 06-01..06-11 vs post-fix 06-12+ (0 churn swaps, thin sample, confounded by away-ops quiet). 7 external sources: PF>=1.5 floor, win-rate strategy-dependent+secondary, MIN 30 trades/metric (US 17/KR 5 => descriptive not inferential), segment by holding-time. 8 HEALTHY-THRESHOLD lines proposed, structural (holding-days/churn/fee/liveness) as robust gates.",
  "brief_path": "handoff/current/research_brief_65.3.md",
  "gate_passed": true
}
```
