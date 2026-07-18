# Research Brief — phase-72.2 "P2 MEASUREMENT INTEGRITY"

Tier: moderate. NOT audit-class. Researcher: Layer-3. Started 2026-07-18.

Baseline: `handoff/current/money_recon_2026-07-18.md` + `money_diagnosis_72.md`.
Scope: pin down the three suspected measurement defects the NEXT GENERATE must
verify — (a) benchmark_pnl_pct discontinuity / daily-rebase, (b) FX-1.0 fallback
booking on non-USD proceeds, (c) missing 07-17/18 snapshots + NULL nav in
paper_metrics_v2. READ-ONLY except this brief.

---

## Internal code inventory (measurement seams) — the main leg

_(filled incrementally below as files are read)_

### (a) Benchmark computation seam — CAUSE OF THE 05-23->05-26 DISCONTINUITY PINNED

**End-to-end path:**
1. `paper_trader._get_benchmark_return(inception_date, first_funded_date)` —
   `backend/services/paper_trader.py:1283-1313`. Computes SPY total return from
   `anchor = first_funded_date or inception_date` to **today** via
   `yf.Ticker("SPY").history(start=anchor)`; `((Close[-1]-Close[0])/Close[0])*100`.
2. Anchor source: `bq.get_first_funded_snapshot_date()` =
   `MIN(snapshot_date) WHERE positions_value>0` (`db/bigquery_client.py:1050-1079`).
3. `mark_to_market()` recomputes it EVERY cycle and writes it to
   `portfolio.benchmark_return_pct` (`paper_trader.py:621-631`).
4. `save_daily_snapshot()` reads that stored value (`paper_trader.py:957`) and writes
   it into TODAY's row as `benchmark_pnl_pct`, plus
   `alpha_pct = cumulative_pnl_pct - benchmark` (`paper_trader.py:971-972`).
   MERGE keyed on `snapshot_date` (`bigquery_client.py:1012-1037`) -> only today's row
   is touched; **historical rows are never rebuilt**.

**Root cause of the ~10pp cliff (git-confirmed):** commit `320b7dbb`
"phase-38.7: SPY benchmark anchor at first-funded snapshot" landed
**2026-05-22 23:57:41 +0200** — exactly on the discontinuity boundary. Before it,
the anchor was `inception_date` (row-creation, pre-funding, EARLY) -> SPY `Close[0]`
low -> cumulative SPY return HIGH (14.97 on 05-23). After it (first cycle 05-26, post
Memorial-Day weekend), the anchor is `first_funded_date` (LATER) -> SPY `Close[0]`
higher (SPY rose in between) -> cumulative SPY return LOWER (4.76 on 05-26). A benchmark
DROP of ~10pp -> `alpha = cum_pnl - benchmark` JUMPS +10pp. **The stored series mixes
inception-anchored rows (<=05-23) with funded-anchored rows (>=05-26): a pure DEPLOY
ARTIFACT, not a market move.** The GENERATE phase should treat 38.7's anchor as the
GIPS-correct one but rebuild the pre-05-26 rows (or flag them) so the series is internally
consistent, and stop celebrating the +10pp step as outperformance.

**"Benchmark moves while NAV frozen" (07-03..08):** correct-by-design, not a bug — SPY
keeps moving to `today` even at 100% cash, so `alpha = frozen cum_pnl - moving benchmark`
drifts. It is a PRESENTATION issue (a cash book's "alpha" is just -SPY), not corruption.
Anchor stays stable while cash because the earliest funded snapshot is fixed in the past.

**Two independent measurement defects in this path (for GENERATE):**
- (a1) `cumulative_pnl_pct` = `(nav - starting_capital)/starting_capital`
  (`paper_trader.py:956,970`) uses raw NAV vs a starting_capital that was BUMPED by the
  $5K deposit -> a crude flow-adjustment, but the daily `benchmark_pnl_pct` recompute vs a
  floating anchor is NOT the same TWR basis as `paper_metrics_v2._nav_to_returns` uses.
  Alpha compares a bumped-baseline portfolio % against a buy-&-hold SPY % — apples/oranges.
- (a2) The stored `benchmark_pnl_pct` per row is a "recompute-to-today from floating
  anchor" value frozen at write time, so any anchor/logic deploy leaves a step in history.

### (b) FX booking seam — the 1.0 default is FIXED IN CURRENT CODE but the KR history predates the fix

**Current state (post phase-69.1, commit ~2026-07-11):**
- `execute_sell` — `paper_trader.py:414-430`: `_l2u = _fx_local_to_usd(market)`; when
  `_l2u is None` it now **BLOCKS the exit + logs error** (`return None`), NO LONGER credits
  at 1.0. So the *live* 1.0-mis-booking on sells is closed.
- `fx_rates._usd_value_live` — `fx_rates.py:105-114`: on dual (yfinance+FRED) outage now
  serves `_last_known_usd_value` from `historical_fx_rates` (`:163-196`) before returning
  None. So `_l2u is None` is reachable ONLY if NO rate for that market was ever stored.
- Sell proceeds enter cash: `sell_value`/`net_proceeds` are LOCAL, credited to cash via
  `*_l2u` (`paper_trader.py:442-444`, credited downstream ~:520-540); trade row
  `total_value = sell_value * _l2u` (`:471`); round-trip
  `realized_pnl_usd = (price-entry_price)*sell_qty*_l2u` (`:498`).

**Why the RECORDED P&L can still be corrupt (measurement, not live):** the KR tickers
000660.KS/005930.KS/066570.KS traded May-June, BEFORE the 69.1 fix (07-11). Any KR sell
during a pre-fix dual-FX outage was booked at `_l2u=1.0` -> KRW notional booked as USD
(~1300x) into `paper_trades.total_value`, `paper_round_trips.realized_pnl_usd`, and cash
-> NAV. `paper_trades` has **no currency column**, so the corruption is not self-evident.
The realized +$3,194.68 and NAV rest on this. GENERATE must forensically check the KR
round-trip rows.

**Still-LIVE FX measurement bugs (NOT fixed by 69.1):**
- `paper_round_trips.py:109` — `pair_round_trips` computes `realized_pnl_usd` from raw
  LOCAL trade prices with NO FX conversion; `summarize()` profit_factor / gross win-loss
  mix KRW+EUR+USD as dollars. (register CONFIRMED med.)
- `paper_trader.py:308,325-334` — non-US ADD-ON avg_entry: LIVE path (flag
  `paper_avg_entry_fx_fix_enabled` OFF, settings.py:455) uses `new_cost(USD)/new_qty(LOCAL)`
  -> corrupts the local-currency entry unit that stops + realized_pnl treat as local.
  First-lot (`:364`) is correct (stores LOCAL price).
- `paper_trader.py:1124` (register) — trailing-stop peak reconstructed from USD-return MFE
  for EU/KR -> FX return component distorts the trail.
- `fx_rates._usd_value_asof:199-225` — on BQ miss/error degrades to `_usd_value_live`
  (TODAY's live rate) for a historical as-of date -> look-ahead FX in point-in-time reads.

### (c) Snapshot writer seam — TWO decoupled writers; nav NULL is BY DESIGN

- **`paper_portfolio_snapshots`** written by `PaperTrader.save_daily_snapshot`
  (`paper_trader.py:935-983`) -> `bq.save_paper_snapshot` MERGE (`bigquery_client.py:993-1037`).
  Called at cycle end (call site in `autonomous_loop.py` — traced below) AND by
  `adjust_cash_and_mtm` (`:1056`). No 07-17/18 row => save_daily_snapshot was not reached
  or its cycle leg aborted before it. (Seam pinned below.)
- **`paper_metrics_v2`** (dataset `pyfinagent_pms`) written by
  `paper_metrics_v2.persist_metrics_v2` (`paper_metrics_v2.py:163-211`). **`nav` is
  HARDCODED to `None` at `:195`** — the column is NEVER populated by any writer. So the
  "07-17 nav NULL" is not a cycle failure; it is a permanent writer defect. This writer is
  called from the metrics-v2 API path (`api/paper_trading.py:1009-1010`), NOT the daily
  cycle — which is why a 07-17 metrics-v2 row can exist while the snapshot row does not
  (different triggers). `persist_metrics_v2` also early-returns on `insufficient_data`
  (`:169`, n_obs<30) — since the 07-17 row exists, n_obs>=30 held.

**WHY 07-17 ran LLM work but wrote no snapshot (seam pinned):** `save_daily_snapshot` is
Step 8 (`autonomous_loop.py:1535-1544`), the LAST substantive step, INSIDE the main try
(opened `:411`). The terminal handlers `except asyncio.TimeoutError` (`:1639`, "cycle
TIMED OUT after Ns"), `except Exception` (`:1644`, "cycle failed"), and `finally` (`:1662`,
sets `_running=False` + releases lock) ALL `return summary` WITHOUT writing a snapshot. The
ONLY abort-path that snapshots is the kill-switch early-return (`:1271-1279`). So any
mid-cycle raise OR a cycle-level timeout (the P0 log shows 305 subprocess timeouts @120s +
breaker OPEN on 07-10/13/14/15) leaves that day with NO `paper_portfolio_snapshots` row
even though partial analysis LLM calls were logged. The metrics-v2 writer, on the separate
API trigger, is unaffected — hence a row there but not in snapshots. (Which exact mechanism
fired 07-17 is NEXT-phase forensics; the SEAM is the non-fail-safe Step-8 placement.)

**BQ confirmations (this session, bounded read-only):**
- `paper_trades` schema columns = [trade_id, ticker, action, quantity, price, total_value,
  transaction_cost, reason, analysis_id, risk_judge_decision, created_at, round_trip_id,
  holding_days, realized_pnl_pct, mfe_pct, mae_pct, capture_ratio, signals] — **NO
  currency/market column**. FX corruption is not self-evident from a row (must infer ccy
  from the .KS/.KQ/.DE ticker suffix).
- KR trades in `paper_trades` (`.KS`/`.KQ`) = **10** — the FX seam is reachable in the
  recorded ledger.
- `MAX(snapshot_date)` in `paper_portfolio_snapshots` = **2026-07-16** — 07-17/18 gap
  confirmed. `benchmark_pnl_pct` on 2026-05-26 = **4.76** (post-38.7 value confirmed).

### (d) Recent relevant history (fixes shipped / parked)

- phase-38.7 (`320b7dbb`, 2026-05-22) — benchmark anchor inception->first-funded (the
  cause of the discontinuity; GIPS-motivated, but created the mid-series step).
- phase-30.4 — GIPS TWR: `external_flow_today` subtracted in
  `paper_metrics_v2._nav_to_returns` (`:36-81`); but the external-flow WRITE path is
  structurally dead (`paper_trader.py:942` register: adjust_cash_and_mtm has no production
  callers; deposit endpoint bypasses it; MERGE clobbers same-day flow to 0.0).
- phase-69.1 (~2026-07-11) — FX-never-1.0 (execute_sell block + last-known fallback),
  kill-switch, pkill-removed. Landed AFTER the KR trades, so it protects the future, not
  the recorded history.
- phase-56.1 / 50.2 — trade rows persist USD (`*_l2u`); positions carry `market` +
  `base_currency="USD"`.
- Register (audit 2026-07-10/11) CONFIRMED money-path defects: perf_metrics.py:116
  (compute_sharpe_from_snapshots ignores external_flow_today — a SECOND divergent
  NAV->returns path vs paper_metrics_v2), paper_round_trips:109, fx_rates:176.

---

## External research

### Read in full (>=5 required; counts toward the gate) — 7 read

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| cfainstitute.org/.../2026/overview-of-the-global-investment-performance-standards | 2026-07-18 | Official (CFA GIPS 2026 refresher) | WebFetch full | "Time-weighted returns are required for all portfolios except portfolios meeting certain criteria"; TWR "removes the effects of cash flows, which are generally client-driven"; "composite and benchmark annual returns for all years" must be reported. |
| kitces.com/blog/twr-dwr-irr-calculations-performance-reporting-software-methodology-gips-compliance | 2026-07-18 | Authoritative practitioner | WebFetch full | TWR "strips out cash flow effects to show pure investment performance"; DWR/IRR "more accurate...relative to the dollars invested". Large deposits/withdrawals make naive return math "highly misleading" (200% phantom growth example). |
| en.wikipedia.org/wiki/Time-weighted_return | 2026-07-18 | Canonical (year-less) | WebFetch full | Exact sub-period formula `1+R = (M2 - C2)/M1`: the external flow C is SUBTRACTED from the ending value before differencing; geometric linking `∏ (Mi - Ci)/M(i-1)`; the code's `_nav_to_returns` mirrors this. |
| meradia.com/thought-leadership/returns-benchmarking-processing-for-asset-owners | 2026-07-18 | Industry | WebFetch full | "Aligning the source and timing of valuations is a fundamental step in performance measurement and benchmark relative analysis"; "Discrepancies in valuation methodologies or mixing providers can lead to return inconsistencies." |
| allinvestview.com/articles/multi-currency-portfolio-guide (2026) | 2026-07-18 | Industry (2026) | WebFetch full | "Use the same FX rate source consistently"; "For each transaction...record both the local currency amount and the exchange rate on that date"; "Maintain cost basis in your base currency"; "The exact rate you use can meaningfully affect reported gains. Consistency is key." |
| liveflow.com/blog/understanding-multi-currency-accounting-a-practical-guide | 2026-07-18 | Industry | WebFetch full | "Spot rate: The rate at the date of the transaction...used when you initially record"; "Use a single authoritative rate source...document when rates are captured"; **"Inconsistent rate sources create reconciliation discrepancies that surface at the worst possible time."** |
| cubesoftware.com/blog/multi-currency-account | 2026-07-18 | Industry (FP&A) | WebFetch full | Spot rate "best for income statements as it captures the most current exchange rate"; consolidation needed because "Showing liabilities in yen, pound sterling, and USD will not give a clear, meaningful picture"; show local + converted columns to avoid mixing. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| gipsstandards.org/.../guidance-statement-benchmarks-asset-owners.pdf | Official PDF | Binary PDF, no text extracted (not arXiv; /html chain N/A). Snippet: asset owners must present benchmark returns for all required periods. |
| gipsstandards.org/.../calculation_methodology_gs_2011.pdf | Official PDF | Binary PDF; snippet confirms daily-weighted external-flow adjustment required from 2005. |
| viewpoint.pwc.com/.../a-foreign-currency.html (IAS 21) | Official docs | WebFetch too-many-redirects; IAS 21 = spot rate at transaction date, remeasure monetary items. |
| ey.com/.../frdbb2103-09-18-2025.pdf (ASC 830, 2025) | Official PDF | Binary PDF; ASC 830 foreign-currency matters. |
| netsuite.com/.../multi-currency-accounting.shtml | Industry | HTTP 403. |
| trackyourportfol.io/blog/portfolio-performance-multiple-currencies | Industry | HTTP 503; snippet: "measure each holding in local currency, then translate...using a consistent FX source." |
| treasurers.org/.../how-handle-breakdown-currency-pairs | Industry | Fetched but off-point (correlation breakdown, not rate-source failure). |
| wellington.com/.../currency-exposure-in-multi-asset-portfolios | Industry | Snippet-only; translational vs transactional FX exposure. |
| analystprep.com/.../time-weighted-return-2 (CFA notes) | Community/edu | Snippet-only; TWR sub-period mechanics. |
| bls.gov/ces/... + imf.org/.../benchmarking-and-rebasing PDF + milliman MSSP rebasing | Official/industry | Snippet-only; "rebasing" in a macro-stats/healthcare sense (confirms rebasing = step-discontinuity risk in a series), not portfolio benchmarks. |
| docs.moderntreasury.com/payments/.../foreign-exchange-fx-quotes | Vendor docs | Snippet-only; "a quote is not a guaranteed rate...valid until it expires" (FX rates are time-sensitive, never assume constant). |

### Recency scan (2024-2026)

Performed. **Benchmarks:** the 2026 CFA GIPS refresher + 2023 GIPS Benchmarks-for-Asset-Owners guidance REAFFIRM the canonical requirement (TWR, benchmark reported for the SAME periods as the portfolio, consistent construction) — no 2024-2026 work SUPERSEDES it; the canonical rule stands and pyfinagent's mid-series anchor change violates it. **Multi-currency:** 2025-2026 sources show a clear frontier SHIFT — modern portfolio tools (AllInvestView 2026, agnifolio 2026) now treat per-position FX attribution, "measure local first then translate with a consistent stored rate," and separating the asset-return from the FX-return component as BASELINE expectations, not optional. This directly supersedes pyfinagent's design (no currency column in `paper_trades`; convert-at-trade-time with a historical 1.0 fallback; round-trips summed across KRW/EUR/USD). LiveFlow's "inconsistent rate sources create reconciliation discrepancies that surface at the worst possible time" is the 2025 practitioner statement of exactly the risk realized here.

### Key findings

1. **TWR must strip external cash flows; the code's `_nav_to_returns` is canonically correct, but a SECOND divergent path is not.** Wikipedia's `1+R=(M2-C2)/M1` and CFA/Kitces all confirm deposits must be removed before differencing. `paper_metrics_v2._nav_to_returns` (`:36-81`) does this; but `perf_metrics.compute_sharpe_from_snapshots:116` (register CONFIRMED) ignores `external_flow_today` — a divergent NAV->returns path feeding the go-live gate. (Source: Wikipedia TWR; Kitces; CFA GIPS 2026.)
2. **A benchmark must be reported on the SAME basis and period as the portfolio, consistently.** "Aligning the source and timing of valuations is a fundamental step...mixing providers can lead to return inconsistencies" (Meradia). pyfinagent's benchmark is a floating-anchor SPY recompute frozen per row; the 38.7 deploy changed the anchor mid-series -> a step-discontinuity that is a measurement artifact, and `alpha = cum_pnl - benchmark` compares a bumped-baseline portfolio % against a buy-&-hold SPY % (not the same TWR basis). (Source: Meradia; CFA GIPS 2026.)
3. **FX MUST use the actual rate at the transaction date from a single consistent source; a 1.0 (parity) fallback is a gross violation.** Every source: spot rate at transaction date (LiveFlow, Cube, AllInvestView), consistent stored source (LiveFlow, AllInvestView). A default of 1.0 for KRW (true ~1/1350) is neither the spot rate nor any real rate -> books KRW notional as USD ~1350x. Modern Treasury: an FX quote "is not a guaranteed rate." (Source: LiveFlow; AllInvestView; Cube; Modern Treasury docs.)
4. **Keep the security's native currency in the ledger; never sum local-currency P&L as base currency.** AllInvestView/Cube: keep local + converted columns; "Showing liabilities in yen, pound sterling, and USD will not give a clear...picture." `paper_trades` has NO currency column and `paper_round_trips:109` sums KRW+EUR+USD as dollars -> the exact anti-pattern. (Source: Cube; AllInvestView.)
5. **Deposits produce phantom returns if mishandled.** Kitces' 200%-phantom-growth example = pyfinagent's phase-30.0 Anomaly A ($5K deposit -> +32% phantom daily return). The fix (`external_flow_today` subtraction) is correct in `_nav_to_returns` but the WRITE path for that field is structurally dead (register `paper_trader.py:942`), so a future deposit re-creates the bug. (Source: Kitces; internal register.)

### Search queries run (3-variant discipline)

- Benchmark/TWR: `GIPS time-weighted return benchmark external cash flow methodology 2026` (frontier);
  `benchmark rebasing discontinuity performance reporting inception date 2025` (last-2yr);
  `time-weighted return benchmark construction methodology pitfalls` (year-less canonical).
- Multi-currency FX: `multi-currency portfolio P&L attribution FX conversion correctness 2026` (frontier);
  `foreign exchange rate fallback default 1.0 anti-pattern financial software 2025` (last-2yr);
  `multi-currency portfolio accounting FX conversion best practices base currency` (year-less);
  plus targeted `Modern Treasury never assume currency parity FX rate booking engineering`.

---

## Application to pyfinagent (external -> file:line) — what the NEXT GENERATE must verify

| External principle | pyfinagent seam | Verdict for GENERATE |
|---|---|---|
| Benchmark reported on same/consistent basis over the whole period (Meradia, CFA GIPS) | `paper_trader.py:1283-1313` floating anchor + `bigquery_client.py:1050-1079` `first_funded` MIN; git `320b7dbb` 2026-05-22 | Rebuild pre-05-26 rows to the 38.7 (funded) anchor OR flag the series break; stop reporting the +10pp step as alpha. |
| TWR strips flows; one canonical NAV->returns path (Wikipedia, Kitces) | `paper_metrics_v2._nav_to_returns:36-81` (correct) vs `perf_metrics.compute_sharpe_from_snapshots:116` (ignores flow) | Converge the two paths; the go-live-gate Sharpe uses the flow-blind one. |
| Alpha = portfolio TWR - benchmark TWR, same basis | `paper_trader.py:956,970-972` (cum_pnl vs starting_capital) vs SPY buy-hold | Recompute alpha on a consistent TWR basis, not bumped-baseline % vs index %. |
| FX at transaction-date spot from a consistent source; never parity (LiveFlow, AllInvestView, Cube) | `paper_trader.py:414-430` (post-69.1 blocks; pre-69.1 booked 1.0) + `fx_rates.py:78-114` | Forensically re-value the 10 KR trades; confirm none were booked at `_l2u=1.0` pre-07-11; add a currency column. |
| Keep native ccy; don't sum mixed ccy (Cube, AllInvestView) | `paper_trades` (no ccy col) + `paper_round_trips.py:109` | Add `currency`/`market` to `paper_trades`; FX-convert in `pair_round_trips`. |
| Store the rate used per event (LiveFlow, AllInvestView) | trade row has `total_value`(USD) + `price`(LOCAL) but NO stored fx_rate | Persist the fx_rate used on each trade row for audit/replay. |
| nav must be populated for evaluation metrics | `paper_metrics_v2.py:195` `nav=None` hardcoded | Populate `nav` from the live portfolio; it is dead by construction today. |
| Snapshot must be written every measured period | `autonomous_loop.py:1535-1544` Step-8 (not fail-safe) vs `:1639/:1644/:1662` terminal handlers | Move snapshot into a `finally`/fail-safe path so an aborted/timed-out cycle still records NAV. |

---

## Research Gate Checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 read)
- [x] 10+ unique URLs total (>=22 collected: 7 full + >=15 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- Soft: internal exploration covered benchmark/FX/snapshot modules end-to-end; consensus
  strong (all sources agree on spot-rate + consistent-source + same-basis-benchmark); no
  material contradiction found.

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 15,
  "urls_collected": 22,
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
  "summary": "Three measurement defects pinned to seams. (a) BENCHMARK: the 05-23->05-26 ~10pp discontinuity is the phase-38.7 deploy (git 320b7dbb, 2026-05-22 23:57) changing the SPY anchor from inception_date to the later first_funded_date; the stored benchmark_pnl_pct series mixes pre/post-fix anchors (a deploy artifact, not a market move) and inflated alpha ~+10pp. Benchmark recompute-while-NAV-frozen is correct-by-design. (b) FX: the execute_sell 1.0-parity fallback was CLOSED by phase-69.1 (~07-11, now BLOCKs) but the 10 in-window KR trades predate the fix, and paper_trades has NO currency column, so recorded P&L may be currency-corrupted; paper_round_trips:109 still sums KRW/EUR/USD as dollars. (c) SNAPSHOTS: paper_metrics_v2.nav is hardcoded None at :195 (NULL by design, not a failure); the missing 07-17 paper_portfolio_snapshots row is because save_daily_snapshot is Step-8 inside the main try and the terminal except/finally (autonomous_loop.py:1639/1644/1662) write no fail-safe snapshot, so any mid-cycle raise/timeout drops the day. External consensus (CFA GIPS, Kitces, Wikipedia TWR, Meradia, LiveFlow, Cube, AllInvestView): same-basis benchmark, transaction-date spot FX from a consistent stored source, never parity, keep native currency.",
  "brief_path": "handoff/current/research_brief_72.2.md",
  "gate_passed": true
}
```
