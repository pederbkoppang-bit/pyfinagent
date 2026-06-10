# live_check_55.1 — Data-integrity forensics: live-system evidence

**Step:** 55.1. **Date:** 2026-06-10. **Required shape (masterplan):** data-integrity section of the post-mortem + Playwright captures of the live /paper-trading page + BQ row evidence (queries + result excerpts). No fixes in this step.

Post-mortem: `handoff/current/55.1-away-week-postmortem.md`. Raw query dumps: `/tmp/55_1/*.json` (session-local cache; excerpts below are the durable record).

## A. Playwright MCP captures (live UI)

Method disclosure: the operator's NextAuth-protected instance on :3000 was NOT modified. A second dev server was started on **:3100 with `LIGHTHOUSE_SKIP_AUTH=1`** (the existing `frontend/src/middleware.ts:24` bypass), same code + same live backend (:8000) + same BQ data, then stopped after capture (operator :3000 verified still up, HTTP 302 auth redirect intact). Same rendering path ⇒ valid UI evidence.

| Capture | File (handoff/current/captures_55.1/) | What it evidences |
|---|---|---|
| Cockpit, ALL filter | `55_1_positions_cockpit_ALL.png` | NAV card **345,950.68 USD** (vs stored NAV 23,837.12), Cash **22,883.73 USD** (= BQ `current_cash` exactly), Total P&L **+1,629.75%**, Risk Monitor "Max position **1527.8%**", Allocation donut center **$345,951**, currency-exposure card (USD $23,598.72 / **6.8%**, KRW $238.40 / 0.1% — corrupted denominator), positions table LG row: Current "**live 219500,00 USD**" (KRW tick as USD), Market Value "238,40 USD" (correct), Entry "$248000.00" |
| Cockpit, KR filter | `55_1_cockpit_KR_vsKOSPI.png` | "**vs KOSPI +0,00%**" card; accessibility snapshot captured the tooltip verbatim: *"KR holdings return (USD). Per-market KOSPI excess is not yet exposed by the API."* |
| Trades tab, KR filter | `55_1_trades_KR_value_fee.png` | KR trade rows: Value column shows KRW magnitudes as USD (e.g. 364 175,06-style), Fee column "**$1056.20**" (SamsungElec SELL 06-05), "$737.26", "$677.64" — KRW fees with `$` prefix. DOM text extraction (browser_evaluate) of the same rows is pasted in §C |
| /paper-trading/manage (top) | `55_1_manage_markets_toggle.png` | Ops bar: GATE **NOT ELIGIBLE 2/5**, KILL **ACTIVE 0.0% / 3.4%**; NAV 345,949.46 USD; "Top up fund" (deposit mechanics behind starting_capital 20K); Trading settings STARTING CAPITAL **$10 000** |
| /paper-trading/manage (scrolled) — **phase-50.6 confirm #1** | `55_1_manage_markets_toggle_scrolled.png` | LIVE-LOOP MARKETS toggle: **US checked, EU/KR UNCHECKED** while the live loop's last cycle reports `universe_source: "US+EU+KR"` (§C /status) → display/config mismatch (break B14). Also MAX POSITIONS PER SECTOR=2, DAILY LOSS LIMIT=4, TRANSACTION COST 0.1 |
| /backtest — **phase-50.6 confirm #3** | `55_1_backtest_us_usd_spy_strip.png` | Header scope strip pills "**US · USD · SPY**" + OPEN status visible above tabs |

Phase-50.6 confirm #2 (positions currency-exposure card) is in capture #1.

## B. BQ row evidence (queries + excerpts)

All queries read-only, bounded, run 2026-06-10 via google-cloud-bigquery (ADC; the pinned BQ MCP was not attached this session and the claude.ai connector token was expired — CLAUDE.md fallback rule 6).

**B1. Portfolio row** — `SELECT * FROM financial_reports.paper_portfolio LIMIT 5`:
```json
{"portfolio_id":"default","starting_capital":20000.0,"current_cash":22883.73,"total_nav":23837.12,
 "total_pnl_pct":19.19,"benchmark_return_pct":2.49,"inception_date":"2026-03-20T14:01:20+00:00",
 "updated_at":"2026-06-09T18:13:21+00:00","market":"US","base_currency":"USD"}
```

**B2. Snapshots** — `SELECT snapshot_date,total_nav,cash,positions_value,daily_pnl_pct,cumulative_pnl_pct,external_flow_today FROM financial_reports.paper_portfolio_snapshots WHERE snapshot_date BETWEEN '2026-05-29' AND '2026-06-10' ORDER BY snapshot_date` → 8 rows, full table in post-mortem §1 (digest reconciles ≤0.05pp; external_flow_today=0.0 all days).

**B3. KR trade rows (the corruption)** — `SELECT created_at,ticker,action,quantity,price,total_value,transaction_cost FROM financial_reports.paper_trades WHERE created_at>='2026-05-31' ORDER BY created_at`:
```
2026-06-01 BUY  000660.KS qty=0.3124 px=2,363,000  total_value=738,196.09  fee=0.49
2026-06-02 SELL 000660.KS qty=0.3124 px=2,360,000  total_value=737,259.28  fee=737.26
2026-06-04 BUY  000660.KS qty=0.3274 px=2,298,000  total_value=752,280.44  fee=0.49
2026-06-04 BUY  005930.KS qty=3.2103 px=  351,500  total_value=1,128,428.32 fee=0.74
2026-06-05 SELL 000660.KS qty=0.3274 px=2,070,000  total_value=677,641.41  fee=677.64
2026-06-05 SELL 005930.KS qty=3.2103 px=  329,000  total_value=1,056,195.94 fee=1,056.20
2026-06-09 BUY  066570.KS qty=1.4684 px=  248,000  total_value=364,175.06  fee=0.24
```
USD re-derivation (qty×px×fx_asof): 487.87 / 486.08 / 490.83 / 736.25 / 434.39 / 677.05 / 238.40 → stored/true ratios 1,513-1,560x (KRW). The 06-09 LG row's 364,175.06 equals the operator screenshot's Value cell; the 06-05 Samsung fee 1,056.20 equals the screenshot's "$1,056.20".

**B4. Corruption scope** — `SELECT COUNT(*) total, COUNTIF(ENDS_WITH(ticker,'.KS') OR ...) non_usd, MIN/MAX(kr created_at), kr_buys, kr_sells FROM paper_trades`:
```json
{"total_rows":52,"non_usd_rows":7,"first_kr":"2026-06-01T19:33:33+00:00","last_kr":"2026-06-09T18:12:39+00:00","kr_sells":3,"kr_buys":4}
```

**B5. Positions (marks clean)** — `SELECT * FROM financial_reports.paper_positions LIMIT 20`:
```
DELL      qty=1.929296 entry=370.70  current=370.595  market_value=714.99  (USD ✓)
066570.KS qty=1.468448 entry=248,000 current=248,000  market_value=238.40  cost_basis=238.40 (=qty×₩×fx ✓)
```

**B6. FX coverage** — `SELECT pair,MIN(date),MAX(date),COUNT(*) FROM financial_reports.historical_fx_rates GROUP BY pair` → KRWUSD n=19 since 2026-05-15; June rows 06-01 0.000661, 06-02 0.000659, 06-04 0.000652, 06-05 0.000641, 06-09 0.000655 (gaps 06-03/06-08 → `date<=d` as-of fallback). Malformed rows present (MAX(date) returns 'EURUSD=X' — string column contamination, break B12).

**B7. Round trips** — `SELECT ... FROM financial_reports.paper_round_trips WHERE exit_date>='2026-05-31'` → 17 rows; MU 06-08→06-09 −44.95 USD (−6.2716%), 000660.KS 06-04→06-05 −47.85 (−9.9217%), 005930.KS −46.30 (−6.4011%); all `realized_pnl_usd` FX-correct.

**B8. Kill-switch audit trail** — `handoff/kill_switch_audit.jsonl`:
```
{"ts":"2026-06-03T19:04:39","event":"peak_update","nav":24666.57}
{"ts":"2026-06-04T19:00:08","event":"sod_snapshot","nav":24541.5,"date":"2026-06-04"}
{"ts":"2026-06-05T19:02:17","event":"sod_snapshot","nav":23862.58,"date":"2026-06-05"}   <- no pause event follows
```
Arithmetic in post-mortem §8: daily 2.77-2.82% < 4.0 limit; trailing 3.26% < 10.0 limit → CORRECTLY-DID-NOT-TRIP.

## C. API endpoint outputs (verbatim)

`GET /api/paper-trading/performance` and `GET /api/paper-trading/metrics-v2`: full verbatim JSON embedded in post-mortem §6 (performance: win_rate 0.64, profit_factor 0.0229 [defective], expectancy 13.68%, median_holding_days 17; metrics-v2: psr 0.9993, dsr 0.0, n_obs 35 — real values since n_obs≥30; the insufficient_data-nulls branch did not apply and that is honestly reported).

`GET /api/paper-trading/status` (excerpt): `portfolio.nav=23837.12, cash=22883.73, starting_capital=20000.0`; last cycle `universe_source:"US+EU+KR", universe_size:583, kill_switch:{daily_loss_pct:0.0, trailing_dd_pct:3.3528}` — the B14 markets-toggle mismatch evidence.

`GET /api/paper-trading/reconciliation` (excerpt): series starts 2026-04-14 `paper_nav:9499.5` vs `backtest_nav:20000.0` — evidences the deposit history behind starting_capital=$20K vs the $10K label.

Trades-table DOM extraction (browser_evaluate on :3100, KR filter):
```
9.6.2026 | BUY | 066570.KS | ₩248,000 | fee $0.24
5.6.2026 | SELL | 005930.KS | ₩329,000 | fee $1056.20
5.6.2026 | SELL | 000660.KS | ₩2,070,000 | fee $677.64
4.6.2026 | BUY | 005930.KS | ₩351,500 | fee $0.74
...
```

## D. Tool runs

- `python scripts/risk/tca_report.py` → `{"wrote":"handoff/tca_last_week.json","rows":70,"median_bps_liquid":5.9964,"alert_triggered":false}` (SYNTHETIC seeder — labeled, not away-week TCA).
- `python scripts/harness/paper_execution_parity.py` → **FAILED**: `alpaca.common.exceptions.APIError: {"code":40010001,"message":"client_order_id must be unique"}` (probe id `probe-alp-1` reused; break B13). Honest failure report per contract.

## E. Constraint compliance

- NO fix work performed (git status clean of backend/frontend source edits; only handoff/ artifacts written).
- NO LLM trading-cycle spend ($0: BQ reads, yfinance, local scripts, Playwright).
- Operator :3000 instance untouched and verified up post-capture.
