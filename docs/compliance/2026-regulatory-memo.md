# 2026 Regulatory Memo -- pyfinagent

- **Date**: 2026-04-18
- **Classification**: Internal
- **Author**: pyfinagent operator (Ford / Peder B. Koppang)
- **Distribution**: project owner, any future brokered-execution counterparties

## 1. Scope

This memo covers three US equity-trading regulatory changes that
became effective (or were substantially finalized) between
2024-05-28 and 2026-04-17 and that affect the pyfinagent paper-
trading + future-live-trading stack. The memo applies to all code
paths that emit or persist trade decisions, specifically:

- `backend/services/paper_trader.py`
- `backend/services/portfolio_manager.py`
- `backend/services/execution_router.py`
- `backend/backtest/backtest_engine.py`

Pyfinagent is PAPER-ONLY as of this memo date. The controls
described below are prepared for go-live; during paper operation
they serve learning-signal accuracy + evaluator realism.

## 2. Regulatory changes

### 2.1 T+1 Settlement (SEC Rule 15c6-1 amendment)

- **Citation**: SEC Release 34-96930 (Feb 15 2023; 88 Fed. Reg.
  13872, Mar 6 2023).
- **Effective**: 2024-05-28.
- **Change**: Standard settlement cycle shortened from T+2 to T+1
  for most broker-dealer transactions in US securities (equities,
  corporate bonds, muni bonds, UITs, some ETFs). Also introduces
  Rule 15c6-2 mandating allocations/confirmations/affirmations
  complete by end of trade date.
- **Sources**:
  https://www.sec.gov/rules-regulations/2023/02/34-96930,
  https://www.finra.org/investors/insights/understanding-settlement-cycles

### 2.2 Wash-sale rule (IRC §1091 / IRS Publication 550)

- **Citation**: 26 U.S.C. §1091; IRS Pub 550 (2025 ed.); Tax
  Topic 409.
- **Effective**: Unchanged; long-standing rule.
- **Rule**: A loss from selling a security is disallowed for tax
  purposes if "substantially identical" security is purchased
  within the 61-calendar-day window: 30 days BEFORE through 30
  days AFTER the sale. The disallowed loss is ADDED to the cost
  basis of the replacement position.
- **Pitfall**: the window is CALENDAR days, not trading days.
  The ledger must include weekends + holidays.
- **Source**: https://www.irs.gov/publications/p550

### 2.3 FINRA 4210 intraday margin (SR-FINRA-2025-017)

- **Citation**: FINRA SR-2025-017 (filed 2025-12-29, approved
  2026-04-17, Federal Register 2026-07485).
- **Effective**: 12-month interim period from approval; mandatory
  by ~2027-04-17.
- **Change**: Replaces the legacy $25K pattern-day-trader regime
  with real-time intraday margin-deficit standards. Member firms
  must compute each customer's intraday deficit in real time; a
  deficit not cured by end-of-day-5 triggers a 90-day freeze on
  increasing shorts or debits.
- **Source**: https://www.finra.org/rules-guidance/rule-filings/sr-finra-2025-017

## 3. System impact

| Change | Current system state | Required behaviour |
|---|---|---|
| T+1 | paper_trader deducts cash immediately on BUY; no settlement tracking | Split cash into `settled_cash` + `pending_proceeds`; only settled_cash may fund new BUYs |
| Wash-sale | No tax-lot or wash-sale tracking anywhere | Maintain 61-day loss ledger keyed by ticker; filter buys within the window; add disallowed loss to replacement cost basis |
| FINRA 4210 | No margin facility modeled | Add `realtime_margin_guard` that checks projected gross long vs available margin before BUY; block on deficit with `MARGIN_DEFICIT` reason |

## 4. Operational controls (implemented phase-4.8.10)

- **`backend/services/funding_guard.py::t1_funding_guard`**:
  blocks BUY when `buy_notional > settled_cash`, even if
  `pending_proceeds` would cover it.
- **`backend/services/funding_guard.py::realtime_margin_guard`**:
  blocks BUY when projected `gross_long + buy_notional >
  available_margin`. Configurable `deficit_threshold_pct`.
- **`backend/services/wash_sale_filter.py::WashSaleLedger`**:
  records closed-loss trades; `is_wash_sale(symbol, buy_date)`
  returns True for any buy within 61 calendar days of a recorded
  loss. `filter_candidates(buys, ledger)` partitions candidate
  buys into (allowed, blocked).
- **`scripts/compliance/wash_sale_filter.py --test`**: end-to-end
  sanity test exercising known positive + negative wash pairs +
  funding + margin fixtures.

## 5. Monitoring

- `scripts/audit/regulatory_memo_audit.py --check` runs in CI
  (manual invocation today; a future phase wires to GitHub
  Actions alongside `pip-audit.yml` from Cycle 85).
- Wash-sale filter emits one row per filtered candidate to
  `handoff/wash_sale_events.jsonl` (future). During paper
  trading, evaluator agent consumes this for Sharpe-adjusted
  scoring.
- FINRA 4210 intraday margin is MONITORED during the 12-month
  interim; hard-block enforcement flips after 2027-04-17.

## 6. Review cadence

- **Quarterly**: 2026-07-18, 2026-10-18, 2027-01-18, 2027-04-18.
  At each review: re-read this memo + check for new SEC / FINRA
  releases since last review.
- **Event-driven**: any regulatory release matching keywords
  "settlement cycle", "margin", "wash sale", "AI supervision",
  "broker-dealer recordkeeping" triggers an ad-hoc memo update.
- Escalation path: open a github issue labeled `regulatory` +
  tag the owner.

## 7. Open items / risk register

- Current pyfinagent entity is NOT a registered broker-dealer.
  If go-live uses Alpaca as executing broker, Alpaca's SEC 17a-4
  obligations apply to the order record; pyfinagent-side
  rationale storage (Cycle 86 `compliance_logger`) is a
  supervisory record under Rule 3110, not the regulatory
  trade-order record.
- GCS Bucket Lock bucket `pyfinagent-rationale-worm` is NOT
  yet created (gated on `COMPLIANCE_WORM_BUCKET` env). Must
  create before any real trading.
- Wash-sale ledger is currently process-local. If multiple
  paper-trader processes run in parallel, the ledger must be
  centralized (BQ table) -- single-process is sufficient for
  today's cron model.
- Options / ETF "substantially identical" edge cases are not
  handled; pyfinagent is equities-only by design.
