# Contract -- Cycle 87 / phase-4.8 step 4.8.10

Step: 4.8.10 2026 regulatory memo + wash-sale filter

## Research gate

Researcher (22 URLs: SEC 34-96930 T+1, IRS Pub 550, FINRA
SR-2025-017 intraday margin, DTCC Net Debit Cap, FINRA 2026
Oversight Report) + Explore (paper_round_trips returns
{ticker, entry_date, exit_date, realized_pnl_pct, ...}; no existing
wash-sale or margin tracking; docs/compliance/ missing).

Key findings:
- **Wash-sale window is 61 CALENDAR days** (t-30..t+30 inclusive).
  Use calendar days, not trading days; off-by-one is a canonical
  pitfall.
- **T+1 settlement** (SEC 15c6-1 amended; effective 2024-05-28)
  requires tracking `settled_cash` vs `pending_proceeds`
  separately. Same-day sell -> same-day buy is NOT funded.
- **FINRA 4210 intraday margin** (SR-2025-017 approved 2026-04-17)
  requires real-time margin-deficit detection; 12-month interim
  before hard enforcement.

## Scope

Files created:

1. **NEW** `docs/compliance/2026-regulatory-memo.md`
   7-section internal memo: Scope / Regulatory changes (T+1,
   wash-sale, FINRA 4210) / System impact / Operational controls /
   Monitoring / Review cadence / Open items.

2. **NEW** `backend/services/wash_sale_filter.py` library:
   - `WashSaleLedger`: tracks (symbol, sell_date,
     disallowed_loss_amount) entries; auto-expires after 61 days.
   - `record_loss(trade)` and `is_wash_sale(symbol, buy_date)`
     using calendar-day arithmetic.
   - `filter_candidates(buys, ledger)` returns (allowed, blocked)
     with `wash_sale=True` flag on blocked + `disallowed_loss_bps`.

3. **NEW** `backend/services/funding_guard.py`:
   - `t1_funding_guard(settled_cash, pending_proceeds, buy_notional)`
     returns (allowed, reason). Blocks when buy_notional exceeds
     settled_cash even if pending_proceeds would cover.
   - `realtime_margin_guard(gross_long, available_margin,
     buy_notional, deficit_threshold_pct=0.0)` returns
     (allowed, reason). Blocks when projected gross long would
     exceed available margin.

4. **NEW** `scripts/compliance/wash_sale_filter.py` CLI with
   `--test` flag:
   - Synthesizes trade history with KNOWN wash-sale pairs (same
     ticker loss at t=0, rebuy at t+15) + non-wash pairs (rebuy at
     t+61) + guaranteed-funding and margin-deficit cases.
   - Runs WashSaleLedger + funding/margin guards.
   - Asserts: wash pairs flagged, non-wash not flagged, unsettled
     funding blocks same-day buy, margin deficit blocks oversized
     buy.
   - Emits `handoff/wash_sale_filter_test.json` + exit 0 on pass.

5. **NEW** `scripts/audit/regulatory_memo_audit.py` verifies:
   - memo exists with all 7 required sections
   - library modules expose the named functions
   - test script runs green
   - 61-day window is CALENDAR (docstring + algorithm check)

## Immutable success criteria

1. memo_landed -- docs/compliance/2026-regulatory-memo.md exists.
2. wash_sale_filter_active -- wash_sale_filter.py --test exits 0
   with known wash/non-wash pairs correctly classified.
3. t1_funding_guard_active -- audit proves same-day buy funded
   from same-day unsettled sell is BLOCKED.
4. realtime_margin_handler_active -- audit proves oversized buy
   (gross_long > available_margin) is BLOCKED.

## Verification (immutable, from masterplan)

    test -f docs/compliance/2026-regulatory-memo.md && \
    python scripts/compliance/wash_sale_filter.py --test

Plus: `python scripts/audit/regulatory_memo_audit.py --check`.

## Anti-rubber-stamp

qa must verify:
- 61-day window uses calendar-day subtraction, not business-day.
- wash-sale fixtures contain BOTH positives (loss @t=0, buy @t+15)
  AND negatives (buy @t+61) so the filter's discrimination is
  actually tested.
- t1 guard fixture has same-day sell unsettled -> same-day buy
  blocked; a one-day-later buy allowed.
- margin guard fixture has gross_long + buy_notional > margin ->
  blocked; smaller buy allowed.
- Memo references REAL citations (SEC 34-96930, IRC §1091, FINRA
  4210), not hand-waved.

## References

- Researcher cycle-87 findings (22 URLs).
- Explore cycle-87 findings (paper_round_trips shape).
- Cycle-86 `backend/services/compliance_logger.py` pattern reuse
  (dataclass + validate + dual-backend).
