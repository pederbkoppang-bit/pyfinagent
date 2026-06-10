# live_check_56.1 — FX/value/fee data-correctness fix: live evidence

**Step:** 56.1. **Date:** 2026-06-10. **Required shape (masterplan):** the finding-ID → fix map + the KRW-fixture test output + a Playwright capture of the corrected /paper-trading page.

## A. Finding-ID → fix map (criterion 4; every change cites a 55.x ID)

| Finding | Fix | File(s) changed |
|---|---|---|
| **F-2** (55.1 B1/B2: KR trade rows persisted LOCAL) | BUY `total_value` × `_local_to_usd`; SELL `total_value` + `transaction_cost` × `_l2u` — row-build only (cash/round-trip paths untouched: already correct, double-conversion hazard) | `backend/services/paper_trader.py` (:265 area, :413-414 area) |
| **F-2** (regression proof) | KRW fixture tests: BUY row USD, SELL row USD value+fee, cash-credit no-double-convert, US byte-identity | `backend/tests/test_phase_50_2_multicurrency.py` (4 new tests) |
| **F-2** (historical backfill — operator-gated) | Dry-run-default migration restating the 7 corrupted rows with per-trade_id pinned literals + GIPS disclosure docstring; NOT executed | `scripts/migrations/backfill_56_1_kr_trade_values.py` (new) |
| **F-2** (rows flagged, not silently kept) | Header-comment caveat naming the 7 pre-fix KR rows + the pending migration | `frontend/src/components/paper-trading/trades-columns.tsx` (:10-14 comment) |
| **F-1** (55.1 B3-B6: client-side local-as-USD NAV) | Shared FX-safe helper `positionMarketValueUsd` (mvUsd pattern extracted); `useLiveNav` reduces over it; `RiskMonitorCard` concentration uses it | `frontend/src/lib/format.ts` (new helper), `frontend/src/lib/useLiveNav.ts` (:34-43), `frontend/src/components/paper-trading/cockpit-helpers.tsx` (:300-306) |
| **F-12** (55.1 B7: "vs KOSPI" shows holdings return) | Per the 55.1 verdict, disclosure strengthened: per-market non-US card label renamed to "<MKT> holdings" (tooltip retained); ALL/US keep true "vs SPY" excess | `frontend/src/components/paper-trading/cockpit-helpers.tsx` (benchLabel) |
| **F-13** (55.1 B8: stale "$10K" subtitle) | Amount-neutral subtitle ("Autonomous AI-managed virtual fund") + comment explaining the deposit-adjusted base | `frontend/src/app/paper-trading/layout.tsx` (:336 area) |

Nothing was changed without a finding ID. NOT touched (do-no-harm): all US momentum-core paths (screener, optimizer, backtest engine, portfolio_manager decision logic), the SELL cash-credit/round-trip conversions (already correct), kill switch, snapshots.

## B. KRW-fixture test output (criterion 1 — regression-proof)

**PRE-FIX (verbatim, before the paper_trader.py edit):**
```
FAILED backend/tests/test_phase_50_2_multicurrency.py::test_krw_buy_row_persists_usd_total_value
FAILED backend/tests/test_phase_50_2_multicurrency.py::test_krw_sell_row_persists_usd_total_value_and_fee
2 failed, 8 passed in 1.16s
  -> AssertionError: total_value=364175.1 looks like LOCAL currency (KRW), not USD
```
(The US byte-identity test passed pre-fix — proving the do-no-harm invariant held both sides.)

**POST-FIX:**
```
$ python -m pytest backend/tests/test_phase_50_2_multicurrency.py -q
10 passed in 1.07s
```

**Immutable verification command (verbatim):**
```
$ python -m pytest backend/tests -k 'fx or paper_trader or krw' -q
26 passed, 723 deselected, 1 warning in 2.97s
```

**Four-FX-point consistency post-fix (criterion 1):** (1) trade recording — NOW USD (this fix; tests assert); (2) mark-to-market — unchanged-correct (55.1 §2.1#2; positions stored USD); (3) cash ledger — unchanged-correct (55.1 §3 penny-exact; the SELL test asserts the credit is NOT double-converted: expected 1,000 + (364,175.10−364.18)×0.000655, observed match <$0.5); (4) fees — BUY fee was already USD; SELL fee NOW USD (test asserts ~$0.24 vs the old ₩364.18 magnitude).

## C. Playwright captures of the corrected live UI (criterion 2)

Method: same skip-auth :3100 dev instance as 55.1 (operator :3000 untouched, verified 302 after; :3000 is `next dev` and hot-reloads the same fixed files).

| Capture (handoff/current/captures_56.1/) | Evidence |
|---|---|
| `56_1_positions_cockpit_fixed.png` | **NAV card 23,856.94 USD** (was 345,950.68); **Total P&L +19.28%** (was +1,629.75%); **vs SPY +16.79%** (was +1,627.26%); Risk Monitor **Max position 3.0%** (was 1,527.8%) with Position size OK; Allocation donut center **$23,857** (was $345,951); **Currency exposure USD 98.9% / KRW $238.40 1.0%** (was 6.8%/0.1% on the corrupted denominator); subtitle "Autonomous AI-managed virtual fund" (F-13) |
| `56_1_cockpit_KR_holdings_label.png` | KR filter: benchmark card now labeled "**KR HOLDINGS**" (was "VS KOSPI"); DOM probe verbatim: `["KR HOLDINGS"]` matching no `vs KOSPI` |

Value/Fee columns for the 7 historical KR rows still render the stored local magnitudes — by design until the operator approves the backfill (§D); the caveat is documented in the trades-columns header comment and here.

## D. Backfill migration (criterion 3 — OPERATOR-GATED, not executed)

`scripts/migrations/backfill_56_1_kr_trade_values.py` — dry-run by default; `--execute` requires explicit operator approval. Dry-run output (excerpt, all 7 UPDATEs printed):
```
-- backfill_56_1_kr_trade_values: 7 UPDATEs against sunny-might-477607-p8.financial_reports.paper_trades (DRY-RUN)
UPDATE `...paper_trades` SET total_value = 487.87
WHERE trade_id = '6019c11b-...' AND ABS(total_value - 738196.09) < 0.02;
UPDATE `...paper_trades` SET total_value = 486.08, transaction_cost = 0.49
WHERE trade_id = 'd9260ae7-...' AND ABS(total_value - 737259.28) < 0.02 AND ABS(transaction_cost - 737.26) < 0.02;
...
-- dry-run only. Re-run with --execute AFTER operator approval.
```
GIPS disclosure + materiality classification (immaterial to composite returns; material to ledger consumers → tier-3/4 correct-with-disclosure) is persisted in the script docstring. Idempotent: UPDATEs pin trade_id AND the exact corrupted value (re-run matches 0 rows). **Until approved: the 7 rows are FLAGGED (trades-columns comment + this section), not silently kept.**

**OPERATOR ASK:** approve with `python scripts/migrations/backfill_56_1_kr_trade_values.py --execute` (or decline; rows stay flagged).

## E. Build/quality gates

- `cd frontend && npm run build` → compiled successfully (route table emitted).
- `python -c "import ast; ast.parse(...paper_trader.py...)"` → syntax OK.
- Migration dry-run executes with zero GCP deps (deferred import).
- No emojis introduced; palette untouched; minimal diffs only.
