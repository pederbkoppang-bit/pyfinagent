# Live Check — phase-32.1 Breakeven-Stop Ratchet

**Date:** 2026-05-21 (cycle UTC ~22:15)
**Verification target (from masterplan):** BQ row from `financial_reports.paper_positions` showing `stop_advanced_at_R` populated AND `stop_loss_price = avg_entry_price` (NOT `entry × 0.92`) for at least one current high-MFE position. Query run AFTER migration is applied AND after at least one `mark_to_market` invocation.

## Sequence of operations executed

1. Migration `scripts/migrations/phase_32_1_add_stop_advanced_at_R.py --apply` — Job ID `3d50be31-0824-4f4f-ab8a-e908f4a0763a`. Verification OK: `[('stop_advanced_at_R', 'STRING')]`.
2. Re-ran migration to confirm idempotency — Job ID `64fc1510-64fe-4199-b50b-b1894b3b504e`. Same Verification OK output, zero schema change.
3. Invoked `PaperTrader.mark_to_market()` from a Python REPL via `Settings()` + `BigQueryClient(settings)` (real BQ + yfinance live prices). Result: NAV=$22,454.30, positions_value=$12,449.52, position_count=11.
4. Queried `financial_reports.paper_positions` (post-MTM) via `mcp__claude_ai_Google_Cloud_BigQuery__execute_sql_readonly`.

## Result table (verbatim from BQ, ordered by MFE descending)

| Ticker | Sector | Entry | Now | MFE % | Stop BEFORE (per phase-31.0 baseline) | Stop AFTER (current) | stop_advanced_at_R | Classification |
|---|---|---|---|---|---|---|---|---|
| SNDK | Technology | 989.90 | 1392.56 | +57.64 | NULL (NO_STOP) | **989.90 (= entry)** | 2026-05-20T22:15:41.517413+00:00 | RATCHET_FIRED_AT_ENTRY |
| MU | Technology | 506.65 | 731.99 | +57.62 | 466.12 (-8% static) | **506.65 (= entry)** | 2026-05-20T22:14:54.803717+00:00 | RATCHET_FIRED_AT_ENTRY |
| INTC | Technology | 82.57 | 118.96 | +53.85 | NULL (NO_STOP) | **82.57 (= entry)** | 2026-05-20T22:15:21.426576+00:00 | RATCHET_FIRED_AT_ENTRY |
| COHR | Technology | 320.91 | 358.50 | +28.36 | 295.24 (-8% approx) | **320.91 (= entry)** | 2026-05-20T22:15:09.754406+00:00 | RATCHET_FIRED_AT_ENTRY |
| WDC | Technology | 404.00 | 459.62 | +27.75 | NULL (NO_STOP) | **404.00 (= entry)** | 2026-05-20T22:15:45.869664+00:00 | RATCHET_FIRED_AT_ENTRY |
| LITE | Technology | 881.64 | 868.07 | +19.50 | NULL (NO_STOP) | **881.64 (= entry)** | 2026-05-20T22:15:36.666217+00:00 | RATCHET_FIRED_AT_ENTRY |
| ON | Technology | 98.40 | 110.21 | +19.49 | NULL (NO_STOP) | **98.40 (= entry)** | 2026-05-20T22:15:14.951978+00:00 | RATCHET_FIRED_AT_ENTRY |
| DELL | Technology | 216.09 | 242.93 | +19.14 | NULL (NO_STOP) | **216.09 (= entry)** | 2026-05-20T22:15:26.129302+00:00 | RATCHET_FIRED_AT_ENTRY |
| GLW | Technology | 175.89 | 180.69 | +19.05 | NULL (NO_STOP) | **175.89 (= entry)** | 2026-05-20T22:15:31.319416+00:00 | RATCHET_FIRED_AT_ENTRY |
| KEYS | Technology | 330.19 | 342.08 | +11.47 | 303.78 (-8% static) | **330.19 (= entry)** | 2026-05-20T22:14:59.894311+00:00 | RATCHET_FIRED_AT_ENTRY |
| GEV | Industrials | 1078.49 | 1024.52 | +3.15 | 992.22 (-8% approx) | 992.22 (unchanged) | NULL | STATIC_8PCT_BELOW_ENTRY (below 8% threshold, correctly skipped) |

## Headline

**10 of 11 positions ratcheted to breakeven on the first live MTM.** The 11th (GEV, MFE +3.15%) correctly did NOT fire because its MFE is below the 8% threshold — confirming both the trigger and the no-false-fire branch.

Specific cross-checks vs phase-31.0 baseline:
- All 7 prior NO_STOP positions (SNDK, INTC, WDC, LITE, ON, DELL, GLW) now have stops at `entry_price`. The phase-31.0 audit baseline showed these positions had no stop coverage at all; they now have a breakeven floor.
- Static-8%-entry positions (MU, COHR, KEYS) had their stops advanced UP from the entry-anchored -8% level to the entry itself.
- The high-MFE positions identified in the audit (SNDK +57.64%, MU +57.62%, INTC +53.85%, COHR +28.36%, WDC +27.75%) now have a floor that locks in zero-loss instead of being entry-stop-anchored.

## Verification command output

```
verification.command per masterplan::phase-32.1.verification.command:
$ python -m pytest backend/tests/test_phase_32_1_breakeven_ratchet.py -v
7 passed in 1.02s

$ grep -n '_advance_stop' backend/services/paper_trader.py
449:            new_stop, advance_iso = self._advance_stop(pos, new_mfe)
749:    def _advance_stop(

$ python -c "import ast; ast.parse(open('backend/services/paper_trader.py').read())"
(no output -- parse OK)
```

## Success criteria check (verbatim from masterplan)

| # | Criterion | Status |
|---|---|---|
| 1 | `_advance_stop_helper_in_paper_trader` | PASS — helper at `paper_trader.py:749` |
| 2 | `called_from_mark_to_market_before_mfe_write` | PASS — call at `paper_trader.py:449` after `new_mfe = max(...)` and before `pos.update({...})` |
| 3 | `mfe_geq_1R_mutates_stop_to_entry` | PASS — 10 of 11 positions show `stop_loss_price = avg_entry_price` post-MTM |
| 4 | `stop_advanced_at_R_audit_field_added_nullable` | PASS — column added (STRING NULLABLE), idempotent migration |
| 5 | `backfill_high_mfe_positions_on_first_run` | PASS — all 7 NO_STOP positions + 3 static-8% positions advanced to entry on first MTM |
| 6 | `unit_test_4_cases_pass` | PASS — 7 tests pass (spec required ≥4) |
| 7 | `no_regression_check_stop_losses` | PASS — full sweep 266 passed, 1 skipped, 0 failures |
