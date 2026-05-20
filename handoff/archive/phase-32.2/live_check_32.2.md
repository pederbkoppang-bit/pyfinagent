# Live Check — phase-32.2 HWM-Trailing Stop + Kaminski-Lo Adversarial Guard

**Date:** 2026-05-21 (cycle UTC ~00:37)
**Verification target (from masterplan):** BQ row from `paper_positions` showing `stop_loss_price > avg_entry_price * 0.95` for at least one position whose `mfe_pct > 20%` (trail moved stop above entry-anchored). AND a mean-reversion-flagged entry whose `stop_loss_price` is unchanged from entry-anchored (guard active).

## Sequence of operations executed

1. Migration `scripts/migrations/phase_32_2_add_entry_strategy.py --apply` — Job IDs (DDL) + `321de5e7-...` (backfill UPDATE). 11 rows updated with `entry_strategy='momentum'`.
2. Re-ran migration to confirm idempotency — `ALTER TABLE` no-op, 0 rows needing backfill.
3. Invoked `PaperTrader.mark_to_market()` from Python against real BQ + yfinance. NAV=$22,454.30, 11 positions.
4. Queried `paper_positions` post-MTM via `mcp__claude_ai_Google_Cloud_BigQuery__execute_sql_readonly`.
5. Invoked `PaperTrader._advance_stop` from the Python REPL with three synthetic positions (mean_reversion, pairs, momentum) to exercise the adversarial guard live against the production helper.

## Result table — positive trail verification (BQ verbatim)

All 11 rows have `entry_strategy='momentum'` (backfill default). The trail formula is `new_trail = entry × (1 + mfe_pct/100) × (1 - 8/100)`. The monotonic-max gate accepted every new trail because all prior stops were at the breakeven level (= entry) from 32.1.

| Ticker | Strategy | Entry | Now | MFE % | Stop AFTER 32.2 | stop_vs_entry % | stop_vs_peak % | Classification |
|---|---|---|---|---|---|---|---|---|
| SNDK | momentum | 989.90 | 1392.56 | +57.64 | **1435.60** | +45.02 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| MU | momentum | 506.65 | 731.99 | +57.62 | **734.68** | +45.01 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| INTC | momentum | 82.57 | 118.96 | +53.85 | **116.87** | +41.54 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| COHR | momentum | 320.91 | 358.50 | +28.36 | **378.95** | +18.09 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| WDC | momentum | 404.00 | 459.62 | +27.75 | **474.82** | +17.53 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| LITE | momentum | 881.64 | 868.07 | +19.50 | **969.29** | +9.94 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| ON | momentum | 98.40 | 110.21 | +19.49 | **108.17** | +9.93 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| DELL | momentum | 216.09 | 242.93 | +19.14 | **236.84** | +9.60 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| GLW | momentum | 175.89 | 180.69 | +19.05 | **192.65** | +9.53 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| KEYS | momentum | 330.19 | 342.08 | +11.47 | **338.61** | +2.55 | -8.00 | TRAILED_ABOVE_BREAKEVEN |
| GEV | momentum | 1078.49 | 1024.52 | +3.15 | 992.22 | -8.00 | -10.81 | BELOW_BREAKEVEN (MFE < threshold, breakeven hasn't fired yet -- correct) |

**5 of 11 positions have MFE > 20%** (SNDK, MU, INTC, COHR, WDC). All 5 satisfy `stop_loss_price > avg_entry_price * 0.95`. The live_check's positive-trail requirement is met five-fold.

**Trail mechanics validated:** stop_vs_peak = -8.00% across ALL trailed positions (the configured `paper_trailing_stop_pct`). The geometry holds: stop = peak × 0.92 in every case.

## Mean-reversion guard verification (NO live MR position; verified via production-helper REPL)

Production has zero `entry_strategy='mean_reversion'` and zero `entry_strategy='pairs'` positions today — the autonomous loop's universe has been emitting momentum / triple-barrier entries, and the backfill defaulted all 11 existing rows to `'momentum'`. To demonstrate the Kaminski-Lo Proposition 2 guard fires on the LIVE PRODUCTION CODE PATH (not just in unit tests), three synthetic positions were fed through the loaded `PaperTrader._advance_stop` via a Python REPL after the migration + live MTM:

```python
trader._advance_stop(
    {"ticker": "MR_SIM", "avg_entry_price": 100.0, "stop_loss_price": 100.0,
     "stop_advanced_at_R": "2026-05-20T22:00:00+00:00",
     "entry_strategy": "mean_reversion"},
    new_mfe=30.0,
)
# -> (None, None)        [trail SKIPPED -- Kaminski-Lo guard active]

trader._advance_stop({...same with entry_strategy="pairs"}, new_mfe=30.0)
# -> (None, None)        [trail SKIPPED -- guard active]

trader._advance_stop({...same with entry_strategy="momentum"}, new_mfe=30.0)
# -> (119.60..., None)   [trail FIRED -- production formula entry*1.3*0.92]
```

Verbatim REPL output:
```
mean_reversion guard result (expect (None, None)): (None, None)
pairs guard result (expect (None, None)): (None, None)
momentum trail result (expect (119.60, None)): (119.60000000000001, None)
```

Coverage by deterministic test cases at `backend/tests/test_phase_32_2_hwm_trailing.py`:
- `test_kaminski_lo_guard_mean_reversion` — PASS
- `test_kaminski_lo_guard_pairs` — PASS
- `test_default_momentum_trails_when_entry_strategy_is_none` — PASS (the fail-CLOSED conservative default)

## Verification command output

```
$ python -m pytest backend/tests/test_phase_32_2_hwm_trailing.py -v
6 passed in 1.01s

$ grep -n 'mean_reversion' backend/services/paper_trader.py
780:            if entry_strategy in {"mean_reversion", "pairs"}:

$ grep -n 'trailing_stop_pct' backend/config/settings.py
339:    paper_trailing_stop_pct: float = Field(
```

## Success criteria check (all 7 PASS)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `trailing_logic_ports_from_signals_server` | PASS | algorithm + formula ported from `signals_server.py:1052-1154`; adapted from "trigger exit" to "advance stop" |
| 2 | `paper_trailing_stop_pct_setting_default_8` | PASS | `settings.py:339`, `Field(8.0, ge=0.5, le=50.0)` |
| 3 | `stop_loss_price_monotonic_max_never_down` | PASS | helper returns `(None, None)` when `new_trail <= current_stop`; `test_trail_monotonic_never_moves_down` covers regression case |
| 4 | `adversarial_guard_skips_mean_reversion_and_pairs_entries` | PASS | `paper_trader.py:780` `if entry_strategy in {"mean_reversion","pairs"}: return (None, None)`; REPL + unit tests confirm |
| 5 | `fail_closed_conservative_default_is_apply_trail` | PASS | `entry_strategy = (pos.get("entry_strategy") or "").lower().strip()` → `""` for None/unknown; `""` not in guard set → trail applied; `test_default_momentum_trails_when_entry_strategy_is_none` confirms |
| 6 | `entry_strategy_field_or_lookup_implemented` | PASS | Option A executed — column added via idempotent migration, 11 rows backfilled with `'momentum'`, `_POSITION_RT_FIELDS` extended |
| 7 | `unit_test_3_cases_pass` | PASS | 6 tests pass (spec floor was 3) |
