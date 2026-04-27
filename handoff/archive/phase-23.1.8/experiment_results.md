---
step: phase-23.1.8
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: see contract.md front-matter
---

# Experiment Results — phase-23.1.8

## What was built

Two surgical edits made the Positions table genuinely reactive (Market Value + P&L now recompute on every 30s yfinance live-price tick) and ensured every BUY now carries a stop-loss safety net (8% below entry per O'Neil canon when neither the Gemini nor lite analyzer proposes one).

## Files modified

| File | Change |
|---|---|
| `frontend/src/app/paper-trading/page.tsx` | Inside `positions.map` block: 4 new inline expressions (`livePrice`, `liveMarketValue`, `liveCostBasis`, `livePnlPct`) derived from `live?.price` + `pos.quantity` + `pos.cost_basis`. Replaced `<Dollar value={pos.market_value}/>` → `<Dollar value={liveMarketValue}/>` and `<PnlBadge value={pos.unrealized_pnl_pct}/>` → `<PnlBadge value={livePnlPct}/>`. Falls back to BQ values when no live price available. No `useMemo` (React 19 Compiler handles ~10-row recompute; per research brief). |
| `backend/config/settings.py` | NEW field `paper_default_stop_loss_pct: float = Field(8.0, ge=1.0, le=50.0)` after `paper_trailing_dd_limit_pct` (line 174). |
| `backend/services/portfolio_manager.py` | `_extract_stop_loss` adds `settings: Optional["Settings"] = None` parameter + final fallback chain: when neither `risk_limits.stop_loss` nor `risk_limits.stop_loss_pct` returns a value, AND `settings` provided AND `analysis.price_at_analysis` available, return `price * (1 − default_pct/100)`. Caller in `decide_trades` (line 146) updated to pass `settings=settings`. |
| `tests/services/test_extract_stop_loss.py` | NEW (10 tests covering: explicit-stop priority, stop_loss_pct path, lite-path default fallback, settings override, no-settings backward compat, no-price graceful None, empty risk_assessment uses default, falsy stop_loss falls through, missing-attr settings, non-numeric default). |

## Verbatim verification command output

```
$ source .venv/bin/activate && python -c "from backend.config.settings import get_settings; s = get_settings(); assert hasattr(s, 'paper_default_stop_loss_pct'); ..."
ok default=8.0% lite_stop=92.00 gemini_stop=87.5
exit=0
```

The verification proves:
1. `paper_default_stop_loss_pct` field exists and is 8.0 (default)
2. Lite-path BUY (just `risk_assessment={"reason": ...}` + `price_at_analysis=100`) now produces `stop_loss_price = 92.0` instead of None
3. No-price input still returns None (graceful degrade preserved)
4. Explicit `risk_limits.stop_loss=87.5` still wins over the 8% default (chain ordering preserved — backward compat)

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/services/ tests/api/test_settings_api_signal_stack.py -v --no-header -q
collected 125 items
tests/services/test_extract_stop_loss.py ..........            [  8%]
tests/services/test_macro_regime.py ............                [ 17%]
tests/services/test_meta_scorer.py ..............               [ 28%]
tests/services/test_news_screen.py .....................        [ 45%]
tests/services/test_pead_signal.py ..................           [ 60%]
tests/services/test_sector_calendars.py ................        [ 72%]
tests/services/test_signal_attribution.py ....................  [ 88%]
tests/api/test_settings_api_signal_stack.py ..............      [100%]
============================= 125 passed in 0.34s ==============================
```

10 new + 115 prior = 125/125 tests pass. Zero regression across 8 cycles (phase-23.1.1 → 23.1.8).

## Frontend type-check

```
$ cd frontend && npx tsc --noEmit
(silent — 0 errors)
```

Type contracts verified end-to-end: `pos.quantity: number`, `pos.cost_basis?: number | null`, `pos.market_value: number | null`, `Dollar` and `PnlBadge` props all hold under the new derived expressions.

## What the operator sees on tomorrow's market open

Before this cycle (the screenshot in the user's question):
- 10 positions all showing **+0.00% P&L** despite Current ≠ Entry (because Market Value was stale BQ value)
- Market Value all stuck at $949.xx (the entry-time mark-to-market value)
- Stop Loss showing **"—"** for every position

After this cycle, when the market opens at 09:30 ET tomorrow:
- The 30s yfinance live-price polling already runs (`useLivePrices` at line 267)
- Each tick updates `live.price` → `liveMarketValue = live.price × pos.quantity` recomputes → `<Dollar value={liveMarketValue}/>` re-renders
- P&L recomputes the same way: `((live_mv − cost_basis) / cost_basis) × 100`
- Tomorrow's BUYs (and any future BUY with lite Claude) will have `stop_loss_price` populated automatically (8% below entry by default; operator can override via `PAPER_DEFAULT_STOP_LOSS_PCT` env var)

## Backward compatibility

- **Existing positions in BQ**: the 10 positions already in BQ have `stop_loss_price=None`. They'll continue to render "—" in the Stop Loss column until they're sold + re-bought (new positions inherit the default). No retroactive update — that's intentional; the operator chose those entry prices without a stop, and we don't second-guess history.
- **Old code paths that called `_extract_stop_loss` without settings**: still work (returns None when no chain matches; `settings=None` is the default).
- **Frontend**: when `live?.price` is null (yfinance offline / no quote yet), falls back to `pos.market_value` / `pos.unrealized_pnl_pct` from BQ — same as today.

## Out of scope (per contract)

- More frequent server-side `mark_to_market` (Phase 2 — current daily cadence + live UI is sufficient)
- Per-ticker custom stop_loss UI (Phase 2 — single global default for v1)
- Surfacing the new setting in the Settings page UI (Phase 2 — env var override works for now)
- Trailing stop-loss logic (Phase 2 — fixed % default for now)
- Retroactive update of existing positions to apply the new default (intentionally not done — see Backward compatibility above)

## Honest disclosure

The 10 positions in the table tonight will keep showing "—" for Stop Loss because they were booked BEFORE this cycle landed. Tomorrow's first BUY will be the first row to show a real stop-loss value. The frontend Market Value + P&L fix DOES apply to the existing 10 positions immediately on next render — they'll start ticking the moment the market opens.

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit
3. **Restart frontend** so the React bundle picks up the new positions.map logic (backend change is via settings hot-reload but frontend bundle is built)
