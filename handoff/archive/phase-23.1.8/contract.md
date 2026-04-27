---
step: phase-23.1.8
title: Positions table reactivity — live Market Value + P&L + Stop Loss default
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && python -c "from backend.config.settings import get_settings; s = get_settings(); assert hasattr(s, \"paper_default_stop_loss_pct\"); assert 1.0 <= s.paper_default_stop_loss_pct <= 50.0; from backend.services.portfolio_manager import _extract_stop_loss; lite = {\"price_at_analysis\": 100.0}; risk = {\"reason\": \"strong momentum\"}; sl = _extract_stop_loss(risk, lite, settings=s); assert sl is not None and 50.0 <= sl <= 99.0, f\"expected stop 50-99, got {sl}\"; assert _extract_stop_loss({}, {}, settings=s) is None, \"no price -> None\"; gemini = {\"price_at_analysis\": 100.0}; gemini_risk = {\"risk_limits\": {\"stop_loss\": 87.5}}; sl2 = _extract_stop_loss(gemini_risk, gemini, settings=s); assert sl2 == 87.5, f\"explicit stop should win over default, got {sl2}\"; print(f\"ok default={s.paper_default_stop_loss_pct}% lite_stop={sl:.2f} gemini_stop={sl2}\")"'
research_brief: handoff/current/phase-23.1.8-research-brief.md
---

# Contract — phase-23.1.8

## Hypothesis

Two surgical edits make the Positions table genuinely reactive to live prices and ensure every BUY carries a stop-loss safety net:

1. **Frontend** — derive `liveMarketValue = livePrice × quantity` and `livePnlPct = ((live_mv − cost_basis) / cost_basis) × 100` on each render from the existing 30s yfinance live-price polling. No backend or BQ change.

2. **Backend** — `_extract_stop_loss` adds a settings-driven default (8% below entry price, O'Neil canon) when neither the Gemini `risk_limits.stop_loss` nor the lite-path analyzer provides a value. New `paper_default_stop_loss_pct: float = 8.0` in `settings.py` (range 1.0-50.0) gives the operator a knob.

## Plan

1. **Frontend `frontend/src/app/paper-trading/page.tsx`** (lines 555-600 area):
   - Inside the `positions.map((pos) => {` block, after the `ageLabel` declaration, add 3 inline expressions: `livePrice`, `liveMarketValue`, `livePnlPct` (with safe fallbacks to `pos.market_value` / `pos.unrealized_pnl_pct` when no live price).
   - Replace `<Dollar value={pos.market_value} />` with `<Dollar value={liveMarketValue} />`.
   - Replace `<PnlBadge value={pos.unrealized_pnl_pct} />` with `<PnlBadge value={livePnlPct} />`.
   - **No `useMemo`** — React 19 Compiler handles ~10-row recomputation; adding memoization adds boilerplate + stale-closure risk for zero measurable benefit.

2. **Backend `backend/config/settings.py`** — add field after `paper_trailing_dd_limit_pct` (around line 174):
   ```python
   paper_default_stop_loss_pct: float = Field(
       8.0, ge=1.0, le=50.0,
       description="Default stop-loss as % below entry price when analysis does not provide one (lite-path BUY). O'Neil canonical: 7-8%."
   )
   ```

3. **Backend `backend/services/portfolio_manager.py::_extract_stop_loss`**:
   - Add `settings: Optional["Settings"] = None` parameter
   - After the existing `risk_limits.stop_loss` and `risk_limits.stop_loss_pct` chains, add a final fallback: when both return None AND `settings` provided AND `analysis.price_at_analysis` available, return `price × (1 − default_pct/100)`.
   - Update the 1 caller in `decide_trades` (around line 146) to pass `settings=settings`.

4. **Tests** at `tests/services/test_extract_stop_loss.py`:
   - Explicit `risk_limits.stop_loss=87.5` wins over default
   - Explicit `risk_limits.stop_loss_pct=10` produces `price * 0.90`
   - Lite path (no risk_limits, just `risk_assessment={"reason": ...}`) AND no price returns None
   - Lite path WITH `price_at_analysis=100` AND settings → returns `100 * 0.92 = 92.0` (using default 8%)
   - Settings-None preserves old behavior (returns None when no explicit stop)
   - Default override: when settings has `paper_default_stop_loss_pct=15`, returns `price × 0.85`

5. **Frontend tests/type-check** — `cd frontend && npx tsc --noEmit` must remain clean. (No new TS unit test file — the change is too small to warrant; tsc is the regression guard.)

## Out of scope

- Server-side periodic mark_to_market more than once per cycle (Phase 2 — current daily cadence is fine; live UI now compensates)
- Per-ticker custom stop_loss in the Settings UI (Phase 2 — single global default sufficient for v1)
- Exposing the new setting in the Settings page UI (Phase 2 — backend default works without UI surfacing)
- Trailing stop-loss logic (Phase 2 — fixed % default for now)
- Frontend "stale data" warning when `live?.price` is null and `pos.market_value` is also stale (Phase 2)

## Files modified

- `frontend/src/app/paper-trading/page.tsx` — 3 lines added + 2 cell replacements
- `backend/config/settings.py` — 1 new Field (5-line addition)
- `backend/services/portfolio_manager.py` — `_extract_stop_loss` signature + fallback chain + 1 caller update
- `tests/services/test_extract_stop_loss.py` — NEW (~6 tests)

## Verification

The front-matter command does five things in one shot:
1. Asserts the new `paper_default_stop_loss_pct` field exists on `Settings` and is in range
2. Calls `_extract_stop_loss` with the lite-path shape (only `price_at_analysis` + `risk_assessment.reason`) and asserts it now returns a finite value (the default kicks in)
3. Asserts no-price input returns None (graceful degrade preserved)
4. Asserts explicit `risk_limits.stop_loss=87.5` STILL takes priority over the default (chain ordering preserved)
5. Prints the computed stops for visual confirmation

Frontend behavior is verified by `cd frontend && npx tsc --noEmit` exiting 0 (the type contracts on `pos.cost_basis`, `pos.quantity`, `Dollar` props all hold).

## References

- `handoff/current/phase-23.1.8-research-brief.md` — full brief (371 lines, 5 sources read in full, gate_passed: true)
- `frontend/src/app/paper-trading/page.tsx:555-600` — positions.map render
- `backend/services/portfolio_manager.py:229-239` — current `_extract_stop_loss`
- `backend/config/settings.py:140-174` — paper-trading settings block
- O'Neil "How to Make Money in Stocks" — canonical 7-8% momentum stop
- `quant-investing.com` 85-year backtest: 10% stop reduced max monthly loss from −49.79% to −11.34%
