# Experiment Results — phase-50.6 (Multi-market UI)

**Date:** 2026-06-01. **Status:** complete (3 UI deliverables + settings-API field;
build/types/tests green; Playwright skip-auth visual verified all three surfaces; gate
restored).

## What was built (additive; DO-NO-HARM)

- **(a) Backtest scope strip** — `BacktestScopeStrip` (US/USD/SPY chips + mount-guarded
  `isMarketOpen("US")` badge), mounted under the backtest page title. No change to the
  US-only pipeline's cells/tables.
- **(b) Multi-currency NAV-breakdown widget** — `MultiCurrencyNavBreakdown`, client-side
  groups `positions[].market_value` (USD) by `MARKET_CURRENCY[resolveMarket(...)]` (+ cash
  on the All view) → per-currency USD + %NAV bar. Mounted on the positions page below the
  3-card row, scoped to the active market filter.
- **(c) `paper_markets` settings toggle** — `PaperMarketsField` (native fieldset/legend +
  US/EU/KR checkboxes; ≥1 enforced), wired into `/paper-trading/manage` Trading settings;
  persists via `settings_api` (list→CSV→validator round-trip).

## Files changed

| File | Change |
|------|--------|
| `backend/api/settings_api.py` | +`paper_markets` on `FullSettings`/`SettingsUpdate`/`_settings_to_full`/`_FIELD_TO_ENV`; PUT loop serializes lists as CSV. |
| `backend/tests/test_phase_50_6_settings_paper_markets.py` | NEW — 5 tests (exposure, accept, read, CSV round-trip, single). |
| `frontend/src/lib/types.ts` | +`paper_markets?: string[]` on `FullSettings`. |
| `frontend/src/components/MultiCurrencyNavBreakdown.tsx` | NEW widget (client-side currency grouping; JIT-safe dot map). |
| `frontend/src/components/BacktestScopeStrip.tsx` | NEW US/USD/SPY + market-hours strip (mount-guarded). |
| `frontend/src/components/paper-trading/cockpit-helpers.tsx` | +`PaperMarketsField` (native checkbox group; never-empty guard). |
| `frontend/src/app/paper-trading/manage/page.tsx` | import + render `PaperMarketsField`. |
| `frontend/src/app/paper-trading/positions/page.tsx` | import + mount `MultiCurrencyNavBreakdown`. |
| `frontend/src/app/backtest/page.tsx` | import + render `BacktestScopeStrip`. |

## Verification output (verbatim)

```
npx tsc --noEmit                                  -> EXIT_TSC=0
npx eslint <7 changed files>                      -> EXIT_ESLINT=0 (5 warnings/0 errors; mount-guards)
npm run build (next build)                        -> GREEN, 24/24 routes
   (first attempt MODULE_NOT_FOUND = .next contention w/ the kickstarted dev server;
    re-ran clean once the dev server settled)
npm run test (vitest)                             -> 23 files / 178 tests pass
pytest test_phase_50_6 + test_phase_54_1          -> 16 passed
pytest -k "settings or config"                    -> 22 passed (no regression)
```

### Playwright skip-auth (gate restored → 302 after)
- `/paper-trading/manage`: markets fieldset `[US ✓+disabled, EU, KR]`; click EU →
  `[US ✓, EU ✓, KR]`, US unlocks, "unsaved" + Save enabled. (Save NOT clicked — writes .env.)
- `/paper-trading/positions`: `Currency exposure` → `USD $24,023.58 98.5%` (all-US book).
- `/backtest`: scope strip → `US · USD · SPY · OPEN`.
- Console: 0 React errors/warnings, no hydration text.

## Acceptance-criteria mapping (phase-50.6 — VERBATIM masterplan criteria)

| # | Criterion (verbatim) | Result |
|---|----------------------|--------|
| 1 | paper-trading + backtest pages show per-position market/exchange + local currency + a multi-currency NAV breakdown (USD total + per-currency sub-totals) + a market-open/closed indicator | PASS — paper-trading per-position market/exchange + currency shipped (goal-multimarket-ux: `positions-columns.tsx` MarketChip + currency cells); THIS step adds `MultiCurrencyNavBreakdown` (USD total + per-currency) + the market-open/closed indicator is in the gate bar (phase-54); backtest shows the US/USD/SPY + open/closed scope strip (single-market pipeline) |
| 2 | a paper_markets toggle exists in settings UI wired to the backend setting; icons via @/lib/icons, no emoji | PASS — `PaperMarketsField` (native fieldset/checkbox) on `/paper-trading/manage` wired to settings_api (`paper_markets` CSV round-trip); zero emoji (colored dots, not emoji); Phosphor icons via `@/lib/icons` where used elsewhere |
| 3 | cd frontend && npm run build SUCCEEDS with the changes | PASS — `next build` GREEN, 24/24 routes (first attempt failed on `.next` contention w/ the kickstarted dev server; clean re-run succeeded) |
| 4 | live_check_50.6.md records build pass + API wiring + an OPERATOR-TO-CONFIRM visual section | PASS — `handoff/current/live_check_50.6.md` (build/types/API proofs + Playwright visual + operator-to-confirm section) |

## DO-NO-HARM / scope honesty

- Default behavior byte-identical: `paper_markets` default `["US"]`; the loop unchanged
  unless the operator flips the toggle. Backtest pipeline (US-only ML) untouched (strip
  is additive). NAV widget is client-side only (no API shape change).
- No money-path / risk / kill_switch code touched. No `.env` hand-edit (settings_api owns
  the env write). No new dependency. No emoji; navy palette; JIT-safe literal class maps;
  native HTML checkbox group (W3C) over bespoke ARIA. Reused `format.ts` (no currency fork).
- The `isMarketOpen` US badge reuses the shared helper verbatim (same heuristic the cockpit
  uses); it is a labeled UI hint (holiday-blind; backend gate authoritative).
