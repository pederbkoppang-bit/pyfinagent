# live_check 50.6 — Multi-market UI (build/types/API proofs + operator visual)

**Captured:** 2026-06-01 (local). Additive UI + a settings-API field; DO-NO-HARM to
the shipped cockpit + the US-only backtest pipeline.

## Build / types / test proofs

```
npx tsc --noEmit                      -> EXIT 0 (no type errors)
npx eslint <7 changed files>          -> EXIT 0 (5 warnings, 0 errors; all the
                                         documented mount-guard set-state-in-effect)
npm run build (next build)            -> GREEN, 24/24 routes (route table printed)
npm run test (vitest)                 -> 23 files / 178 tests pass
pytest test_phase_50_6_settings...    -> 5 pass (paper_markets exposure + CSV round-trip)
pytest test_phase_54_1 + 50_6         -> 16 pass ; settings/config regression 22 green
```

## API proof (criterion c — settings round-trip)

`paper_markets` now flows through `backend/api/settings_api.py`:
- `FullSettings.paper_markets: list[str] = ["US"]` (default unchanged) + `SettingsUpdate.paper_markets: Optional[list[str]]`.
- `_settings_to_full` reads it; `_FIELD_TO_ENV["paper_markets"]="PAPER_MARKETS"`; the PUT
  loop serializes a list as **CSV** (`",".join`).
- Read side: the 54.1 `settings.py` `_parse_paper_markets` validator parses CSV → list.
- Round-trip test (no live `.env` write): `["US","EU","KR"] → "US,EU,KR" → ["US","EU","KR"]`.

## Live visual verification (Playwright skip-auth Path A; gate restored after → 302)

Evidence screenshots (gitignored, repo root): `cockpit-50-6-manage-markets.png`,
`cockpit-50-6-currency-widget.png`, `cockpit-50-6-backtest-strip.png`.

1. **`/paper-trading/manage` — live-loop markets toggle (criterion c):**
   `browser_evaluate` → fieldset present; boxes `[US checked+disabled, EU off, KR off]`
   (US disabled = the last-checked-box guard, never-empty). Clicked **EU** →
   `[US checked+enabled, EU checked, KR off]`, "unsaved" shown, Save enabled.
   (Did NOT click Save — that writes `.env`; the client-state toggle is the proof.)
2. **`/paper-trading/positions` — multi-currency NAV breakdown (criterion b):**
   widget present; all-US book → one row `USD $24,023.58 98.5%` (graceful
   single-currency; grows to EUR/KRW rows when international holdings exist).
3. **`/backtest` — scope strip (criterion a):**
   `[aria-label="Backtest scope..."]` present → `US · USD · SPY · OPEN`
   (US/USD/SPY labels + mount-guarded `isMarketOpen("US")` badge). The backtest's own
   cells/baseline table were NOT touched.
4. Console: 0 React errors, 0 warnings, no hydration text (favicon-404 + auth-session-500
   are pre-existing under skip-auth, not from this change).

## OPERATOR TO CONFIRM (visual, behind the NextAuth wall)

When you next log in (real session), please eyeball:
- **`/paper-trading/manage` → Trading settings → "Live-loop markets"**: the US/EU/KR
  checkboxes; toggling persists on Save (writes `PAPER_MARKETS` CSV). **Leaving it at US
  only is byte-identical to today.** Changing it steers which markets the live loop
  screens/trades (international only after the data-quality gate).
- **`/paper-trading/positions` → "Currency exposure"** card (below the 3-card row): the
  per-currency USD split. With an all-US book it shows one USD row.
- **`/backtest`** header: the `US · USD · SPY · OPEN/CLOSED` scope strip under the title.

## DO-NO-HARM / scope

- Backtest page: ADD-only strip; existing USD-literal cells + baseline table untouched.
- NAV widget: client-side only; no `/api/paper-trading/portfolio` shape change; graceful
  single-market/empty.
- `paper_markets` default unchanged (`["US"]`); the loop is byte-identical unless the
  operator changes the toggle. No `.env` hand-edit (the PUT path uses settings_api's own
  `_update_env_var`). No money-path logic change. No emoji; navy palette; JIT-safe maps.
