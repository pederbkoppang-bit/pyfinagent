# Research Brief — phase-50.6 (Multi-market UI)

**Tier:** moderate. **Date:** 2026-06-01. **Gate: PASSED** (`gate_passed: true`).

> NOTE (audit trail): the `researcher` subagent (`abfe6ce9161055b73`) ran the gate
> FIRST and wrote the full brief here; the scheduled `mas-harness` optimizer cron
> (StartInterval 1800) then OVERWROTE this rolling file at 17:04 local with an
> optimizer "cycle iteration 1" brief — a handoff-file collision between the
> scheduled optimizer and the manual masterplan cycle. This file is reconstructed
> faithfully from the researcher's returned summary + gate envelope. The clobbering
> cron has been booted out for the remainder of the run (re-enable:
> `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist`).

## Gate envelope (as returned by the researcher)

```json
{"tier":"moderate","external_sources_read_in_full":7,"snippet_only_sources":10,
 "urls_collected":17,"recency_scan_performed":true,"internal_files_inspected":13,
 "gate_passed":true}
```

## Decisive findings

1. **The backtest page is a US-only / USD-hardcoded ML walk-forward pipeline** —
   distinct from the live multi-market paper loop. There is no multi-market backtest to
   build. Criterion (a) is satisfied by a small ADDITIVE context strip labelling the
   backtest as US / USD / SPY + an `isMarketOpen("US")` open/closed badge, reusing
   `format.ts`. Do NOT refactor its `$`-literal cells / baseline table (DO-NO-HARM R2).
2. **The NAV-breakdown widget needs NO backend change.** Each `PaperPosition` already
   carries `market` + USD `market_value` (`paper_trader.py:298-299,311`); group
   `positions[].market_value` (USD) by `resolveCurrency`/`MARKET_CURRENCY[resolveMarket]`
   client-side → USD total + per-currency sub-totals + %NAV. Point-in-time snapshot (no
   retro-FX revaluation; PortfolioPilot convention). Mirror `PortfolioAllocationDonut` +
   its `DOT_BG_CLASS` JIT-safe pattern.
3. **The settings toggle is the real gap (criterion c).** `backend/api/settings_api.py`
   `FullSettings` (:60) + `SettingsUpdate` (:123) did NOT expose `paper_markets`; add it +
   `_settings_to_full` (:316) + `_FIELD_TO_ENV` (:253) with **list→CSV serialization** in
   the PUT loop (the 54.1 `settings.py::_parse_paper_markets` validator parses CSV on
   read). `types.ts` (:524-575): add `paper_markets?: string[]`. UI: a native
   `<fieldset><legend>` checkbox group (W3C APG; native over bespoke ARIA), wired to the
   existing `manageSettings`/`manageDirty`/`setManageDirty` diff flow at
   `/paper-trading/manage`.
4. **Reuse, don't rebuild.** `format.ts` (`formatCurrency`/`formatUsd`/`resolveCurrency`/
   `resolveMarket`/`MARKET_CURRENCY`/`MARKET_DOT_CLASS`/`MARKET_BENCHMARK_LABEL`/
   `isMarketOpen`), `cockpit-helpers.tsx` (`Dollar`/`MarketChip`/`PaperSettingNum`),
   `PortfolioAllocationDonut.tsx`. `backend/services/fx_rates.py` exists but is NOT needed
   for the snapshot widget. `types.ts:608-657` `PaperPortfolio`/`PaperPosition` carry
   `market` + `base_currency`.

## External consensus (7 sources read in full; 17 URLs; recency scan done)

Dual local+base display per holding, base-currency headline NAV + per-currency breakdown,
locale-aware `Intl.NumberFormat` (narrowSymbol; KRW at 0 decimals), no retro-FX
revaluation, and native HTML controls over bespoke ARIA for the multi-select (W3C
fieldset/legend + checkbox APG). Tremor 3.18.7 is in the stack but a native checkbox group
is simpler + more on-pattern. Recency scan (2024-2026): no source contradicts the
approach; market-session indicators + consolidated filter bars are current shipping norms.

## DO-NO-HARM risks (carried into the contract)

- R1 homepage/cockpit regression — reuse the shipped helpers, don't fork.
- R2 backtest page — ADD a strip only; never touch its USD-literal cells/baseline table.
- R3 settings round-trip — list→CSV must parse back through `_parse_paper_markets`
  (assert with a test).
- R4 hydration — the market-hours badge reads `new Date()`; mount-guard it.
- R5 empty markets — the toggle must never persist an empty universe (≥1 enforced).
- R6 visual-only correctness — Playwright skip-auth visual is the real acceptance evidence.

## References

`frontend/src/lib/format.ts`, `cockpit-helpers.tsx`, `PortfolioAllocationDonut.tsx`,
`frontend/src/app/{backtest,paper-trading/positions,paper-trading/manage,settings}/page.tsx`,
`frontend/src/lib/types.ts`, `backend/api/settings_api.py`, `backend/services/fx_rates.py`,
`docs/runbooks/browser-mcp.md`. External: W3C APG (radio/checkbox/fieldset), Intl.NumberFormat
(MDN), multi-currency portfolio display (PortfolioPilot/IBKR), market-session indicators.
