# Contract — phase-50.6 (Multi-market UI)

**Date:** 2026-06-01. **Tier:** moderate. **Step:** phase-50.6 (P3).

> NOTE: this file was clobbered by the scheduled `mas-harness` optimizer cron at 17:04
> (a handoff-file collision). Restored here; the cron has been booted out for the run.
> Correction vs the first draft: the success criteria below are now copied **VERBATIM**
> from `.claude/masterplan.json` phase-50.6 `verification.success_criteria` (the first
> draft paraphrased them — Q/A flagged it).

## N* delta (N* = Profit − Risk − Burn)

**Risk↓ / operability** (speculative): surfaces multi-market scope (currency, market
hours) + a per-currency NAV breakdown + an operator control for which markets the live
loop trades. No P delta; no money-path logic change (the `paper_markets` toggle writes an
existing field already honored by the loop; default `["US"]` unchanged).

## Research-gate summary

`researcher` ran FIRST (gate **PASSED**: 7 sources read in full, 17 URLs, recency scan,
13 internal files). Brief: `handoff/current/research_brief.md` (reconstructed after the
optimizer clobber; gate envelope preserved). Decisive: backtest page is US-only/USD ML
(criterion (a) = additive strip, no refactor); NAV widget = client-side, no backend
change; settings toggle = the real gap (settings_api `paper_markets` + CSV round-trip);
reuse `format.ts` + `cockpit-helpers` + the donut pattern.

## Immutable success criteria — VERBATIM from masterplan phase-50.6 (do NOT edit)

1. paper-trading + backtest pages show per-position market/exchange + local currency + a
   multi-currency NAV breakdown (USD total + per-currency sub-totals) + a
   market-open/closed indicator
2. a paper_markets toggle exists in settings UI wired to the backend setting; icons via
   @/lib/icons, no emoji
3. cd frontend && npm run build SUCCEEDS with the changes
4. live_check_50.6.md records build pass + API wiring + an OPERATOR-TO-CONFIRM visual
   section

(`verification.live_check`: REQUIRED -- build pass + API-wiring proofs + operator visual
confirmation of the market/currency UI.)

### Criterion-1 coverage note (per-position market/exchange + local currency)

The **paper-trading** per-position market/exchange + local-currency display already
shipped (goal-multimarket-ux: `positions-columns.tsx` MarketChip + currency-aware cells);
this step ADDS the multi-currency NAV breakdown + retains the market-open/closed indicator
(now in the gate bar per phase-54). The **backtest** page is single-market (US/USD) — its
"per-position market/exchange + currency" is the US/USD/SPY scope strip + the US
open/closed badge (the pipeline has no multi-market rows). No emoji; the checkbox group
uses native inputs + colored dots (not emoji); where an icon is used elsewhere it is via
`@/lib/icons` (Phosphor) per the rule.

## Plan steps (additive; DO-NO-HARM)

1. settings_api: `paper_markets` on `FullSettings`/`SettingsUpdate`/`_settings_to_full`/
   `_FIELD_TO_ENV` + PUT list→CSV. Pytest round-trip (list→CSV→validator) + default
   unchanged.
2. types.ts: `paper_markets?: string[]`.
3. `PaperMarketsField` (native fieldset/checkbox, ≥1 enforced) → `/paper-trading/manage`.
4. `MultiCurrencyNavBreakdown` (client-side currency grouping) → positions page.
5. `BacktestScopeStrip` (US/USD/SPY + market-hours badge) → backtest page header.
6. Verify: tsc 0; `npm run build` green; vitest 178; settings pytest; zero emoji;
   Playwright skip-auth visual of all three surfaces; restore the auth gate (302). Write
   `live_check_50.6.md` (build/API proofs + operator-to-confirm visual).
7. Fresh qa → log → flip → commit.

## Guardrails / DO-NO-HARM

- Backtest page: ADD a strip only; do NOT touch its USD-literal cells/baseline table.
- NAV widget: client-side only; no `/portfolio` shape change; graceful single/empty.
- `paper_markets` default `["US"]` unchanged; loop byte-identical unless the operator
  toggles. No `.env` hand-edit (settings_api owns the env write). No money-path change.
- No emoji; navy palette; JIT-safe literal class maps; native checkbox group (W3C) over
  bespoke ARIA. Reuse `format.ts` — no currency fork. Mount-guard the market-hours read.
- Visual verification mandatory (frontend.md rule 5): Playwright skip-auth + operator
  confirm.

## References

`handoff/current/research_brief.md`; `frontend/src/lib/format.ts`; `cockpit-helpers.tsx`;
`PortfolioAllocationDonut.tsx`; `frontend/src/app/{backtest,paper-trading/positions,
paper-trading/manage}/page.tsx`; `frontend/src/lib/types.ts`; `backend/api/settings_api.py`;
`docs/runbooks/browser-mcp.md`.
