# Contract -- Cycle 74: Google-Finance price-tick flash animation

**Cycle:** 74 (2026-05-26)
**Class:** UX polish (no SSOT or data-flow change)
**Masterplan flip:** NONE (UX-only, follows cycle-73 chart-side SSOT)

## Research gate

- Researcher `a3f10c3c35c087f50`, tier=moderate, 11 sources read in full, 28 URLs, recency scan performed, internal_files_inspected=8, **gate_passed=true**.
- Brief: `handoff/current/research_brief_phase_flash_animation.md`.
- Canonical reference: lab49/react-value-flash (production financial-app flash lib): 200ms hold + 100ms fade = ~500ms total, ease-in-out, `#00d865` up / `#d43215` down. We adopt **500ms total** with `bg-emerald-500/15` up / `bg-rose-500/15` down (low-opacity tint preserves slate-text contrast).
- A11y: WCAG SC 2.3.3 does NOT apply (passive price ticks are not user-initiated interaction); SC 2.2.2 governs and a 500ms flash is compliant (well under the 5s ceiling). `prefers-reduced-motion: reduce` honored via global `@media` override that sets `animation: none`. ARIA: `aria-live="off"` (MDN's explicit stock-ticker default; do NOT announce every tick).

## N* delta

- **B primary:** the cockpit currently shows live-priced numbers as static text -- when a price ticks, the value silently changes from $124.50 to $124.65 with no visual signal. Operators reading the cockpit pre-attentively miss every tick. Flash-on-change makes ticks SEEN, matching the Google Finance / Bloomberg / Robinhood Legend convention.
- **R secondary:** zero behavioral change, zero data-flow change, zero risk-guard change. The numbers themselves are unchanged; only their visual presentation gains a 500ms tint on update.

## Scope -- 5 files, NEW hook + 4 modifications

| File | Action | Detail |
|---|---|---|
| `frontend/src/lib/useFlashOnChange.ts` | **NEW** | `useFlashOnChange(value, { decimals=2, durationMs=500 })` returns `"up" \| "down" \| null`. Tracks prev via `useRef`. Compares `value.toFixed(decimals)` so 100.001 vs 100.002 don't strobe on rounding noise. `setTimeout` cleanup on unmount. |
| `frontend/tailwind.config.js` | MOD | Add `theme.extend.keyframes.flashUp` + `flashDown` (`bg-emerald-500/15` -> transparent over 500ms ease-in-out; rose for down) + `theme.extend.animation.flashUp` + `flashDown`. |
| `frontend/src/app/globals.css` | MOD | Add `@media (prefers-reduced-motion: reduce) { .animate-flash-up, .animate-flash-down { animation: none !important; } }`. |
| `frontend/src/components/paper-trading/positions-columns.tsx` | MOD | Wrap 3 cells with FlashCell: Current price (line ~86 `shown`), Market Value (line ~107 via `Dollar`), P&L (line ~138 via `PnlBadge`). Each cell reads its own value through the hook + applies returned class. |
| `frontend/src/app/page.tsx` + `frontend/src/components/paper-trading/cockpit-helpers.tsx` | MOD | Wire NAV + P&L Today + vs SPY KPI tiles. SummaryHero MetricCards share the same `lp.liveNav` source so both pages flash in sync. |

**Out of scope per researcher:** RedLineMonitor's ReferenceLine (cycle-73 live-now overlay is its own visual signal), paper-trading/reality-gap chart prop, paper-trading/nav historical row (one-shot snapshot, never ticks), ReportHeader, StockChart, RiskDashboard, paper-trading/manage form inputs.

## Files

NEW:
- `frontend/src/lib/useFlashOnChange.ts`

MODIFIED:
- `frontend/tailwind.config.js`
- `frontend/src/app/globals.css`
- `frontend/src/components/paper-trading/positions-columns.tsx`
- `frontend/src/app/page.tsx`
- `frontend/src/components/paper-trading/cockpit-helpers.tsx`

ZERO backend changes. ZERO new npm deps. ZERO test scaffolding renames.

## Immutable success criteria

1. `cd frontend && npx tsc --noEmit` exit 0.
2. `cd frontend && npx vitest run` -- 178+ passed (current baseline).
3. `python tests/verify_phase_23_1_17.py` -- ok (SSOT invariant intact).
4. Grep audit: `grep -rn "[^\x00-\x7F]" frontend/src/lib/useFlashOnChange.ts frontend/src/components/paper-trading/positions-columns.tsx` returns clean (zero non-ASCII / zero emojis introduced).
5. `frontend/package.json` diff against HEAD shows ZERO new dependencies.
6. ZERO files modified under `backend/`.
7. `LIVE_MARKER_COLOR`-style static literal class map in the hook -- no template-string concatenation (cycle-68 JIT-safety lesson).
8. `prefers-reduced-motion: reduce` override present in `globals.css` AND the hook itself short-circuits if `window.matchMedia("(prefers-reduced-motion: reduce)").matches` returns true (defense in depth).
9. Hook returns `null` on FIRST render (no flash on initial value populating from undefined/null).
10. Hook's `setTimeout` ID is cleared on unmount AND on subsequent value changes within the duration window (no leak; no double-flash).
11. ARIA: every FlashCell wrapper that emits a flash sets `aria-live="off"` per MDN stock-ticker default. Confirmed cited source: MDN aria-live docs.
12. NO `npm run build` invoked (memory rule).
13. NO `rm -rf .next/*` invoked (memory rule).
14. NO emojis in any code or comment introduced this cycle.

## /goal integration gates

1. tsc + vitest green. 2. No `npm run build`. 3. Zero emojis. 4. Log LAST / no masterplan flip.
