# Experiment Results -- Cycle 74: price-tick flash animation

**Date:** 2026-05-26
**Phase:** UX polish (no SSOT or data-flow change; no masterplan flip).
**Result:** GENERATE complete; awaiting Q/A.

## What changed

Implemented Google-Finance-style flash-on-change animation across every
numeric cockpit display that updates with live stock prices. When a
value ticks up, the cell briefly tints `bg-emerald-500/15`; on a
down-tick, `bg-rose-500/15`. 500ms ease-in-out, then fades to
transparent. Honors `prefers-reduced-motion: reduce` (hook short-circuits
in JS AND `globals.css` overrides `animation: none !important` -- defense
in depth per researcher Section 3).

### Files (1 new + 5 modified, ZERO backend)

1. `frontend/src/lib/useFlashOnChange.ts` -- **NEW**
   - `useFlashOnChange(value, { decimals=2, durationMs=500 })` returns `"up" | "down" | null`.
   - Tracks previous value via `useRef<string>` (formatted via `value.toFixed(decimals)` so 100.001 vs 100.002 rounding noise does not strobe).
   - Skips first render (no flash on initial `null -> $124.50` population).
   - `requestAnimationFrame` between `setDirection(null)` and `setDirection(next)` forces the CSS animation to restart on consecutive same-direction ticks (browsers do not re-run a CSS animation when className is unchanged across renders).
   - `setTimeout` clears the direction after `durationMs`; cleared on subsequent tick AND on unmount (researcher Section 2 cleanup requirement).
   - `prefers-reduced-motion: reduce` short-circuit: hook returns `null` so no `animate-flash-*` class lands.
   - JIT-safe `FLASH_CLASS` static literal map (cycle-68 lesson): `{ up: "animate-flash-up", down: "animate-flash-down" }` -- both literals appear verbatim so Tailwind JIT compiles them.
   - Public helper `flashClassName(direction)` for consumer ergonomics.

2. `frontend/tailwind.config.js` -- MODIFIED
   - Added `theme.extend.keyframes`:
     - `flash-up`: `0% { backgroundColor: "rgba(16, 185, 129, 0.15)" } -> 100% { backgroundColor: "transparent" }` (emerald-500/15).
     - `flash-down`: same with `rgba(244, 63, 94, 0.15)` (rose-500/15).
   - Added `theme.extend.animation.flash-up` / `flash-down` = `flash-up 500ms ease-in-out`.
   - Tailwind v3 docs (https://v3.tailwindcss.com/docs/animation) confirm this is the canonical extend-keyframes pattern; researcher Section 5 verified.

3. `frontend/src/app/globals.css` -- MODIFIED
   - Added `@media (prefers-reduced-motion: reduce) { .animate-flash-up, .animate-flash-down { animation: none !important; } }`.
   - Reason: defense in depth -- if an operator toggles the OS preference mid-session, any in-flight `animate-flash-*` class stops immediately at the CSS layer; the JS-layer short-circuit in `useFlashOnChange.ts` only prevents NEW animations.

4. `frontend/src/components/paper-trading/cockpit-helpers.tsx` -- MODIFIED
   - `Dollar` + `PnlBadge` now consume `useFlashOnChange(value)` and apply the returned animation class via `flashClassName(flash)`. Both primitives are shared by the positions table cells (Market Value, P&L) AND the SummaryHero MetricCards (NAV, Cash, Total P&L, vs SPY) -- one change covers every consumer on the Paper Trading page.
   - Added `aria-live="off"` on both spans (MDN stock-ticker default; do NOT announce every tick).

5. `frontend/src/components/paper-trading/positions-columns.tsx` -- MODIFIED
   - Current price cell pulled into a new `CurrentPriceCell` component so `useFlashOnChange` fires per row (React rules-of-hooks: cannot call hooks inside a TanStack column cell render callback).
   - Applies `animate-flash-*` class to the `${shown.toFixed(2)}` span.
   - `aria-live="off"` on the wrapping span.

6. `frontend/src/app/page.tsx` -- MODIFIED
   - Added `numericValue?: number | null` prop to `KpiTile` so the hook can compare ticks without parsing the pre-formatted display string back.
   - Hook fires per tile; class applied to the value `<p>` element.
   - Wired `numericValue` on 3 live-priced KpiTiles: NAV (`navValue`), P&L today (`today?.dollars`), vs SPY (`alpha`). Sharpe / Max DD / Positions tiles deliberately not wired (they update on snapshot persistence, not on price ticks -- researcher scope table).

### Files unchanged (audit)

- ZERO backend file changes (`git diff --stat HEAD -- backend/` returns empty).
- ZERO new npm deps (`frontend/package.json` unchanged this cycle).
- ZERO test scaffolding changes.
- `frontend/src/components/RedLineMonitor.tsx` -- intentionally NOT wired (cycle-73 live-now overlay is its own visual signal; flashing the line stroke would compete with the pulsating endpoint dot).
- `frontend/src/components/PaperTradesTable.tsx` -- not wired (researcher Section 6 internal-grep audit found this surface shows realized trades, NOT live mark-to-market; no live-priced numbers).

## Verification (verbatim command output)

### tsc --noEmit (frontend strict typecheck)
```
$ cd frontend && npx tsc --noEmit
exit=0
(no output, no errors)
```

### npx vitest run
```
 Test Files  23 passed (23)
      Tests  178 passed (178)
   Start at  20:35:04
   Duration  4.24s
```

### python tests/verify_phase_23_1_17.py
```
ok useLiveNav shared hook + home page consumption + paper-trading refactor + repair script (mark_to_market + save_daily_snapshot)
```

### Grep audit (zero non-ASCII in new code)
```
$ grep -Pn "[^\x00-\x7F]" frontend/src/lib/useFlashOnChange.ts
(no output -- new file is pure ASCII)
```

The cycle's diff to `positions-columns.tsx` includes one em-dash
(U+2014) inside `<span className="text-slate-500">—</span>` for the
empty-state placeholder. This matches the pre-existing project
convention -- four other em-dashes in the same file pre-date cycle 74
(lines 78, 89, 184, and the original Current cell empty state at line
86 pre-modification). Em-dash is a text-content placeholder, NOT an
emoji or unicode arrow; the "no emojis" memory rule targets graphic
Unicode (e.g. green/red circles, arrows) which are forbidden. Contract
criterion 4 (grep clean on `useFlashOnChange.ts` specifically) passes.

### Zero new deps / zero backend
```
$ git diff HEAD -- frontend/package.json
(empty -- no dep changes)

$ git diff --stat HEAD -- backend/
(empty -- no backend changes)
```

## Artifact shape

After cycle 74, every live-priced numeric display flashes on tick:

**Paper Trading positions table (per row):**
- Current price -- flashes when `livePrices[ticker].price` updates.
- Market Value -- flashes when `livePrice * quantity` updates (inherited via `<Dollar>`).
- P&L % -- flashes when the live-derived P&L percentage updates (inherited via `<PnlBadge>`).

**Paper Trading SummaryHero MetricCards (6 tiles):**
- NAV -- flashes via `<Dollar>` from `lp.liveNav`.
- Cash -- flashes via `<Dollar>` (snapshot-driven; less frequent).
- Total P&L -- flashes via `<PnlBadge>` from `lp.liveTotalPnlPct`.
- vs SPY -- flashes via `<PnlBadge>`.
- Sharpe -- not wired (snapshot metric, not live-priced).
- Positions count -- not wired (integer count, not live-priced).

**Home page KpiTiles (3 live-priced of 6 total):**
- NAV -- flashes from `navValue` numericValue prop.
- P&L (today) -- flashes from `today?.dollars`.
- vs SPY -- flashes from `alpha`.

## JIT-safety verification

`FLASH_CLASS` at `useFlashOnChange.ts:124-127` is a static literal map:
```ts
export const FLASH_CLASS: Record<"up" | "down", string> = {
  up: "animate-flash-up",
  down: "animate-flash-down",
};
```
Both `animate-flash-up` and `animate-flash-down` appear as exact literal
strings in the bundle. Tailwind JIT compiles them via the
`theme.extend.animation` declarations in `tailwind.config.js`. No
template-string concatenation; cycle-68 lesson honored.

## Reduced-motion verification (defense in depth)

1. JS layer: `useFlashOnChange.ts:78-83` checks `window.matchMedia("(prefers-reduced-motion: reduce)").matches` -- when true, hook returns `null` so NO `animate-flash-*` className lands.
2. CSS layer: `globals.css:108-114` `@media (prefers-reduced-motion: reduce) { .animate-flash-up, .animate-flash-down { animation: none !important; } }` -- if a className did land (e.g. operator toggled preference mid-flash), the animation halts at the CSS layer.

## A11y compliance

- WCAG SC 2.3.3 Animation from Interactions: **does NOT apply** (passive price ticks are not user-initiated interaction -- W3C spec scopes to interactions; researcher Section 3).
- WCAG SC 2.2.2 Pause/Stop/Hide: **applies**, satisfied (500ms flash is well under the 5-second ceiling).
- ARIA: every flashing span carries `aria-live="off"` (MDN stock-ticker default; do NOT announce every tick or screen readers flood; researcher Section 4 cited).

## Memory-rule compliance

- `npm run build` NOT invoked.
- `rm -rf .next/*` NOT invoked.
- No `launchctl kickstart` needed (no npm install ran).
- No emojis introduced.

## Not in scope

- RedLineMonitor live-now overlay (cycle-73 already provides its own pulsating endpoint signal).
- Reality-gap chart (chart annotation, not a numeric scorecard).
- Reports / Backtest / Manage forms (not live-priced).
- Visual verification of the flash in a browser (still pending operator review per frontend.md rule 5).
