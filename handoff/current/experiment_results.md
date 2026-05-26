# Experiment Results -- Cycle 76: NumberFlow trend coloring + slowed slide

**Date:** 2026-05-26
**Phase:** UX visibility hardening (cycle 75 shipped the right pattern but at 900ms silent slide; operator: "didn't notice at all"). No SSOT or data-flow change. No masterplan flip. ZERO new npm deps.
**Result:** GENERATE complete; awaiting Q/A.

## What changed

Added a 700ms emerald (up) / rose (down) color flash on the digits that
slide, matching Google Finance's pre-attentive tick signal. The slide
itself slows from 900ms (NumberFlow's default) to 700ms so it lines up
with the tint. Reduced-motion preserved.

Implementation pattern: NumberFlow's `<number-flow>` custom element does
NOT expose `::part(up)` / `::part(down)` selectors (researcher
`ae08ef2407507449a` verified against the lib's `lite.ts` source). We
emit our own `data-pyfa-trend="up" | "down" | "flat"` host attribute via
a new `useTrend` hook and target it in `globals.css` via
`number-flow[data-pyfa-trend="up"]::part(digit)`.

### Files (1 new + 4 modified, ZERO backend, ZERO new deps)

1. `frontend/src/lib/use-trend.ts` -- **NEW** hook. `useTrend(value, durationMs=700)` returns `"up" | "down" | "flat"`. Tracks prev value via `useRef`, sets trend on change, auto-resets to "flat" after 700ms via `setTimeout` (matches CSS animation). Cleared on subsequent change AND on unmount.

2. `frontend/src/app/globals.css` -- added:
   - `@keyframes pyfa-tint-up` (0% color #34d399 emerald-400 -> 100% inherit).
   - `@keyframes pyfa-tint-down` (0% color #fb7185 rose-400 -> 100% inherit).
   - Selectors `number-flow[data-pyfa-trend="up"]::part(digit)` and `::part(symbol)` -> `animation: pyfa-tint-up 700ms ease-out`.
   - Mirror selectors for "down" -> `pyfa-tint-down`.
   - `@media (prefers-reduced-motion: reduce) { number-flow::part(digit), number-flow::part(symbol) { animation: none !important; } }` -- disables both tint AND NumberFlow's slide for reduced-motion operators.

3. `frontend/src/components/paper-trading/cockpit-helpers.tsx` -- in `Dollar` + `PnlBadge`: call `useTrend(value)`, add `transformTiming={{ duration: 700 }}` + `data-pyfa-trend={trend}` to NumberFlow.

4. `frontend/src/components/paper-trading/positions-columns.tsx` -- in `CurrentPriceCell`: same edits.

5. `frontend/src/app/page.tsx` -- in `KpiTile`: same edits (one NumberFlow site, fires for all 6 tiles).

## Verification (verbatim command output)

### tsc --noEmit
```
$ cd frontend && npx tsc --noEmit
exit=0
(no output)
```

### npx vitest run
```
 Test Files  23 passed (23)
      Tests  178 passed (178)
   Start at  21:31:27
   Duration  4.05s
```

### python tests/verify_phase_23_1_17.py
```
ok useLiveNav shared hook + home page consumption + paper-trading refactor + repair script (mark_to_market + save_daily_snapshot)
```

### `data-pyfa-trend` attribute presence (4 NumberFlow consumers)
```
$ grep -n "data-pyfa-trend=" frontend/src/components/paper-trading/cockpit-helpers.tsx frontend/src/components/paper-trading/positions-columns.tsx frontend/src/app/page.tsx
cockpit-helpers.tsx:47  data-pyfa-trend={trend}   (PnlBadge)
cockpit-helpers.tsx:68  data-pyfa-trend={trend}   (Dollar)
positions-columns.tsx:56  data-pyfa-trend={trend}  (CurrentPriceCell)
page.tsx:179  data-pyfa-trend={trend}              (KpiTile)
```
4 prop sites confirmed.

### globals.css trend CSS
```
$ grep -E "pyfa-tint|data-pyfa-trend|prefers-reduced-motion" frontend/src/app/globals.css
@keyframes pyfa-tint-up { ... }
@keyframes pyfa-tint-down { ... }
number-flow[data-pyfa-trend="up"]::part(digit),
number-flow[data-pyfa-trend="up"]::part(symbol) { animation: pyfa-tint-up 700ms ease-out; }
number-flow[data-pyfa-trend="down"]::part(digit),
number-flow[data-pyfa-trend="down"]::part(symbol) { animation: pyfa-tint-down 700ms ease-out; }
@media (prefers-reduced-motion: reduce) { ... animation: none !important; }
```

### Zero new deps + zero backend
```
$ git diff HEAD -- frontend/package.json
(empty)

$ git diff --stat HEAD -- backend/
(empty)
```

## A11y compliance

- WCAG SC 2.3.3 Animation from Interactions: N/A (passive ticks; researcher cycle 74).
- WCAG SC 2.2.2 Pause/Stop/Hide: satisfied (700ms tint << 5s ceiling; slide 700ms).
- `aria-live="off"` preserved on every NumberFlow consumer (MDN stock-ticker default).
- Reduced-motion: `respectMotionPreference: true` (NumberFlow default) halts the slide; the new `@media (prefers-reduced-motion: reduce)` block also halts the tint via `animation: none !important`.

## Artifact shape

After cycle 76, every cycle-75 NumberFlow consumer also color-tints on
tick:

- **Up-tick** ($124.50 -> $124.65): changing digits ("50" -> "65") slide
  in their cells AND briefly turn emerald-400 (#34d399) before fading
  back to the parent text color over 700ms.
- **Down-tick** ($124.65 -> $124.50): same slide + brief rose-400
  (#fb7185) tint.
- **Flat** (no change): no slide, no tint.

Surfaces wired (4 NumberFlow consumers, ~25 simultaneous instances):
- Paper Trading positions table (Current, Market Value, P&L)
- Paper Trading SummaryHero MetricCards (NAV, Cash, Total P&L, vs SPY)
- Paper Trading trades table (Total Value, inherited via Dollar)
- Home KpiTiles (NAV, P&L today, vs SPY, Sharpe, Max DD, Positions)

## Memory-rule compliance

- NO `npm install` (zero new deps; no `launchctl kickstart` needed).
- NO `npm run build`.
- NO `rm -rf .next/*`.
- ZERO emojis introduced.

## Not in scope

- Reducing live-prices polling interval below 60s (would burn API quota; not the right fix for visibility).
- Browser visual verification of the tint + slide combination (still pending operator review per `frontend.md` rule 5 -- the operator's last screenshot proved NumberFlow IS rendering via the Norwegian locale formatting `$23 823,74`; this cycle adds the visibility cue they asked for).
