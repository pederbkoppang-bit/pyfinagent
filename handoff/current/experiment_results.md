# Experiment Results -- Cycle 77: fix CSS element-name bug + bump to 900ms

**Date:** 2026-05-26
**Phase:** UX bugfix + timing tune (operator: "i dont see the color tint when digits are moving up and down" + "runs a bit fast"). No SSOT or data-flow change. No masterplan flip. ZERO new npm deps.
**Result:** GENERATE complete; awaiting Q/A.

## Root cause analysis (the load-bearing find)

Cycle 76 shipped the CSS selector `number-flow[data-pyfa-trend="up"]::part(digit)`. Operator reported no tint. Inspection of `@number-flow/react@0.6.0`'s React wrapper source (`dist/NumberFlow-client-BTpPLmzo.mjs` line ~140) shows the wrapper renders `<number-flow-react>` (NOT `<number-flow>`) -- the lib appends `-react` via its internal `elementSuffix: '-react'` config when called from the React entry point:

```js
React.createElement("number-flow-react", { ...rest, ... });
```

So the cycle-76 selector never matched the actual DOM element. `...rest` IS spreading `data-pyfa-trend` to the host element correctly -- the prop pass-through works fine; the bug is purely in the CSS element name.

This cycle fixes the selector and bumps all durations to 900ms per researcher `a750bbbd767273170` (NumberFlow's own default; M3 `extra-long3` token; matches NN/g + Smashing 2025 Doherty band).

## What changed (5 files modified)

1. `frontend/src/app/globals.css`:
   - `number-flow[data-pyfa-trend="up"]::part(digit|symbol)` -> `number-flow-react[data-pyfa-trend="up"]::part(digit|symbol)`.
   - Mirror for "down".
   - Reduced-motion `@media` selector also updated to `number-flow-react`.
   - Keyframe animation duration 700ms -> 900ms.
2. `frontend/src/lib/use-trend.ts` -- default `durationMs` 700 -> 900.
3. `frontend/src/components/paper-trading/cockpit-helpers.tsx` -- two `transformTiming` literals 700 -> 900 (Dollar + PnlBadge).
4. `frontend/src/components/paper-trading/positions-columns.tsx` -- one `transformTiming` 700 -> 900 (CurrentPriceCell).
5. `frontend/src/app/page.tsx` -- one `transformTiming` 700 -> 900 (KpiTile).

## Verification (verbatim command output)

### tsc --noEmit
```
$ cd frontend && npx tsc --noEmit
tsc=0
```

### npx vitest run
```
 Test Files  23 passed (23)
      Tests  178 passed (178)
   Start at  21:46:23
   Duration  4.45s
```

### python tests/verify_phase_23_1_17.py
```
ok useLiveNav shared hook + home page consumption + paper-trading refactor + repair script (mark_to_market + save_daily_snapshot)
```

### Audit greps
```
$ grep -rn "transformTiming.*700\b" frontend/src/
none

$ grep -rn "transformTiming.*900\b" frontend/src/
page.tsx:177            transformTiming={{ duration: 900 }}
positions-columns.tsx:54          transformTiming={{ duration: 900 }}
cockpit-helpers.tsx:44      transformTiming={{ duration: 900 }}
cockpit-helpers.tsx:65      transformTiming={{ duration: 900 }}
(4 hits, one per NumberFlow consumer)

$ grep -c "number-flow\[data-pyfa-trend" frontend/src/app/globals.css
0  (stale selector gone)

$ grep -c "number-flow-react\[data-pyfa-trend" frontend/src/app/globals.css
4  (up::part(digit) + up::part(symbol) + down::part(digit) + down::part(symbol))

$ grep -E "pyfa-tint-(up|down) [0-9]+ms" frontend/src/app/globals.css
animation: pyfa-tint-up 900ms ease-out;
animation: pyfa-tint-down 900ms ease-out;
```

### Zero deps + zero backend
```
$ git diff HEAD -- frontend/package.json
(empty)

$ git diff --stat HEAD -- backend/
(empty)
```

## Market Value note (operator's second clarification)

Operator said they "forgot to put a red circle on market value in the positions table". Market Value already animates via `<Dollar>` (renders NumberFlow per cycle 75 + 76). This was operator-annotation cleanup, not a missing wiring. With the cycle-77 selector fix, Market Value will tint along with Current and P&L.

## A11y compliance (unchanged from cycle 76)

- WCAG SC 2.3.3 N/A (passive ticks).
- SC 2.2.2 satisfied (900ms << 5s ceiling).
- `aria-live="off"` preserved.
- Reduced-motion: NumberFlow's `respectMotionPreference: true` halts the slide; the corrected `@media (prefers-reduced-motion: reduce)` block now correctly halts the tint via `number-flow-react::part(*)`.

## Memory-rule compliance

- NO `npm install` (zero new deps).
- NO `npm run build`.
- NO `rm -rf .next/*`.
- ZERO emojis introduced.

## Not in scope

- Browser visual verification still pending operator hard-refresh.
- Polling interval (60s) unchanged.
