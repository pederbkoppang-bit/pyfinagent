# Experiment Results -- Cycle 75: Google-Finance digit-flip via NumberFlow

**Date:** 2026-05-26
**Phase:** UX correction (cycle-74 shipped wrong pattern; this cycle replaces with the requested pattern). No SSOT or data-flow change. No masterplan flip.
**Result:** GENERATE complete; awaiting Q/A.

## What changed

Replaced the cycle-74 background-tint flash (Bloomberg pattern) with
per-digit slide animation via `@number-flow/react@0.6.0` (Google Finance
pattern that the operator pointed at on alphabet.googlefinance.com).
When 382.18 ticks to 382.45, NumberFlow animates only the changing "18"
digits sliding up to "45"; "382" stays still. The cycle-74 hook +
keyframes + globals.css override are FULLY removed -- no dead code.

### Files changed

**ADDED (1 dep):**
- `@number-flow/react@0.6.0` in `frontend/package.json` + `package-lock.json` (MIT, ~12-15kB gzipped per researcher).

**DELETED (1 file):**
- `frontend/src/lib/useFlashOnChange.ts` -- cycle-74 hook. NumberFlow owns its prev-value tracking, animation timing, and `prefers-reduced-motion` handling internally; the custom hook is redundant.

**MODIFIED (5 files):**

1. `frontend/tailwind.config.js` -- removed `theme.extend.keyframes.flash-up` + `flash-down` + matching `animation` entries.
2. `frontend/src/app/globals.css` -- removed the `@media (prefers-reduced-motion: reduce) { .animate-flash-* { animation: none !important; } }` block. NumberFlow's `respectMotionPreference: true` default supersedes this.
3. `frontend/src/components/paper-trading/cockpit-helpers.tsx` -- `Dollar` + `PnlBadge` refactored:
   - `Dollar` now renders `<NumberFlow value={v} format={{style:'currency', currency:'USD', minimumFractionDigits:2, maximumFractionDigits:2}} willChange aria-live="off" className="text-slate-100"/>`.
   - `PnlBadge` now renders `<NumberFlow value={v/100} format={{style:'percent', signDisplay:'always', minimumFractionDigits:2, maximumFractionDigits:2}} willChange aria-live="off" className={isPositive?'text-emerald-400':'text-rose-400'}/>`. Intl.NumberFormat `style:'percent'` expects raw decimal -- divide the prop by 100. NumberFlow appends "+" via `signDisplay:'always'`; manual cycle-74 prefix removed.
4. `frontend/src/components/paper-trading/positions-columns.tsx` -- `CurrentPriceCell` now renders NumberFlow with the same Dollar-style format. LiveBadge sibling preserved.
5. `frontend/src/app/page.tsx` -- `KpiTile` prop signature unified:
   - Was: `value: string` + `numericValue: number | null`.
   - Now: `value: number | null` + optional `fallback?: string` + optional `format?: Format`.
   - Uses NumberFlow's exported `Format` type (subset of `Intl.NumberFormatOptions` that excludes "scientific" / "engineering" notation; TS error caught + fixed during typecheck).
   - All 6 call sites updated: NAV (currency), P&L today (currency + signDisplay always), vs SPY (percent, divide alpha by 100), Sharpe (plain decimal), Max DD (percent, divide dd30 by 100), Positions (integer maximumFractionDigits=0). All 6 KpiTiles now animate when their value changes -- previously only the 3 live-priced tiles had cycle-74 flash; under NumberFlow every tile gets the digit-slide treatment uniformly.

**INHERITS automatically (no edit):**
- `frontend/src/components/paper-trading/trades-columns.tsx` (`<Dollar value={total_value}/>` at lines 9, 86) -- researcher caught this; the operator's original list missed it. Now flips to digit-slide via the Dollar refactor.

## Verification (verbatim command output)

### tsc --noEmit (frontend strict typecheck)
```
$ cd frontend && npx tsc --noEmit
exit=0
(no errors after Format type fix)
```

### npx vitest run
```
 Test Files  23 passed (23)
      Tests  178 passed (178)
   Start at  21:08:26
   Duration  3.76s
```

### python tests/verify_phase_23_1_17.py
```
ok useLiveNav shared hook + home page consumption + paper-trading refactor + repair script (mark_to_market + save_daily_snapshot)
```

### Dead-code shrapnel grep (cycle-74 leftovers)
```
$ grep -rn "useFlashOnChange\|flashClassName\|FLASH_CLASS\|animate-flash-" frontend/src/
no dead-code shrapnel
$ grep -n "flash" frontend/tailwind.config.js frontend/src/app/globals.css
no flash references in tailwind/globals
$ test -f frontend/src/lib/useFlashOnChange.ts
deleted as expected
```

### Dependency diff
```
$ git diff HEAD -- frontend/package.json
+    "@number-flow/react": "^0.6.0",
(exactly one new entry)

$ git diff --stat HEAD -- backend/
(empty -- ZERO backend changes)
```

### Memory rule: launchctl kickstart after npm install
```
$ launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.frontend"
exit=0 (launchd watchdog refreshed; stale dev server bundle invalidated)
```

## Artifact shape -- surfaces wired

Every live-priced numeric surface in the cockpit now uses NumberFlow:

**Paper Trading positions table (per row):**
- Current price (`CurrentPriceCell` -> NumberFlow).
- Market Value (via `<Dollar>` -> NumberFlow).
- P&L % (via `<PnlBadge>` -> NumberFlow).

**Paper Trading SummaryHero MetricCards (6 tiles):**
- NAV, Cash, Total P&L, vs SPY, Sharpe, Positions -- all via `<Dollar>` / `<PnlBadge>` / `<SharpeValue>` (Sharpe still uses sharpe-color text; not a digit-slide candidate by design).

**Paper Trading trades table (researcher catch):**
- Total Value (via `<Dollar>`).

**Home page KpiTiles (6 tiles):**
- NAV (currency), P&L today (currency + signDisplay), vs SPY (percent), Sharpe (decimal), Max DD (percent), Positions (integer). All 6 now uniformly animate via NumberFlow.

## NumberFlow integration notes (for future maintainers)

- The lib exports a `Format` type (subset of `Intl.NumberFormatOptions`) -- importing this avoids the TS2322 error that strict `Intl.NumberFormatOptions` triggers (scientific/engineering notation excluded by design).
- Percent style: pass raw decimal (1.42% -> 0.0142), NOT the percent number. Caller must divide by 100.
- `signDisplay: "always"` replaces manual "+" prefix from cycle 74.
- `willChange` prop recommended for the ~25 simultaneous instances per researcher Section 5 perf guidance.
- Reduced motion: `respectMotionPreference: true` default. Fallback = instant snap (no animation). No manual override needed.
- All consumers already had `"use client"` (lib requires it; researcher confirmed).
- Peer-dep install: `@tremor/react@^3.18.7` pins `react@^18.0.0` in peerDeps but the project runs React 19 in practice. Installed with `--legacy-peer-deps` to bypass the stale peer pin (Tremor v3 documented as React 19 compatible despite peerDep lag).

## Memory-rule compliance

- `npm install --legacy-peer-deps @number-flow/react@0.6.0` invoked once.
- `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` invoked immediately after install (per `feedback_npm_install_requires_launchctl_kickstart.md`).
- `npm run build` NOT invoked.
- `rm -rf .next/*` NOT invoked.
- No emojis introduced.

## Not in scope

- Visual verification of the digit slide in a browser (still pending operator review per `frontend.md` rule 5 -- "unit tests cannot see what the operator sees").
- RedLineMonitor live-now overlay (cycle 73 owns this; not a digit-display surface).
- Reality-gap chart / NAV chart annotation (chart axes, not numeric scorecards).
