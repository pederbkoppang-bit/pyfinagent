# Contract -- Cycle 75: Google-Finance digit-flip animation (replace cycle-74 background flash)

**Cycle:** 75 (2026-05-26)
**Class:** UX correction (cycle-74 shipped the wrong pattern -- Bloomberg background tint instead of Google's per-digit slide). No SSOT or data-flow change. No masterplan flip.

## Research gate

- Researcher `ad12953b2b579e884`, tier=moderate, 6 sources read in full, 10 snippet-only, 16 URLs, recency scan performed, internal_files_inspected=8, **gate_passed=true**.
- Brief: `handoff/current/research_brief_phase_number_flow.md`.
- Canonical reference: `@number-flow/react@0.6.0` by Maxwell Barvian (`barvian`), MIT, ~12-15kB gzipped, React 19 + Next.js 15 compatible (all upstream blockers closed: issues #22, #95, #107). Reduced-motion supported by default via `respectMotionPreference: true` prop (drops the need for the cycle-74 globals.css override + JS short-circuit).
- **Operator-list correction:** researcher's internal grep flagged `frontend/src/components/paper-trading/trades-columns.tsx:9,86` which uses `<Dollar value={total_value}/>` -- this consumer inherits NumberFlow automatically when we refactor the Dollar primitive. Not in operator's original list but in scope.

## N* delta

- **B primary:** the cycle-74 background-tint flash matches the Bloomberg / Robinhood pattern -- NOT what the operator wanted. They specifically pointed at Google Finance, which uses a per-digit slide (382.18 -> 382.45: only "18" digits slide, "382" stays still). This cycle replaces the wrong pattern with the right one.
- **R secondary:** ZERO behavioral change, ZERO data flow change. The numbers themselves are unchanged; only their visual presentation gains per-digit slide on update.

## Scope -- delete, remove blocks, replace bodies, install dep

### DELETE entire file (cycle-74 hook no longer needed -- NumberFlow owns its state)

- `frontend/src/lib/useFlashOnChange.ts`

### REMOVE blocks (cycle-74 keyframes / overrides redundant)

- `frontend/tailwind.config.js` -- the `flash-up` / `flash-down` keyframes + animation entries.
- `frontend/src/app/globals.css` -- the `@media (prefers-reduced-motion: reduce)` block for `.animate-flash-*`. NumberFlow's `respectMotionPreference: true` default supersedes this.

### REPLACE bodies (swap inline-flash code for NumberFlow)

- `frontend/src/components/paper-trading/cockpit-helpers.tsx`:
  - `Dollar` (lines 20-37 in cycle-74): becomes `<NumberFlow value={v} format={{style:'currency', currency:'USD', minimumFractionDigits:2, maximumFractionDigits:2}} willChange className="text-slate-100"/>`. Inherits at every Dollar consumer including `trades-columns.tsx:9,86` (researcher catch).
  - `PnlBadge` (lines 39-55 in cycle-74): becomes `<NumberFlow value={v/100} format={{style:'percent', signDisplay:'always', minimumFractionDigits:2, maximumFractionDigits:2}} willChange className={isPositive?'text-emerald-400':'text-rose-400'}/>`. **Important:** Intl.NumberFormat `style:'percent'` expects raw decimal -- divide the prop by 100 (1.42% -> pass 0.0142). Cycle-74 manually appended "+"; NumberFlow does it via `signDisplay:'always'`.

- `frontend/src/components/paper-trading/positions-columns.tsx`:
  - `CurrentPriceCell` (cycle-74 component): replace the inline-flash span with NumberFlow with the same Dollar format. LiveBadge sibling stays as-is.

- `frontend/src/app/page.tsx`:
  - `KpiTile`: drop the `value: string` + `numericValue: number | null` two-prop pattern; consolidate to `value: number | null` + optional `format?: Intl.NumberFormatOptions`. Drop `useFlashOnChange` import. Render the value via NumberFlow when `format` provided, fall back to text node when not (e.g. Positions count which is integer-only). Update 3 call sites: NAV (format=currency), P&L today (format=currency with signDisplay='always'), vs SPY (format=percent / 100 divisor).

### ADD dep

- `npm install @number-flow/react@0.6.0` (frontend cwd; the ONE new dep this cycle).
- **MANDATORY post-install:** `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` per memory rule `feedback_npm_install_requires_launchctl_kickstart.md`. pkill races the launchd watchdog; kickstart is the reliable path.

## Files

NEW deps:
- `@number-flow/react@0.6.0` added to `frontend/package.json` + `package-lock.json`.

DELETED:
- `frontend/src/lib/useFlashOnChange.ts` (cycle-74 hook, no longer needed).

MODIFIED:
- `frontend/tailwind.config.js` (remove flash keyframes block).
- `frontend/src/app/globals.css` (remove reduced-motion override for flash classes).
- `frontend/src/components/paper-trading/cockpit-helpers.tsx` (Dollar + PnlBadge refactor).
- `frontend/src/components/paper-trading/positions-columns.tsx` (CurrentPriceCell refactor).
- `frontend/src/app/page.tsx` (KpiTile prop signature + 3 wired sites).

INHERITS (no edit; auto-flips because Dollar + PnlBadge refactor):
- `frontend/src/components/paper-trading/trades-columns.tsx` (`Dollar` consumer at lines 9, 86).
- Every `<Dollar value={...}/>` site project-wide (researcher catches any other).

ZERO backend changes. ZERO test scaffolding renames.

## Immutable success criteria

1. `cd frontend && npx tsc --noEmit` exit 0.
2. `cd frontend && npx vitest run` -- 178+ passed (cycle-74 baseline; cascade tests that touched `useFlashOnChange` need updating IF any exist -- grep first; if not, suite stays at 178).
3. `python tests/verify_phase_23_1_17.py` -- ok (SSOT invariant intact).
4. `frontend/package.json` diff against HEAD shows EXACTLY ONE new dep: `@number-flow/react`. Lock file updates allowed.
5. `git diff --stat HEAD -- backend/` returns empty (ZERO backend changes).
6. `frontend/src/lib/useFlashOnChange.ts` no longer exists (`test -f .../useFlashOnChange.ts` returns 1).
7. `frontend/tailwind.config.js` no longer contains `flash-up` or `flash-down` strings (grep returns empty).
8. `frontend/src/app/globals.css` no longer contains `.animate-flash-` substring (grep returns empty).
9. No code references `useFlashOnChange`, `flashClassName`, `FLASH_CLASS`, or `animate-flash-` anywhere in `frontend/src/` (grep returns empty -- no dead-code shrapnel from cycle 74).
10. NumberFlow rendered in: Dollar, PnlBadge, CurrentPriceCell, KpiTile (greppable via `<NumberFlow `).
11. Reduced-motion handled by NumberFlow's built-in `respectMotionPreference` prop (default true -- no manual override needed).
12. ZERO emojis introduced.
13. NO `npm run build` invoked.
14. NO `rm -rf .next/*` invoked.
15. `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` invoked AFTER `npm install` per memory rule.

## /goal integration gates

1. tsc + vitest green. 2. ZERO `npm run build`. 3. Zero emojis. 4. Log LAST / no masterplan flip. 5. ONE new dep justified by researcher + operator approval (the package is canonical for the requested pattern; researcher confirms no superior alternative).
