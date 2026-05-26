# Contract -- Cycle 76: NumberFlow trend coloring + slowed slide

**Cycle:** 76 (2026-05-26)
**Class:** UX visibility hardening (operator: "didn't notice at all" after cycle 75 -- default 900ms silent slide is too subtle). No SSOT or data-flow change. No masterplan flip. ZERO new npm deps.

## Research gate

- Researcher `ae08ef2407507449a`, tier=moderate, 6 sources read in full, 14 snippet-only, 20 URLs, recency scan performed, internal_files_inspected=6, **gate_passed=true**.
- Brief: `handoff/current/research_brief_phase_numberflow_trend.md`.
- **Load-bearing finding:** `::part(up)` / `::part(down)` do NOT exist in `@number-flow/react@0.6.0`. Trend coloring requires a custom `data-pyfa-trend` host attribute set by React, then CSS targets `number-flow[data-pyfa-trend="up"]::part(digit)`.
- **`transformTiming` prop:** accepts `EffectTiming` (Web Animations API); default duration is 900ms; setting `{ duration: 700 }` keeps the lib's easing curve intact.
- **Reduced-motion:** NumberFlow's `respectMotionPreference: true` default still handles the slide; the new CSS animation also includes a `@media (prefers-reduced-motion: reduce)` guard to disable the color flash.

## N* delta

- **B primary:** the cycle-75 digit slide IS happening but is too subtle to notice (operator-reported on the Paper Trading screenshot showing $23 823,74 NAV unchanged for 30s observation). Adding a 700ms color flash on the changing digits (emerald-400 up / rose-400 down) makes the tick land in pre-attentive perception, matching Google Finance's actual UX.
- **R secondary:** ZERO behavioral or data-flow change. Pure presentation.

## Scope -- 1 new hook + 1 CSS block + 4 NumberFlow prop additions

### NEW

- `frontend/src/lib/use-trend.ts` -- tiny `useTrend(value)` hook returning `"up" | "down" | "flat"`. Tracks prev value via `useRef`, sets trend on change, auto-resets to "flat" after 700ms via `setTimeout` (matches CSS animation duration).

### MODIFIED

- `frontend/src/app/globals.css` -- add the `@keyframes pyfa-tint-up` + `pyfa-tint-down` + the `number-flow[data-pyfa-trend="..."]::part(digit|symbol)` selectors + the `@media (prefers-reduced-motion: reduce)` guard.
- `frontend/src/components/paper-trading/cockpit-helpers.tsx` -- in `Dollar` + `PnlBadge`: call `useTrend(value)`, pass `data-pyfa-trend={trend}` + `transformTiming={{ duration: 700 }}` to NumberFlow.
- `frontend/src/components/paper-trading/positions-columns.tsx` -- in `CurrentPriceCell`: same edits.
- `frontend/src/app/page.tsx` -- in `KpiTile`: same edits.

ZERO backend changes. ZERO new npm deps. ZERO test scaffolding renames.

## Immutable success criteria

1. `cd frontend && npx tsc --noEmit` exit 0.
2. `cd frontend && npx vitest run` -- 178+ passed.
3. `python tests/verify_phase_23_1_17.py` -- ok.
4. `git diff HEAD -- frontend/package.json` empty (ZERO new deps).
5. `git diff --stat HEAD -- backend/` empty.
6. `useTrend` hook exists at `frontend/src/lib/use-trend.ts`.
7. `data-pyfa-trend` attribute appears on all 4 NumberFlow consumers (grep returns 4+ hits).
8. `transformTiming={{ duration: 700 }}` on all 4 consumers.
9. `globals.css` contains `@keyframes pyfa-tint-up` AND `pyfa-tint-down` AND `number-flow[data-pyfa-trend="up"]::part(digit)` AND a `@media (prefers-reduced-motion: reduce)` guard for those selectors.
10. ZERO emojis introduced.
11. NO `npm run build`.
12. NO `rm -rf .next/*`.
13. NO `npm install` (zero new deps; no kickstart needed).

## /goal integration gates

1. tsc + vitest green. 2. ZERO `npm run build`. 3. Zero emojis. 4. Log LAST / no masterplan flip. 5. Reduced-motion regression checked.
