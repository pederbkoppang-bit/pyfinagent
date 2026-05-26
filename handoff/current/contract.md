# Contract -- Cycle 77: fix cycle-76 CSS element-name bug + bump durations to 900ms

**Cycle:** 77 (2026-05-26)
**Class:** UX bugfix + timing tune (operator: "i dont see the color tint when digits are moving up and down" + "runs a bit fast"). No SSOT or data-flow change. No masterplan flip. ZERO new npm deps.

## Research gate

- Researcher `a750bbbd767273170`, tier=moderate, 7 sources read in full, 12 snippet-only, 19 URLs, recency scan performed, internal_files_inspected=5, **gate_passed=true**.
- Brief: `handoff/current/research_brief_phase_tick_duration.md`.
- Recommendation: slide 900ms + tint 900ms EQUAL + `useTrend` `durationMs` 900ms (NumberFlow's own default; M3 `extra-long3` token; matches NN/g + Smashing 2025 + Doherty band).

## Root-cause for "no tint visible"

Inspection of `@number-flow/react@0.6.0` source (`dist/NumberFlow-client-BTpPLmzo.mjs`) reveals the React wrapper renders `<number-flow-react>` (NOT `<number-flow>`) via the lib's internal `elementSuffix: '-react'` config. Cycle 76's CSS used `number-flow[data-pyfa-trend="up"]::part(digit)` -- this selector never matched the actual DOM element, so no tint applied. `...rest` IS spreading `data-pyfa-trend` to the host element correctly (the prop forwarding works fine); the bug is purely in the CSS element name.

## N* delta

- **B primary:** the cycle-76 tint code is intact and correct except for one wrong identifier. Fixing the selector exposes the tint immediately. Bumping durations to 900ms makes the animation register cleanly without feeling sluggish.
- **R secondary:** ZERO behavioral or data-flow change. Two-file edit (one CSS selector swap + 7 literal `700`s -> `900`).

## Market Value clarification (operator's second note)

Operator said they "forgot to put a red circle on market value in the position table". Market Value already animates via `<Dollar>` (which renders NumberFlow per cycle 75 + 76). This is operator-annotation cleanup, NOT a missing wiring. No code change needed for this point -- it'll start animating the moment the CSS bug is fixed.

## Scope -- 2 files modified

### MODIFIED

- `frontend/src/app/globals.css`:
  - Selector `number-flow[data-pyfa-trend="up"]::part(digit)` -> `number-flow-react[data-pyfa-trend="up"]::part(digit)` (and the `down` mirror).
  - `pyfa-tint-up` + `pyfa-tint-down` animation duration 700ms -> 900ms.
  - `@media (prefers-reduced-motion: reduce)` selector also updated to `number-flow-react`.
- `frontend/src/lib/use-trend.ts`:
  - Default `durationMs` 700 -> 900.
- `frontend/src/components/paper-trading/cockpit-helpers.tsx`:
  - Two `transformTiming={{ duration: 700 }}` -> `{{ duration: 900 }}`.
- `frontend/src/components/paper-trading/positions-columns.tsx`:
  - One `transformTiming={{ duration: 700 }}` -> 900.
- `frontend/src/app/page.tsx`:
  - One `transformTiming={{ duration: 700 }}` -> 900.

ZERO backend changes. ZERO new npm deps. ZERO test scaffolding renames.

## Immutable success criteria

1. `npx tsc --noEmit` exit 0.
2. `npx vitest run` 178+ passed.
3. `python tests/verify_phase_23_1_17.py` ok.
4. `git diff HEAD -- frontend/package.json` empty.
5. `git diff --stat HEAD -- backend/` empty.
6. `grep -c "number-flow\[data-pyfa-trend" frontend/src/app/globals.css` returns 0 (old selector gone).
7. `grep -c "number-flow-react\[data-pyfa-trend" frontend/src/app/globals.css` returns 2 (up + down).
8. `grep -c "transformTiming.*900" frontend/src/components/paper-trading/cockpit-helpers.tsx frontend/src/components/paper-trading/positions-columns.tsx frontend/src/app/page.tsx` returns 4 (one per NumberFlow consumer).
9. `grep -c "transformTiming.*700" frontend/src` returns 0 (no stale 700ms literals).
10. `grep -c "pyfa-tint-up 900ms\|pyfa-tint-down 900ms" frontend/src/app/globals.css` returns 2.
11. ZERO emojis introduced.
12. NO `npm run build`, NO `rm -rf .next/*`, NO `npm install`.

## /goal integration gates

1. tsc + vitest green. 2. ZERO `npm run build`. 3. Zero emojis. 4. Log LAST / no masterplan flip.
