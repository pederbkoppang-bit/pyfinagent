# Experiment Results — phase-53.2 (UX elevation + WCAG AA)

**Date:** 2026-06-10. **Status:** complete. Bounded adoption pass (P2+P3+P4+P6); build +
tsc green; a11y recorded (axe 0 violations + live keyboard-focus proof); broader rollout
documented as follow-ups. Additive; DO-NO-HARM (behavior + states preserved). $0.

## What was done

The consistent surface (design-tokens.ts, ui/ primitives, states/ library) already
existed but was un-adopted, so 53.2 is a bounded adoption pass:
- **P2** app-wide WCAG-2.2-AA visible-focus baseline (globals.css, unlayered `:where()`).
- **P3** axe script now tests `wcag22a,wcag22aa`.
- **P4** zinc→navy/slate palette unification across 4 components (53 swaps, 0 remaining).
- **P6** `scroll-padding-top` (SC 2.4.11).

## Files changed

| File | Change |
|------|--------|
| `frontend/src/app/globals.css` | +unlayered `:where(...):focus-visible` AA focus baseline (P2) + `html scroll-padding-top` (P6). |
| `frontend/package.json` | axe `--tags` += `wcag22a,wcag22aa` (P3). |
| `frontend/src/components/AnalysisProgress.tsx` / `CommandPalette.tsx` / `DataTable.tsx` / `LiveBadge.tsx` | zinc→navy/slate (P4; 23/18/7/5 swaps). |

## Verification output (verbatim)

```
npx tsc --noEmit                 -> EXIT 0
npx eslint <4 components>        -> 0 errors (3 pre-existing warnings)
npm run build                    -> GREEN (24/24 routes)
npm run axe (/login, +wcag22aa)  -> axe-core 4.11.3, 0 violations found
grep zinc- in the 4 files        -> 0 ; DOM on /paper-trading/positions -> anyZincClassInDom:false
Playwright keyboard focus (/agents): "Analyze" (no-ring, boxShadow none) -> outlineColor rgb(56,189,248) solid;
   ringed controls keep their ring (outline suppressed -> no double-indicator)
```

## Acceptance-criteria mapping (phase-53.2 — VERBATIM)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | research gate passed (UX + a11y sources cited) + documented all-pages audit vs design-tokens.ts + ui/ identifies the unification deltas | PASS — researcher gate (7 sources, recency scan); P1-P10 + OP1/OP2 worklist in research_brief.md |
| 2 | unification changes land; no emoji (icons via @/lib/icons), Recharts dark, scrollbar-thin, error/loading/empty states preserved on every touched page | PASS — P2/P3/P4/P6 landed; zero emoji (preserved); Recharts dark + scrollbar-thin untouched; P4 is className-only (states preserved) |
| 3 | npm run build SUCCEEDS + npx tsc --noEmit passes; an a11y check (keyboard/focus/contrast WCAG AA) recorded; no behavioral/data regression | PASS — build green, tsc 0; axe 0 violations + live keyboard-focus proof; no behavioral change (CSS/palette/test-config only) |
| 4 | live_check_53.2.md records build/types + a11y evidence + OPERATOR-TO-CONFIRM visual (authed pages) | PASS — live_check_53.2.md written |

## DO-NO-HARM / scope honesty

- Bounded ADOPTION, not redesign. P2 is additive zero-specificity CSS that NEVER fights
  component rings (verified: ringed buttons keep their ring; only bare elements get the
  outline). P4 is className-only (no markup/state/behavior removed). P3 is a test-config
  string. No money-path / data code touched (`git diff` = globals.css + package.json + 4
  component palette files).
- Honest a11y scope: axe on /login = 0 violations but covers only the pre-auth page;
  authed-route Lighthouse/axe + manual keyboard/SR are operator-only (NextAuth wall) —
  flagged in live_check OP1/OP2. The broad ErrorState/ui-Button/token-migration rollout is
  documented as follow-ups (phase-44.x), not silently skipped.
- The transient skip-auth console errors (`:8000/portfolio` 404 + `useLiveNav` TypeError)
  were restart artifacts (backend re-checked healthy 200/200), NOT a 53.2 regression; the
  useLiveNav undefined-guard is a pre-existing follow-up.
- No emoji; navy/slate palette; JIT-safe; icons via `@/lib/icons`.
