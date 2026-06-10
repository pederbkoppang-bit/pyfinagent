# live_check 53.2 — UX elevation + WCAG AA (build/types + a11y evidence + operator visual)

**Date:** 2026-06-10. Bounded adoption pass (the consistent surface already existed,
un-adopted). Additive; DO-NO-HARM (behavior + all states preserved).

## Build / types proofs (criterion 3)

```
npx tsc --noEmit                 -> EXIT 0
npx eslint <4 swapped components>-> EXIT 0 (3 warnings, 0 errors; pre-existing TanStack/mount-guard)
npm run build (next build)       -> GREEN (24/24 routes; route table + Static/Dynamic legend)
```

## Accessibility evidence (criterion 3 — keyboard/focus/contrast, WCAG 2.2 AA)

- **axe-core 4.11.3** on `/login` with the now-expanded tag set
  `wcag2a,wcag2aa,wcag21a,wcag21aa,wcag22a,wcag22aa` (P3 — the script previously omitted
  the 2.2 AA rules): **0 violations found**.
- **Keyboard focus (P2), verified live via Playwright** on `/agents`:
  - The "Analyze" sidebar toggle (a control with NO ring class, `boxShadow: none`) now
    shows `outlineColor: rgb(56,189,248)` solid — the AA visible-focus fallback (SC 2.4.7)
    fires on the previously-bare gap elements (12 of 25 interactive elements on that page
    lacked a focus ring).
  - Already-ringed controls (login "Sign in", skip-link) keep their box-shadow
    `ring-2 ring-sky-400`; the global outline is correctly SUPPRESSED on them by their own
    `focus:outline-none` specificity → **no double-indicator** (the `:where()` baseline is
    a true fallback, not an override).
- **Contrast (SC 1.4.3):** the token tiers certify ≥4.6:1 on navy (per `frontend.md` +
  the researcher's W3C check); the P4 zinc→navy/slate swap moved 4 components onto those
  certified tiers. No risk-relevant number was demoted to slate-400/500.

## Unification changes landed (criteria 1 + 2)

The documented all-pages audit (research_brief.md, P1-P10 + OP1/OP2) is the criterion-1
deliverable. Landed this cycle (bounded, low-risk, verified):

- **P2** — `globals.css`: app-wide WCAG-2.2-AA visible-focus baseline. UNLAYERED +
  `:where()` (specificity 0) so it never fights component rings; restores a visible
  outline on the ~no-ring gap elements. (Tailwind v3 compiles `@layer` to flat CSS, so an
  `@layer base` rule was specificity-defeated by `focus:outline-none` in utilities —
  moved unlayered + verified.)
- **P3** — `package.json:14`: added `wcag22a,wcag22aa` so `npm run axe` tests the 2.2 AA
  rules. Ran → 0 violations on /login.
- **P4** — zinc→navy/slate in `AnalysisProgress.tsx` (23), `CommandPalette.tsx` (18),
  `DataTable.tsx` (7), `LiveBadge.tsx` (5): 53 swaps, **0 zinc classes remain** in those
  files AND **0 zinc in the rendered DOM** (verified on /paper-trading/positions; the
  DataTable still renders, 2 rows). Property-aware mapping (bg/border→navy, text→slate,
  light-mode dark-text zinc-700/800/900 inverted to light slate).
- **P6** — `globals.css`: `html { scroll-padding-top: 5rem }` (SC 2.4.11 Focus Not Obscured).

No emoji introduced (the codebase has zero true emoji — non-ASCII are typographic arrows);
icons remain via `@/lib/icons`; Recharts dark theme + scrollbar-thin unchanged;
error/loading/empty states on every touched page PRESERVED (P4 is className-only; no
state/markup removed).

## OPERATOR TO CONFIRM (visual, behind the NextAuth wall)

When you log in (real session), please eyeball:
- **Keyboard focus:** Tab through any page — every interactive element should show a clear
  sky focus indicator (ringed buttons keep their ring; previously-bare controls now get an
  outline). Confirm no element is focus-invisible.
- **Palette:** `/agents` (AnalysisProgress + CommandPalette via Cmd-K), `/paper-trading/positions`
  (DataTable + LiveBadge) — confirm the navy/slate shades read consistently (no off "zinc" gray).
- **Lighthouse a11y on authed routes (OP1):** `npm run lighthouse:auth-home` (only you can,
  behind NextAuth) — target ≥95. axe on /login is 0-violations but covers only the pre-auth page.

## Documented follow-ups (NOT this cycle — DO-NO-HARM bounding)

P1 `states/ErrorState` adoption (~36 inline banners; LOW-MED risk + weakly verifiable
autonomously since error states are hard to trigger), P5 `ui/Button` adoption, P7
`LoadingState` sweep, P8 DataTable on /backtest+/cron, P9 the ~1246-site slate-token
migration. Operator-only: OP1 Lighthouse + OP2 keyboard/screen-reader pass on authed
routes (automation catches only ~20-50% of WCAG). These map to pending phase-44.x.

## Notes

- Transient console errors seen during the skip-auth Playwright probe
  (`:8000/portfolio` 404 + a `useLiveNav` undefined-`cash` TypeError) were restart
  artifacts (the frontend kickstart momentarily refused :3000/auth + the backend was
  mid-cycle); backend re-checked healthy (health=200, portfolio=200). NOT a 53.2
  regression (className/test-config/CSS changes cannot cause a data TypeError). The
  `useLiveNav` undefined-guard gap on a /portfolio 404 is a pre-existing robustness
  follow-up.
