# Contract — phase-53.2 (UX elevation + WCAG AA)

**Date:** 2026-06-10. **Tier:** complex. **Step:** phase-53.2 (P3). Frontend
consistency + accessibility; additive, DO-NO-HARM (preserve behavior + all states).

## N* delta (N* = Profit − Risk − Burn)

**Risk↓ (operability/accessibility):** a consistent, keyboard-accessible surface reduces
operator error + meets WCAG 2.2 AA on the controls. No P/B delta. No data/behavioral change.

## Research-gate summary

`researcher` ran FIRST (gate **PASSED**: 7 sources read in full, 18 URLs, recency scan,
14 internal files). Brief: `handoff/current/research_brief.md`. **Headline:** the phase-47.5
+ 44.1 consistent surface (`design-tokens.ts`, `ui/` primitives, `states/` library) is BUILT
but UN-ADOPTED — so 53.2 is a bounded ADOPTION problem, not a redesign. **Disproven false
work** (so GENERATE manufactures none): zero true emoji (non-ASCII = typographic arrows),
skip-link + `lang="en"` present, page-shell + scrollbar-thin already consistent,
prefers-reduced-motion honored, contrast certified ≥4.6:1 on the token tiers (only
slate-400/500 on risk-numbers is at-risk). **Spec corrections:** SC 2.4.13 Focus Appearance
is AAA (AA focus = 2.4.7 "any visible indicator" + 2.4.11 not-obscured); the `axe` script
under-tests (missing `wcag22aa` tags). `tokens.focusRing` (design-tokens.ts:40) is the
AA-sufficient idiom.

## All-pages unification audit (the documented delta worklist — criterion 1)

The researcher's prioritized worklist (P1-P10 + OP1/OP2) is in `research_brief.md`. Landed
this cycle (bounded, low-risk, strongly-verifiable): **P3** (axe AA tags), **P2** (focus
baseline), **P4** (zinc→navy/slate in 4 components), **P6** (scroll-padding). Documented
follow-ups (map to pending 44.x): P1 `states/ErrorState` adoption (~36 banners; LOW-MED
risk + weakly verifiable autonomously since error states are hard to trigger visually),
P5 `ui/Button` adoption, P7 `LoadingState` sweep, P8 DataTable on /backtest+/cron, P9 the
~1246-site slate-token migration. Operator-only: OP1 Lighthouse on authed routes, OP2
keyboard/screen-reader pass (the ~60% automation can't see).

## Immutable success criteria — VERBATIM from masterplan phase-53.2 (do NOT edit)

1. the research gate passed (UX best-practice + accessibility sources cited in the contract)
   and a documented all-pages audit against design-tokens.ts + the ui/ primitives identifies
   the unification deltas
2. the unification changes land with no emoji (icons via @/lib/icons), Recharts dark theme,
   scrollbar-thin, and error/loading/empty states preserved on every touched page
3. cd frontend && npm run build SUCCEEDS and npx tsc --noEmit passes; an accessibility check
   (keyboard nav + focus + contrast, WCAG AA target) is recorded; no behavioral/data regression
4. live_check_53.2.md records the build/types pass + the accessibility evidence + an
   OPERATOR-TO-CONFIRM visual section (authed pages behind the NextAuth wall)

## Plan steps

1. **P3** — `frontend/package.json:14` axe script: add `wcag22a,wcag22aa` to `--tags`.
2. **P2 + P6** — `frontend/src/app/globals.css` `@layer base`: a ZERO-SPECIFICITY global
   focus baseline `:where(a,button,[role=button],[role=tab],input,select,textarea,summary):focus-visible`
   → `outline: 2px solid sky-400; outline-offset: 2px` (zero specificity ⇒ never fights the
   component `tokens.focusRing`; gives every interactive element an AA-visible indicator) +
   `html { scroll-padding-top: ... }` (SC 2.4.11). No `!important`.
3. **P4** — swap stray `zinc-*` → navy/slate tokens in `AnalysisProgress.tsx`,
   `CommandPalette.tsx`, `DataTable.tsx`, `LiveBadge.tsx` (mapping: bg/border zinc→navy,
   text zinc→slate). Visual-only; preserve all structure/behavior/states.
4. **Verify** — `npm run build` green; `npx tsc --noEmit` 0; `grep zinc- src/` → 0 in the 4
   files; Playwright skip-auth: Tab through a page → visible focus ring (P2), pages render
   unchanged (P4); `npm run axe` on /login now runs the wcag22aa rules; restore the auth
   gate (302). Write `live_check_53.2.md`.
5. **Fresh qa → log → flip → commit.**

## Guardrails / DO-NO-HARM

- ADOPTION not redesign. Preserve behavior + error/loading/empty states on every touched
  page (P4 is visual-only; P2 is additive zero-specificity CSS; P3 is a test-config string).
- No emoji (already true — preserve); icons via `@/lib/icons`; navy/slate palette (never
  zinc); scrollbar-thin + Recharts dark (already consistent — preserve). JIT-safe literals.
- The global focus baseline uses `:where()` (specificity 0) so it NEVER double-rings or
  overrides component focus styles. No `!important`.
- Bounded: do NOT attempt the full slate-token migration (P9) or the 36-banner ErrorState
  migration (P1) this cycle — documented follow-ups. Visual confirmation of authed pages is
  the operator section in live_check (NextAuth wall).

## References

`handoff/current/research_brief.md`; `frontend/src/lib/design-tokens.ts:40` (focusRing);
`frontend/src/app/globals.css`; `frontend/package.json:14`; the 4 zinc components;
`.claude/rules/frontend.md` + `frontend-layout.md`. External: W3C WCAG 2.2 (2.4.7/2.4.11/
1.4.3/1.4.11/2.5.8), Deque axe-core, Material 3 / Apple HIG.
