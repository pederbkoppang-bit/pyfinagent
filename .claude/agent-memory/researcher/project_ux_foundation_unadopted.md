---
name: ux-foundation-unadopted
description: phase-47.5 design-tokens + 44.1 states-lib + ui/ primitives are BUILT but ~UNADOPTED by pages; UX consistency work = adoption not new design; the WCAG-AA gaps are narrow + many prompt-assumed findings are already-satisfied
metadata:
  type: project
---

phase-53.2 UX-elevation research (2026-06-10) found the consistent surface
already EXISTS in code; the pages just don't import it. UX consistency steps
are an ADOPTION problem, not a design problem.

**Why:** phase-47.5 shipped `frontend/src/lib/design-tokens.ts` (tokens.text
5 contrast tiers / surface / border / hover / `focusRing` :40 `:focus-visible
ring-2 ring-sky-400` / transition / status pills) + `ui/Button` (bakes
focusRing + `min-h/w-[24px]` 2.5.8 fix at :52) + `ui/StatusBadge`. phase-44.1
shipped `states/` (LoadingState/EmptyState/ErrorState/OfflineState/StaleDataState).
But grep-proven adoption is near-zero: `tokens` imported by 0 pages (only the 2
ui/ primitives); `ui/Button` + `ui/StatusBadge` by 0 consumers; `LoadingState`/
`ErrorState` by 0 pages; `EmptyState` by exactly 2 (performance:7, reports:26).
phase-43.0 DoD audit independently scores UX 0/12. So the bounded, low-risk,
high-value worklist = swap hand-rolled strings for the existing primitives.

**How to apply:** For any UX-consistency / design-token / states-library step,
do NOT propose a new token system or new primitives — they exist. Propose
ADOPTION deltas (each is a mechanical swap + a grep that proves it). The single
biggest mechanical win = ~36 inline `border-rose-*` error-banner divs across 13
pages -> the unused `ErrorState` (preserve each retry/dismiss/message =
DO-NO-HARM). The full ~1246-site slate-token migration is the UNBOUNDED part —
defer it (maps to pending steps 44.2 / 44.9).

**WCAG-AA reality (corrects common prompt assumptions):**
- 2.4.13 Focus Appearance is **AAA, not AA**. AA focus = 2.4.7 (any visible
  indicator; `:focus-visible` C45 is sufficient) + 2.4.11 (not ENTIRELY obscured;
  fix sticky-chrome with `scroll-padding-top`). The repo's `tokens.focusRing` is
  the correct AA-sufficient idiom (and clears 1.4.11 3:1 on bg-navy-800/70).
- These are ALREADY satisfied — never manufacture them as findings: NO true emoji
  (0; every non-ASCII hit is a typographic arrow -> ↔ ↑↓ ↵, valid); skip-link
  present (layout.tsx:25-28); `lang="en"` (layout.tsx:20); page-shell +
  scrollbar-thin consistent on every route (paper-trading/* inherit from
  paper-trading/layout.tsx:326/439; /login intentionally shell-less);
  prefers-reduced-motion honored in globals.css; animation durations fine (4
  values, token-mappable).
- The REAL AA gap = 9 pages with onClick buttons lacking `:focus-visible`
  (agents, cron, backtest, observability, paper-trading/manage, performance,
  settings, signals, sovereign).
- Contrast is NOT a blanket failure: design-tokens.ts:17-21 certifies slate
  tiers >=4.6:1 on navy; only slate-400/500 on RISK-RELEVANT numbers is at risk
  (frontend.md:46 already forbids it) -> targeted grep, not global swap.
- `frontend/package.json:14` `axe` script tags only wcag2a/2aa/21a/21aa -
  MISSING wcag22a/22aa (under-tests vs the 2.2 AA bar; 1-line fix).
- axe (/login is the only unauthenticated route, NextAuth wall) + Lighthouse
  catch ~30-40% of WCAG; keyboard-nav / reading-order / authed routes are
  OPERATOR-only live_check. DataTable (UX-6) adopted on 3 of 4 (trades,
  positions, reports); gap = /backtest + /cron (large, defer).

Related: [[feedback_no_emojis]] (the rule is real but already satisfied in code).
