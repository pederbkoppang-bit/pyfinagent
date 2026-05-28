# Contract — phase-47.5: UX foundation (design-system enforcement layer)

**Cycle:** 5 of the production-ready+money push (priority-7 UX). FREE (frontend; no project LLM spend).
**Step:** 47.5 | **Phase:** phase-47 | **Status:** in-progress | **Harness:** required | **Tier:** moderate.
Frontend work — held to `.claude/rules/frontend.md` + `frontend-layout.md`.

NOTE: 47.2 (first trade) PARKED on operator LLM-spend gate; 47.1/47.3/47.4 done+pushed. This is the
first priority-7 (UX) cycle — the only unblocked remaining work. W1 (live-promote endpoint) deferred
per test-env-first.

## Research-gate summary (PASSED)
Researcher `a9bfe681d59b10293`, tier=moderate, `gate_passed: true`. 7 sources in full, ~25 URLs,
recency scan, 11 internal files. Brief: `research_brief_phase_44_11_design_system.md`.

Key finding (anti-duplication): phase-44.1's "design tokens" title was misleading — `git show db1e6208`
proves it shipped the states lib + hooks lib + CommandPalette + featureFlags, NOT a token module. So
`@/lib/design-tokens.ts` + the `ui/` dir are genuinely NEW (no overlap). The design system "exists on
paper but isn't enforced": tokens in tailwind.config/globals.css, but text/hover/focus/button classes
hand-composed across ~120 files; `@/lib/motion.ts` (6 presets) orphaned. clsx ^2.1.0 + motion ^12.38.0
already installed -> NO npm install -> NO launchctl kickstart.

## Hypothesis
An ADDITIVE semantic-token module + the first shared ui components (Button, StatusBadge) give every
page ONE vocabulary to migrate to, making "consistent layout/design across all pages" enforceable —
without regressing anything (the ~120 existing sites are NOT migrated this cycle).

## Immutable success criteria (verbatim from masterplan.json phase-47.5)
1. NEW frontend/src/lib/design-tokens.ts exports semantic token maps (text, surface, border, hover, focusRing, transition, status) as JIT-safe literal navy/slate class strings (no zinc, no dynamic class concatenation per frontend.md 1.3)
2. NEW frontend/src/components/ui/Button.tsx (variants primary|secondary|ghost|danger, focus-visible ring + >=24px target) + StatusBadge.tsx (success|warning|error|neutral) + index.ts barrel
3. frontend.md violations fixed: EmptyState.tsx zinc->slate; DataTable.tsx filter drops the bg-white/border-zinc-200 light base
4. cd frontend && npm run build succeeds; no new npm dependency added; additive only (the ~120 existing sites NOT migrated this cycle -> regression-free)

## Plan steps
1. NEW `frontend/src/lib/design-tokens.ts` — `tokens` object: text (slate-100..500 per frontend.md §6),
   surface (navy-800/70 card), border, hover (navy-700/40), focusRing (the exact OpsStatusBar idiom),
   transition, status (emerald/amber/rose/slate/sky /15). All complete literal strings (JIT-safe).
2. NEW `frontend/src/components/ui/Button.tsx` (variants via Record maps + clsx; focus ring + 24px
   target; CSS active:scale-95, NOT Motion), `StatusBadge.tsx` (consumes tokens.status), `index.ts` barrel.
3. Fix `EmptyState.tsx` zinc->slate (lines 34/40/42 -> slate-400/300/500); `DataTable.tsx:80` drop the
   `bg-white`/`border-zinc-200` light-mode base (dark-only project, frontend.md rule 2).
4. Verify: file-existence + grep no-zinc + `cd frontend && npm run build` succeeds. Write live_check_47.5.md.
   Per frontend.md rule 5: EmptyState palette change is verifiable; new Button/StatusBadge variants are
   ADDITIVE (not yet wired into a page) -> variant visual verification marked PENDING for a later wiring cycle.

## Blast radius
Frontend only, additive (4 new files) + 2 small compliance fixes to EmptyState/DataTable. No dep
install. No behavior/trading change. Existing pages untouched (no migration this cycle) -> regression-free.

## References
- `research_brief_phase_44_11_design_system.md` (gate); `ux_roadmap.md` (W2); frontend.md §1/§3/§6 + rules 1-5
- `frontend/src/components/states/EmptyState.tsx`, `frontend/src/components/DataTable.tsx`,
  `frontend/src/components/OpsStatusBar.tsx` (focus idiom), `frontend/src/lib/motion.ts` (orphaned, for later W6)
