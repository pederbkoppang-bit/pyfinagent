# Experiment Results — phase-47.5: UX foundation (design-system enforcement layer)

**Cycle:** 5 of the production-ready+money push (priority-7 UX). FREE (frontend; no project LLM spend).
**Step:** 47.5 | **Result:** ready for Q/A.

## What was built / changed (4 new files + 2 compliance fixes; ADDITIVE)
1. NEW `frontend/src/lib/design-tokens.ts` — `tokens` object with semantic maps: `text`
   (slate-100..500 per frontend.md §6 contrast tiers), `surface` (navy-800/70 card), `border`,
   `hover` (navy-700/40), `focusRing` (the exact proven OpsStatusBar idiom), `transition`, `status`
   (emerald/amber/rose/slate/sky /15). Every value is a COMPLETE literal class string (JIT-safe,
   frontend.md rule 3); navy/slate only (rule 1); no light-mode base (rule 2). Exports `StatusVariant`.
2. NEW `frontend/src/components/ui/Button.tsx` — variants primary|secondary|ghost|danger via Record
   maps + clsx; bakes the uniform focusRing + min 24px target; CSS `active:scale-95` press (NOT the
   ~34kb Motion bundle — reserved for W6 page choreography). Forwards ButtonHTMLAttributes.
3. NEW `frontend/src/components/ui/StatusBadge.tsx` — consumes `tokens.status`; success|warning|error|
   neutral|info. NEW `frontend/src/components/ui/index.ts` barrel.
4. FIX `EmptyState.tsx` — zinc -> slate (lines 34/40/42 -> slate-400/300/500), frontend.md §1/§6.
5. FIX `DataTable.tsx:80` — dropped the `bg-white`/`border-zinc-200` light-mode base ->
   `border-navy-700 bg-navy-900` (dark-only project, frontend.md rule 2).

ADDITIVE boundary: the ~120 existing inline-class sites are NOT migrated this cycle (that's W5) ->
regression-free. No new npm dependency (clsx + motion already installed) -> no launchctl kickstart.

## Verbatim verification output
```
files: design-tokens.ts + ui/{Button,StatusBadge,index} exist; EmptyState no-zinc; DataTable filter light-base dropped
$ npx tsc --noEmit                          -> TSC_EXIT=0  (0 error lines)
$ npm run build (dev server RUNNING)        -> EXIT 1: PageNotFoundError /agents,/agent-map,/paper-trading/learnings
        ^ NOT my change: those pages exist on disk; classic next build vs next dev .next contention
          (frontend launchd runs `next dev` KeepAlive=true on the same .next).
$ launchctl unload frontend; rm -rf .next; npm run build  -> BUILD3_EXIT=0  (all routes compiled)
$ launchctl load frontend                   -> frontend HTTP 302 (NextAuth redirect = up)
```

## Success-criteria mapping (masterplan phase-47.5)
1. design-tokens.ts semantic maps, JIT-safe literals, navy/slate, no dynamic concat — **MET**.
2. ui/Button (4 variants, focus ring + 24px) + StatusBadge (4+ variants) + index barrel — **MET**.
3. EmptyState zinc->slate + DataTable filter light-base dropped — **MET** (grep: EmptyState no-zinc; DataTable filter line clean).
4. npm run build succeeds + no new dep + additive/regression-free — **MET** (isolated BUILD3_EXIT=0; the running-dev failure was .next contention on unrelated pages, documented + resolved; tsc 0 errors).

## Scope honesty (frontend.md rule 5)
The new `Button`/`StatusBadge` are ADDITIVE — not yet wired into any page (regression-free boundary),
so their variant VISUAL verification is **PENDING** a later wiring cycle (W5); nothing renders them yet.
`EmptyState`'s zinc->slate is a token-only swap on a component used by real pages — build+typecheck
verified, low-risk. The `next build` vs `next dev` `.next` contention is a standing operational gotcha
(documented in live_check_47.5.md) — production builds must unload the dev server first; this is NOT a
defect introduced by this change.

## Files
frontend/src/lib/design-tokens.ts, frontend/src/components/ui/{Button.tsx,StatusBadge.tsx,index.ts},
frontend/src/components/states/EmptyState.tsx, frontend/src/components/DataTable.tsx,
.claude/masterplan.json (phase-47.5 added), handoff/current/{contract.md,
research_brief_phase_44_11_design_system.md, live_check_47.5.md}.
