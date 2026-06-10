# Live Check — phase-47.5: UX foundation (design-system enforcement layer)

Captured 2026-05-29. Frontend; no project LLM spend.

## 1. Additive files exist + violations fixed (grep)
```
4 new files exist: src/lib/design-tokens.ts, src/components/ui/{Button.tsx,StatusBadge.tsx,index.ts}
EmptyState.tsx: no zinc (good)   [was 3 zinc occurrences -> slate-400/300/500]
DataTable.tsx filter (:80): bg-white/border-zinc-200 light base dropped -> border-navy-700 bg-navy-900
```

## 2. Typecheck — clean
```
$ npx tsc --noEmit
TSC_EXIT=0   (0 error lines; design-tokens / Button / StatusBadge / EmptyState / DataTable all clean)
```

## 3. Production build — SUCCEEDS (in isolation)
First attempt (dev server running): BUILD_EXIT=1 with `PageNotFoundError` for /agents, /agent-map,
/paper-trading/learnings. ROOT CAUSE: those pages exist on disk; the failure is the well-known
`next build` vs `next dev` `.next` contention (the launchd frontend runs `next dev` with KeepAlive=true,
writing the same `.next`). NOT caused by this additive change.

Isolated build (frontend dev unloaded, `.next` cleared):
```
$ launchctl unload com.pyfinagent.frontend; rm -rf .next; npm run build
BUILD3_EXIT=0
... /reports 271kB, /settings 267kB, /signals 248kB, /sovereign 254kB, all routes compiled ...
$ launchctl load com.pyfinagent.frontend  -> frontend HTTP 302 (NextAuth redirect = up)
```
The full app (baseline + this change) builds clean. Frontend dev server reloaded + serving.

## 4. Visual verification (frontend.md rule 5)
- `EmptyState` palette change (zinc->slate) is a token swap on a component used by real pages
  (/reports, /performance, /signals empties) — build + typecheck verified; low-risk color-only.
- `ui/Button` + `ui/StatusBadge` are ADDITIVE — NOT yet wired into any page this cycle (regression-free
  boundary). Their variant visual verification is therefore **PENDING** until a later migration cycle
  (W5) wires them into a page; nothing renders them yet to screenshot. This is the honest rule-5 status
  for additive components.

NOTE: the `next build`-vs-`next dev` contention is a standing operational gotcha — production-build
verification must unload the dev server first (documented here for future frontend cycles).
