---
step: phase-16.48
title: UX audit pass A -- low-risk pages + spot-check components
cycle_date: 2026-04-26
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/src/app/login/page.tsx (fix min-h-screen)
  - frontend/src/app/signals/page.tsx (canonical two-zone shell)
  - frontend/src/app/performance/page.tsx (canonical two-zone shell + scrollbar-thin + loading/empty states)
  - frontend/src/components/AlphaLeaderboard.tsx (scrollbar-thin)
---

# Sprint Contract -- phase-16.48

## Research-gate summary

`handoff/current/phase-16.48-research-brief.md`. tier=simple, internal-only,
gate_passed=true. 10 files audited (5 pages + 5 components, 2353 LOC).
Zero phosphor-import violations (post-16.39 sweep holds). Zero emoji
violations.

## Violations found (6, in 4 files)

| Sev | File:line | Violation | Fix |
|-----|-----------|-----------|-----|
| HIGH | `login/page.tsx:35` | `min-h-screen` (rule says never) | -> `h-screen overflow-hidden` |
| MED | `signals/page.tsx:88-90` | Single-zone shell; header scrolls with content | Split into fixed-header + scrollable two-zone |
| MED | `performance/page.tsx:41-43` | Same single-zone shell | Same two-zone split |
| LOW | `performance/page.tsx:179` | `overflow-x-auto` missing `scrollbar-thin` | Add class |
| LOW | `performance/page.tsx` cost-history section | Missing loading + empty states | Add skeleton + empty banner |
| LOW | `AlphaLeaderboard.tsx:190` | `overflow-x-auto` missing `scrollbar-thin` | Add class |

## Concrete plan

### Fix 1: login/page.tsx (1 line)
`<div className="flex min-h-screen items-center justify-center bg-[#0B1120]">`
->
`<div className="flex h-screen items-center justify-center overflow-hidden bg-[#0B1120]">`

Auth page; no Sidebar; preserves centering. `overflow-hidden` per rule.

### Fix 2: signals/page.tsx two-zone shell

Restructure JSX from:
```tsx
<div className="flex h-screen overflow-hidden">
  <Sidebar />
  <main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">
    <div className="mb-8"><h2>...</h2><p>...</p></div>
    {/* rest scrolls together */}
  </main>
</div>
```
to:
```tsx
<div className="flex h-screen overflow-hidden">
  <Sidebar />
  <main className="flex flex-1 flex-col overflow-hidden">
    <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
      <div className="mb-6"><h2>...</h2><p>...</p></div>
    </div>
    <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
      {/* rest scrolls; header stays pinned */}
    </div>
  </main>
</div>
```

### Fix 3: performance/page.tsx two-zone shell (same pattern as signals)

### Fix 4: performance/page.tsx:179 + AlphaLeaderboard.tsx:190

Add `scrollbar-thin` to existing `overflow-x-auto` classes.

### Fix 5: performance/page.tsx cost-history loading + empty states

Add inline loading spinner when `loading && !costHistory.length`. Add empty
state ("No cost history yet — runs will appear after the first analysis.")
when `!loading && costHistory.length === 0`.

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
npx tsc --noEmit && \
! grep -q "min-h-screen" src/app/login/page.tsx && \
grep -q "flex flex-1 flex-col overflow-hidden" src/app/signals/page.tsx && \
grep -q "flex flex-1 flex-col overflow-hidden" src/app/performance/page.tsx && \
[ "$(grep -c 'overflow-x-auto scrollbar-thin' src/app/performance/page.tsx src/components/AlphaLeaderboard.tsx)" -ge "2" ] && \
npm run lint 2>&1 | grep -c '@phosphor-icons/react' | grep -q '^0$'
```

Plus:
- `tsc_clean`: exit 0
- `lint_clean`: 0 phosphor warnings, 0 NEW errors (34 pre-existing react-hooks warnings unchanged)
- `no_emoji_introduced`: zero new emoji codepoints in changed files
- `no_backend_changes`: git status confined to 4 frontend files + handoff/* rolling

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. login/page.tsx no longer matches `min-h-screen`; structurally correct (flex centered card preserved).
3. signals + performance shells use canonical two-zone pattern (header pinned, content scrolls).
4. scrollbar-thin added to both noted overflow-x-auto containers.
5. performance cost-history section has loading + empty states.
6. tsc clean; no new lint errors; no phosphor warnings.
7. Settings page unchanged (it was the canonical reference, not a target).
8. No backend changes.
