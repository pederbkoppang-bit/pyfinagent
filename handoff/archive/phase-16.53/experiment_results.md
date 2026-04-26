---
step: phase-16.53
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/app/settings/page.tsx (3 single-line edits dropping max-w-4xl from tab grid wrappers)
---

# Experiment Results -- phase-16.53

## What was done

Removed the `max-w-4xl` class from the 3 tab-content grid wrappers in
`settings/page.tsx`. The constraint was capping each tab's content area
at ~896px wide, leaving ~30% dead whitespace on the right at typical
laptop widths. With the constraint removed, the 2-column BentoCard
layout fills the natural width of the scrollable zone (which itself
is bounded only by the viewport minus the sidebar).

## Deliverable

### `frontend/src/app/settings/page.tsx` (3 edits via replace_all)

Single string replacement applied to all 3 occurrences:
- `grid max-w-4xl grid-cols-1 gap-6 lg:grid-cols-2` -> `grid grid-cols-1 gap-6 lg:grid-cols-2`

The 3 affected wrappers:
- L601: Models & Analysis tab grid
- L794: Cost & Weights tab grid
- L978: Performance tab grid

NOT touched (intentional, per research brief / canonical pattern):
- L580 tab bar `max-w-fit` -- pills should hug content, not stretch
- Any `max-w-*` on individual cards inside the grids -- those are fine

## Verification (verbatim, immutable from masterplan)

```
$ cd frontend && npx tsc --noEmit
(exit 0; no output)

$ cd frontend && npm run lint
0 errors, 34 warnings (all pre-existing in unmodified files)

$ grep -c 'max-w-4xl' frontend/src/app/settings/page.tsx
0
```

## Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/app/settings/page.tsx` | edit | 3 single-string replacements via Edit replace_all |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.53-research-brief.md` | created (internal-only) | -- |

NO new files. NO new dependencies. NO test additions (pure-UI; visual change verified by operator at next browser load).

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | All 3 `max-w-4xl` constraints on tab grid wrappers removed | PASS (grep returns 0) |
| 2 | Tab bar `max-w-fit` retained | PASS (L580 unchanged) |
| 3 | `cd frontend && npx tsc --noEmit` exits 0 | PASS |
| 4 | `cd frontend && npm run lint` -- 0 errors, no new warnings | PASS (34 warnings unchanged from prior cycles) |
| 5 | Operator visual confirmation at next browser load | DEFERRED (operator will refresh /settings) |

## Honest disclosures

1. **Single-file 3-line change.** Tightest possible scope for a UX cycle.
2. **No new tests.** pure-UI layout change; verified by `npx tsc --noEmit` + the user's earlier screenshot evidence + operator visual confirmation.
3. **Cycle-2 not needed.** First-pass clean.
4. **No regression.** No other files modified. The cards inside the grids are unchanged; only the wrapper class.
5. **Operator-flagged via screenshot** -- the same pattern is now self-evident: any future page using `max-w-4xl` on a 2-col tab grid will exhibit the same dead-space anti-pattern. Future audit hint.

## Closes

Net-new task #82 (UAT-16.53). Adds new step phase-16.53 to masterplan.

## Next

Spawn Q/A.
