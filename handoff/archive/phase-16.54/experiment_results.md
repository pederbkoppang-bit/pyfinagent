---
step: phase-16.54
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/components/RedLineMonitor.tsx (1-line: h-64 -> h-48 for non-compact)
---

# Experiment Results -- phase-16.54

## What was done

Reduced the non-compact RedLineMonitor chart container height from
`h-64` (256px) to `h-48` (192px). The compact branch (`h-72`, used by
the homepage hero) is unchanged. This shrinks the sovereign page's
RedLine card by ~64px, materially reducing the dead space below the
neighboring AlphaLeaderboard on the two-hero row.

## Deliverable

### `frontend/src/components/RedLineMonitor.tsx` (1-line edit at L107)

```tsx
- className={compact ? "h-72" : "h-64"}
+ className={compact ? "h-72" : "h-48"}
```

The compact branch (`h-72` = 288px) is preserved -- this is the path
used by `frontend/src/app/page.tsx` homepage hero embed (with
`min-h-[55svh]` wrapper). No homepage regression.

## Verification (verbatim, immutable from masterplan)

```
$ cd frontend && npx tsc --noEmit
(exit 0; no output)

$ cd frontend && npm run lint
0 errors, 34 warnings (all pre-existing in unmodified files)
```

## Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/components/RedLineMonitor.tsx` | edit (L107) | h-64 -> h-48 (non-compact only) |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.54-research-brief.md` | created (internal-only) | -- |

NO other files modified. NO new tests.

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Non-compact RedLine chart container height reduced | PASS (h-64 -> h-48) |
| 2 | Compact branch (homepage hero) preserved | PASS (h-72 unchanged) |
| 3 | `cd frontend && npx tsc --noEmit` exits 0 | PASS |
| 4 | `cd frontend && npm run lint` -- 0 errors, no new warnings | PASS |
| 5 | Operator visual confirmation at next /sovereign load | DEFERRED (operator will refresh) |

## Honest disclosures

1. **Single 1-line change.** Tightest possible scope.

2. **Horizontal scroll on AlphaLeaderboard NOT addressed** (it was visible in the operator's screenshot but the explicit ask was about deadspace + Red Line size). Out of scope per contract.

3. **No regression risk.** Only the non-compact branch changed; homepage hero (which uses compact=true) is unchanged.

4. **Cycle-2 not needed.** First-pass clean.

5. **Pattern hint for future cycles:** `h-48` (192px) is closer to `h-56` (224px) and `h-64` (256px) Tailwind spacing units. If the chart looks too cramped after the visual check, a follow-up could try `h-56` as a middle ground.

## Closes

Net-new task #83 (UAT-16.54). Adds new step phase-16.54 to masterplan.

## Next

Spawn Q/A.
