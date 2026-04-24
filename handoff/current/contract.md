# BudgetDashboard TypeError hotfix (v2) -- Contract

## Step

Re-apply the BudgetDashboard TypeError fix that was silently reverted
between the original 2026-04-21 hotfix landing and the 2026-04-24
commit `1122a021`. The research brief + types.ts interfaces + api.ts
helper WERE included in the commit; the actual `BudgetDashboard.tsx`
component edits were NOT staged because they'd already been reverted
by the same mechanism that reverted `main.py` pre-commit.

## Research-gate summary

- Reuses the prior research brief:
  `handoff/current/phase-bugfix-budget-dashboard-research-brief.md`
  (committed in `1122a021`; 6 sources read-in-full on Next.js 16
  middleware, Auth.js v5, React 19 error-boundary patterns; gate_passed=true).
- Tier: simple -- identical root cause, identical fix, verified once.
- Skip justification: re-applying a previously-gated fix; no new
  research load-bearing.

## Root cause (from prior brief, unchanged)

`BudgetDashboard.tsx` uses raw `fetch()` against `/api/backtest/budget/summary`
without a Bearer header. The endpoint is auth-gated -> 401 body
`{"detail": "Authentication required"}` handed straight to `setData`.
`data.summary` is undefined, and `s.total_monthly.toFixed(0)` at
line 166 crashes.

## Fix (3 files, same as prior cycle)

1. `frontend/src/lib/types.ts` -- restore `BudgetData`, `BudgetSummary`,
   `CostItem`, `MonthlyHistory` interfaces (currently absent per grep).
2. `frontend/src/lib/api.ts` -- restore `getBudgetSummary()` function
   (currently absent per grep).
3. `frontend/src/components/BudgetDashboard.tsx` -- swap raw fetch for
   `getBudgetSummary()` + add `if (!s) return <UnavailableBanner />`
   null-guard after `const s = data.summary`.

## Immutable success criteria

1. `BudgetDashboard.tsx` contains no raw `fetch(` call.
2. `getBudgetSummary()` is exported from `frontend/src/lib/api.ts`.
3. `BudgetData` interface is exported from `frontend/src/lib/types.ts`.
4. `if (!s)` null-guard present in `BudgetDashboard.tsx` immediately
   after `const s = data.summary`.
5. `cd frontend && npx tsc --noEmit` exits 0 (ignoring pre-existing
   `HarnessSprintTile.test.tsx` warning).

Verification:
```
cd frontend && npx tsc --noEmit 2>&1 | grep -v HarnessSprintTile.test.tsx | grep -E 'error' | head -5 ; \
! grep -E 'fetch\(\s*`\$\{process\.env\.NEXT_PUBLIC_API_URL' frontend/src/components/BudgetDashboard.tsx ; \
grep -q 'getBudgetSummary' frontend/src/lib/api.ts ; \
grep -q 'export interface BudgetData' frontend/src/lib/types.ts ; \
grep -q 'if (!s)' frontend/src/components/BudgetDashboard.tsx ; \
echo "all checks passed"
```

## Plan steps

1. Restore the 3 file edits (I have the exact content from the prior cycle's commit).
2. Run `tsc --noEmit`.
3. Run the 4 grep assertions.
4. **Commit immediately** to prevent another silent revert.
5. Q/A.

## Anti-revert rationale

The autonomous-harness cycles appear to `git checkout` working-tree
files on their own schedule. Committing immediately turns edits into
tracked history they can't drop silently.

## References

- Prior brief: `handoff/current/phase-bugfix-budget-dashboard-research-brief.md`
- Prior harness_log entry: `phase-bugfix-budget-dashboard -- 2026-04-21 -- result=PASS`
