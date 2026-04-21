# Contract -- Cycle 76 / phase-4.7 step 4.7.7

Step: 4.7.7 Virtual-fund learnings dashboard
       (reconciliation / kill-switch / regime / MFE-MAE)

## Hypothesis

A single dashboard component that surfaces four learning signals
from the virtual-fund (paper) trading loop:
- Top-10 reconciliation divergences (paper fill vs BQ-sim fill,
  phase-3.7.5 router data)
- Kill-switch trigger distribution (count by trigger reason)
- Regime underperformance buckets (returns by market regime)
- (out-of-scope this cycle) MFE/MAE per trade is already surfaced
  by the existing `MfeMaeScatter.tsx` component on /paper-trading;
  we reference it rather than duplicate.

The data backends partly exist (execution_router shadow fills,
paper_kill_switch.py, regime_detection from the phase-8.5 spec).
For this cycle we ship the UI primitive + discriminating vitest
tests with a stable `Candidate`-style data prop, matching the
4.7.4 pattern.

## Scope

Files created:

1. **NEW** `frontend/src/components/VirtualFundLearnings.tsx`
   - Page-layout-compatible component (NOT a full page).
   - Four sections with `data-section` markers:
     * `reconciliation-divergences` (table of top-10)
     * `kill-switch-distribution` (segmented bar + legend)
     * `regime-underperformance` (table by regime)
     * `page-header` (heading identifies the dashboard)
   - Props: `data: VirtualFundLearningsData` + empty states.
   - Pure presentational; accepts either mocked (test) or live
     fetcher-driven data. No direct fetches this cycle.

2. **NEW** `frontend/src/components/VirtualFundLearnings.test.tsx`
   - 5 vitest tests:
     * learnings_page_landed (page header h2 "Virtual-Fund Learnings")
     * reconciliation_divergences_top10_rendered (table has 10
       rows when fed 15 divergences; sorted desc by abs drift)
     * kill_switch_trigger_distribution_rendered (4 buckets rendered
       with counts summing to total)
     * regime_underperformance_buckets_rendered (3 regime rows with
       negative-return formatting)
     * empty-state fallback

3. **NEW** `frontend/src/app/paper-trading/learnings/page.tsx`
   - Thin wrapper page mounting VirtualFundLearnings.
   - Exists so the "learnings_page_landed" criterion has a real URL.
   - Uses existing Sidebar + Page shell.

4. **MODIFY** `frontend/src/components/Sidebar.tsx`
   - Add "Learnings" nav item under Trading section (or similar)
   - NOT counted as a top-level route (nested under /paper-trading).
   Wait -- 4.7.1 capped top-level at 8. `/paper-trading/learnings`
   is a nested route, so it doesn't count against that budget.
   We still keep it reachable via a Trading-section sidebar link.

## Immutable success criteria (from masterplan)

1. `learnings_page_landed`
2. `reconciliation_divergences_top10_rendered`
3. `kill_switch_trigger_distribution_rendered`
4. `regime_underperformance_buckets_rendered`

## Verification (immutable)

    cd frontend && npm run test -- --filter=VirtualFundLearnings

Tests must pass 5/5. Additional self-imposed checks:
- `cd frontend && npm run build` still clean
- `grep -rn "/paper-trading/learnings" frontend/src/components/Sidebar.tsx`
  returns a nav entry (wiring actually present)
- Curl the new page in prod mode, confirm HTTP 200 + header text

## Anti-rubber-stamp rule

qa-evaluator: (a) check tests would fail if any section's
data-section marker was removed; (b) check top-10 sort is REAL
(feed 15, assert top 2 are the largest drifts, not hardcoded
indices); (c) nested route doesn't violate 4.7.1's <=8 top-level
budget.

## References

- backend/services/execution_router.py (reconciliation drift source)
- backend/services/paper_kill_switch.py (kill trigger events)
- phase-8.5 proposal for regime-detection data shape
- frontend/src/components/MfeMaeScatter.tsx (existing MFE/MAE
  surface -- not duplicated here)
