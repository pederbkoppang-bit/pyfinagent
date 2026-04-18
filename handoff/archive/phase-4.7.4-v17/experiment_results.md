# Experiment Results -- Cycle 76 / phase-4.7 step 4.7.7

Step: 4.7.7 Virtual-fund learnings dashboard

## What was generated

1. **NEW** `frontend/src/components/VirtualFundLearnings.tsx`
   Pure presentational component with four data-section regions:
   - page-header -- identifies the dashboard
   - reconciliation-divergences -- top-10 sorted by abs drift desc
   - kill-switch-distribution -- segmented bar + total + list
   - regime-underperformance -- per-regime table with negative-
     return rose styling

2. **NEW** `frontend/src/components/VirtualFundLearnings.test.tsx`
   5 vitest tests with discriminating assertions:
   - page header renders Virtual-Fund Learnings text
   - top-10 sort: feeds 15 divergences, asserts rows.length==10,
     top row is max-abs drift from input, and 10th abs >= all
     excluded
   - kill-switch: 4 buckets rendered, total equals sum, reason
     "daily_loss_pct" present
   - regime: bear row has text-rose-400 AND -4.40% text
   - empty states render when data prop absent

3. **NEW** `frontend/src/app/paper-trading/learnings/page.tsx`
   Thin wrapper page that mounts the component with the Sidebar
   shell. Route: `/paper-trading/learnings` (NESTED, does NOT
   affect 4.7.1's <=8 top-level route budget).

4. **MODIFY** `frontend/src/components/Sidebar.tsx`
   Added "Learnings" nav item under the Trading section, pointing
   to `/paper-trading/learnings`. Icon: `NavPerformance`.

## Verification (verbatim, immutable)

    $ cd frontend && npm run test -- --filter=VirtualFundLearnings
      Tests  5 passed (5)
      exit=0

## Real-browser exercise

    $ LIGHTHOUSE_SKIP_AUTH=1 PORT=3000 npm run start
    $ curl -sL http://localhost:3000/paper-trading/learnings | \
        grep -oE 'data-section="(page-header|reconciliation-divergences|kill-switch-distribution|regime-underperformance)"'
    data-section="page-header"
    data-section="reconciliation-divergences"
    data-section="kill-switch-distribution"
    data-section="regime-underperformance"

All 4 markers + page header text render in the live server response.

## Success criteria

| Criterion | Result |
|-----------|--------|
| learnings_page_landed | PASS (page renders at /paper-trading/learnings) |
| reconciliation_divergences_top10_rendered | PASS (length==10 + sort verified) |
| kill_switch_trigger_distribution_rendered | PASS (buckets + sum verified) |
| regime_underperformance_buckets_rendered | PASS (rose styling + text) |

## Route-count invariant

    $ python scripts/audit/route_count.py
    top_level_routes: 8
    paths: [/, /agents, /backtest, /paper-trading, /performance,
            /reports, /settings, /signals]

/paper-trading/learnings is nested, not top-level. 4.7.1's budget
holds.

## Known limitations (non-blocking, tracked follow-up)

- Component accepts `data` as a prop; no live backend fetcher yet.
  A follow-up adds `/api/paper-trading/learnings` returning:
    {reconciliation_divergences, kill_switch_triggers, regime_buckets}
  wired from backend/services/execution_router.py +
  paper_kill_switch.py + phase-8.5 regime_detection. Queued.
- MFE/MAE visualization is handled by the existing
  MfeMaeScatter.tsx on /paper-trading; not duplicated here.
- Layout uses a single scrollable container rather than the
  frontend-layout.md §1 two-zone pattern (fixed header + scrollable
  content). qa-evaluator flagged as a style nit, not a violation.
  Tracked for the 4.7.x consistency pass.
