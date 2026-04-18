# Contract -- Cycle 73 / phase-4.7 step 4.7.4

Step: 4.7.4 Autoresearch Run view: candidate leaderboard (DSR/PBO/P&L)

## Hypothesis

Backend already exposes experiments via
`GET /api/backtest/optimize/experiments` with `dsr`, `param_changed`,
`metric_after`. The gaps: (a) PBO is NOT persisted per experiment;
(b) no "realized P&L if promoted" projection; (c) no
`AutoresearchLeaderboard` React component; (d) no test framework.

Scope for this step -- we ship the UI primitive + the test harness.
Backend PBO-persist + the full promotion pipeline live in phase-8.5
(Karpathy autoresearch integration). For 4.7.4 we:

- compute PBO per displayed candidate client-side from the same
  rolling returns-window array the backend returns (thin backend
  patch to include `returns_per_window: number[][]` on experiments)
- derive `realized_pnl_if_promoted` as `metric_after * starting_capital`
  projected over the same paper-trading horizon (transparent formula
  documented in the component's tooltip)

## Scope

Files created / modified:

1. **NEW** `frontend/vitest.config.ts` -- jsdom env + React plugin.
2. **NEW** `frontend/vitest.setup.ts` -- @testing-library/jest-dom.
3. **NEW** `frontend/scripts/run-test.mjs` -- wraps `vitest run`,
   translates masterplan `--filter=X` CLI arg to the vitest
   positional file-pattern.
4. **MODIFY** `frontend/package.json`:
   - add devDeps: vitest, @vitejs/plugin-react, @testing-library/
     react, @testing-library/jest-dom, jsdom
   - add `"test": "node scripts/run-test.mjs"` script
5. **NEW** `frontend/src/components/AutoresearchLeaderboard.tsx`
   - sortable table with columns: Rank, Param Change, DSR, PBO,
     Realized P&L (if promoted), Status
   - data-col attributes so DOM queries are stable
   - 5 s polling via `useEffect + setInterval` (well under 10 s)
   - accepts `candidates: Candidate[]` prop so the unit test can
     render without backend round-trip
   - empty state + loading state + error state
6. **NEW** `frontend/src/components/AutoresearchLeaderboard.test.tsx`
   - 5 tests:
     * renders DSR column header
     * renders PBO column header
     * renders Realized P&L column header
     * refresh interval <= 10s (fake timers)
     * empty state renders when candidates=[]
7. **MODIFY** `frontend/src/app/backtest/page.tsx`:
   - render `<AutoresearchLeaderboard>` inside the optimizer tab,
     above the existing experiment log table.

## Immutable success criteria (from masterplan)

1. `leaderboard_refresh_le_10s`
2. `dsr_column_present`
3. `pbo_column_present`
4. `realized_pnl_if_promoted`

## Verification (immutable, from masterplan.json)

    cd frontend && npm run test -- --filter=AutoresearchLeaderboard

Must exit 0 with all 5 tests green.

## Additional rigor checks (self-imposed per user feedback 2026-04-18)

- **Real browser exercise**: after tests pass, start prod server
  (`LIGHTHOUSE_SKIP_AUTH=1 PORT=3000 npm run start`), curl
  `/backtest`, confirm the page returns 200 and body contains
  `Autoresearch Leaderboard` or an AGENT_NODES marker. Document
  the raw curl output in experiment_results.md.
- **qa-evaluator gets full scope**: prompt explicitly asks for a
  CONDITIONAL if the test suite is too narrow / wouldn't catch
  a real bug (e.g. a renamed column header).
- **No second-opinion shopping**: if qa returns CONDITIONAL, I fix
  the underlying issue and SendMessage back to the same agent, not
  a fresh spawn.

## References

- backend/backtest/quant_optimizer.py (experiments source)
- backend/api/backtest.py (/api/backtest/optimize/experiments)
- backend/backtest/analytics.py (compute_pbo, compute_dsr)
- .claude/rules/frontend.md (Glass Box; no emoji)
- .claude/rules/frontend-layout.md §4 (metric grids)
- https://vitest.dev/guide/cli
