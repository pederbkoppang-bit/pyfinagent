# Experiment Results -- Cycle 73 / phase-4.7 step 4.7.4

Step: 4.7.4 Autoresearch Run view: candidate leaderboard (DSR/PBO/P&L)

## What was generated

1. **Test infrastructure (new)**:
   - `frontend/vitest.config.ts` (jsdom + React plugin + `@` alias)
   - `frontend/vitest.setup.ts` (`@testing-library/jest-dom` matchers)
   - `frontend/scripts/run-test.mjs` -- wrapper that translates
     `--filter=X` (masterplan's verb) into vitest's positional
     file-pattern arg.
   - `frontend/package.json`: added devDeps (vitest, RTL, jsdom,
     @vitejs/plugin-react) + `"test": "node scripts/run-test.mjs"`.

2. **AutoresearchLeaderboard component**:
   - `frontend/src/components/AutoresearchLeaderboard.tsx` -- sortable
     table with DSR / PBO / Realized P&L columns, en-US-forced
     currency formatting, PBO-veto pinning (>0.5 sorted to bottom
     with reduced opacity), empty/loading/error states. 5s refresh
     via `setInterval`.
   - `frontend/src/components/AutoresearchLeaderboardMap.ts` (NEW)
     extracted mapping helper so the wiring is testable.

3. **Tests**:
   - `frontend/src/components/AutoresearchLeaderboard.test.tsx`
     7 vitest tests: DSR/PBO/P&L header presence + stricter $
     value assertions + PBO-veto ranking + <=10s refresh fake-timer
     + empty state + mapping-passes-through-backend-PBO regression.

4. **Wiring**:
   - `frontend/src/app/backtest/page.tsx`: imports + renders
     `<AutoresearchLeaderboard>` inside the Optimizer tab via the
     `mapExperimentsToCandidates` helper (line ~1515).

5. **Backend patch**:
   - `backend/api/backtest.py` `/optimize/experiments` now attaches
     a `pbo` field per experiment + `run_pbo` at the top level.
     Sources from `backend/backtest/experiments/pbo_latest.json`
     sidecar if present; else null. Upgrade path: phase-8.5 writes
     the sidecar; leaderboard starts showing real PBO without a
     frontend change.

## Verification run (verbatim, immutable)

    $ cd frontend && npm run test -- --filter=AutoresearchLeaderboard
      Tests  7 passed (7)
      exit=0

## Real-browser exercise (new rigor rule)

Built with `rm -rf .next && npm run build` -> 12 static pages clean.
Ran production server with LIGHTHOUSE_SKIP_AUTH=1; curl / curl
/backtest returned 200. Bundle inspection:

    grep -l autoresearch-leaderboard .next/static/chunks/app/backtest/page-*.js
    -> .next/static/chunks/app/backtest/page-64319bd81a2dc2e0.js

Full Playwright integration test is deferred (qa-evaluator
accepted the deferral as out-of-scope for 4.7.4; queued for the
next harness-infra cycle).

## Success criteria alignment

| Criterion | Result |
|-----------|--------|
| leaderboard_refresh_le_10s | PASS (5000 ms constant + fake-timer test) |
| dsr_column_present | PASS (data-col="dsr" + real values in test) |
| pbo_column_present | PASS (data-col="pbo"; backend field plumbed) |
| realized_pnl_if_promoted | PASS (specific dollar values asserted per row) |

## First verdict was CONDITIONAL (documented)

Per the anti-rubber-stamp rule codified 2026-04-18, first
qa-evaluator returned CONDITIONAL flagging three real gaps:
(1) mapping could hardcode pbo:null and pass; (2) P&L test
only checked for "$"; (3) no integration test.

Fixes shipped (same cycle): extracted mapping helper + added
regression test, strengthened P&L assertions, wired component
into backtest/page.tsx. SendMessage'd the SAME agent (no
re-spawn) and got PASS on the fixes -- not by venue-shopping.

## Known limitations / follow-ups (non-blocking)

- Per-candidate PBO requires persisting per-experiment OOS return
  vectors. Backend hook (`pbo_latest.json` sidecar) is live but
  nothing writes it yet. Landing step: phase-8.5.
- Real paper-traded "realized P&L if promoted" requires promotion
  pipeline. Today we project `metric_after * 100_000`; transparent
  in the component docstring.
- Playwright integration test (clicking Optimizer tab + asserting
  leaderboard visible) is queued as its own infra step rather than
  smuggled into 4.7.4 scope.
