# Phase 2.6.1 Experiment Results — Harness Dashboard

## What Was Built

### Backend — 5 API Endpoints
All under `/api/backtest/harness/`:
- `GET /harness/log` — Parses `harness_log.md` into structured cycles (hypothesis, verdict, scores, decision)
- `GET /harness/critique` — Returns `evaluator_critique.md` as raw markdown
- `GET /harness/contract` — Returns current `contract.md`
- `GET /harness/validation` — Returns both `validation_results.json` and `subperiod_validation_results.json`
- `GET /harness/criteria` — Returns `evaluator_criteria.md`
- All endpoints return empty/null gracefully if files don't exist

### Frontend — HarnessDashboard Component
New component at `frontend/src/components/HarnessDashboard.tsx`:
- **Current Contract** — Shows active contract from `contract.md` in a markdown viewer
- **Sub-Period Validation** — Table showing Sharpe, DSR, Return, MaxDD, Trades per period (A, B, C)
  - Color-coded Sharpe: green ≥0.8, amber ≥0.5, red <0.5
- **Evaluator Critique** — Renders latest critique in a scrollable markdown viewer
- **Harness Cycles** — Collapsible accordion showing each cycle with:
  - Verdict badges (PASS=green, FAIL=red, CONDITIONAL=amber)
  - Score bars (Statistical, Robustness, Simplicity, Reality Gap) with color coding
  - Hypothesis, generator output, decision details
- Empty state with Phosphor icon when no cycles exist

### Integration
- New "Harness" tab added to backtest page (6th tab, ClockCounterClockwise icon)
- Types: `HarnessCycle`, `HarnessValidation` in types.ts
- API functions: 5 new fetch functions in api.ts
- All fetches use `Promise.all` with `.catch()` for graceful degradation
- Follows frontend conventions: Phosphor icons, BentoCard pattern, scrollbar-thin, no emoji

## Verified
- Backend: All 5 endpoints return valid JSON
- Frontend: Build succeeds, no TypeScript errors
- Both services running on ports 8000 + 3000
