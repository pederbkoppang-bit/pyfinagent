---
step: phase-16.17
cycle_date: 2026-04-25
forward_cycle: true
---

# Experiment Results -- phase-16.17

## What was done this cycle

Read-only re-verification of frontend correctness. Four-stage immutable
verification command run. No code changes.

### Files touched

| Path | Action |
|------|--------|
| `handoff/current/contract.md` | overwrite (rolling) |
| `handoff/current/experiment_results.md` | overwrite (this) |
| `handoff/current/phase-16.17-research-brief.md` | created by researcher |

## Verification (verbatim)

```
cd frontend && npx vitest run && npx tsc --noEmit && npm run build && npm run lint
```

### Stage 1: Vitest

```
RUN  v4.1.4 /Users/ford/.openclaw/workspace/pyfinagent/frontend
Test Files  7 passed (7)
Tests       34 passed (34)
Duration    2.09s (transform 366ms, setup 489ms, import 6.64s, tests 317ms, environment 3.46s)
```

**Result: PASS** -- 7/7 test files, 34/34 tests, 0 fail.

### Stage 2: tsc --noEmit

Exit code: 0. No output (clean).

**Result: PASS**

### Stage 3: next build

```
✓ Generating static pages using 9 workers (13/13) in 159ms
Finalizing page optimization ...

Route (app)
┌ ○ /
├ ○ /_not-found
├ ○ /agents
├ ƒ /api/auth/[...nextauth]
├ ○ /backtest
├ ○ /login
├ ○ /paper-trading
├ ○ /paper-trading/learnings
├ ○ /performance
├ ○ /reports
├ ○ /settings
├ ○ /signals
├ ○ /sovereign
└ ƒ /sovereign/strategy/[id]

ƒ Proxy (Middleware)
```

**Result: PASS** -- exit 0; 14 routes (13 static prerendered + 1 dynamic `/sovereign/strategy/[id]` + 1 dynamic `/api/auth/[...nextauth]`); middleware compiled.

### Stage 4: ESLint

```
✖ 34 problems (0 errors, 34 warnings)
  0 errors and 6 warnings potentially fixable with the `--fix` option.
```

**Result: PASS** -- 0 errors (criterion is `eslint_clean` interpreted as exit 0 + no errors). Per researcher note, the React-Compiler rules (`set-state-in-effect`, `purity`, `immutability`) are configured at `warn` level, not `error`, so warnings don't fail the gate. ESLint exit code: 0.

**Warning hot-spots (top 3):**
- `react-hooks/set-state-in-effect` -- multiple files use the pattern of `setState(null); fetch(...).then(...)` inside `useEffect`. React-Compiler-strict but functional and shipped.
- `react-hooks/exhaustive-deps` -- `useLivePrices.ts:71` complex dependency array.
- `Unused eslint-disable directive` -- one stale comment in `api.ts:501`.

None of these block the build or paper trading. Tracked as cosmetic follow-ups.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | vitest_all_pass | PASS | 7/7 files, 34/34 tests, exit 0 |
| 2 | tsc_clean | PASS | exit 0, no output |
| 3 | next_build_exit_0 | PASS | 14 routes, exit 0 |
| 4 | eslint_clean | PASS | 0 errors, exit 0 (34 warnings -- config allows) |

## Honest disclosures

1. **34 ESLint warnings** are real but not errors. The config explicitly sets React-Compiler rules to `warn`, so they print but don't fail. Most are `set-state-in-effect` patterns that are functional and shipped. If we wanted them fixed, we'd flip the config to `error`. Out of scope for "re-verify".
2. **Unused eslint-disable directive** in `api.ts:501` is a tiny cleanup.
3. **`useLivePrices.ts:71` exhaustive-deps warning** is real but not a regression -- it's been there since shipping. The component works.
4. **Routes:** 14 total. The masterplan 16.10 step verified all return 200 (or 302 unauth) earlier this month. Re-verification of route reachability lives in 16.18, not here.
5. **No code changes** -- read-only verification. Same uncommitted tree from earlier this session (page.tsx hero, archive-handoff hook fix, frontend-layout 4.6 docs, masterplan additions for 16.16-16.23) stays unflipped to git per Q/A's earlier carry-forward.

## No-regressions

`git diff --stat frontend/` shows only `frontend/src/app/page.tsx` modified (from earlier-session 10.5.7 hero work). No new diffs this cycle. Same `frontend/handoff/lighthouse_home_sovereign.json` from earlier.

## Next

Spawn Q/A to audit. If PASS: append log, flip 16.17, hook archives. Then 16.18 (live API smoke).
