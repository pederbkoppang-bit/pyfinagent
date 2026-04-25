---
step: phase-16.17
title: Frontend correctness re-verification (vitest+tsc+build+lint)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.17

## Research-gate summary

Source: `handoff/current/phase-16.17-research-brief.md`. JSON envelope:
```json
{"tier":"simple","external_sources_read_in_full":5,"snippet_only_sources":10,"urls_collected":15,"recency_scan_performed":true,"internal_files_inspected":13,"report_md":"handoff/current/phase-16.17-research-brief.md","gate_passed":true}
```
Floor met: 5/5, 15/10. Recency present (Vitest 4.x, Next.js 15, ESLint 9, React 19).

Watch items from researcher: ESLint React-Compiler rules are `warn` not `error` so build passes; `AutoresearchLeaderboard.test.tsx` uses fake timers — confirm cleanup; `tsc --noEmit` is safe before `next build` because `.next/types` glob only matches existing files.

## Hypothesis

Frontend builds cleanly today. 7 vitest component test files all PASS (4 of them re-verified earlier this session as part of phase-10.5 closure). `tsc --noEmit` clean (verified after the 10.5.7 hero embed). `next build` succeeds. `eslint` clean.

## Success Criteria (verbatim from masterplan)

Verification command (immutable):
```
cd frontend && npx vitest run && npx tsc --noEmit && npm run build && npm run lint
```

- vitest_all_pass
- tsc_clean
- next_build_exit_0
- eslint_clean

## Plan steps

1. Run the 4-stage chained command verbatim, capturing each stage's stdout + exit code
2. If any stage fails, isolate which file caused the regression
3. Spawn Q/A

## What Q/A must audit

1. Each of 4 stages independently re-verified (don't trust Main's stdout)
2. Vitest count matches expectations (~7 test files; the 4 from 10.5 known PASS plus HarnessSprintTile, AutoresearchLeaderboard, VirtualFundLearnings)
3. No code changes claimed by Main (read-only verification)
4. ESLint warning count noted (warnings allowed by config but should be tracked)

## References

- `handoff/current/phase-16.17-research-brief.md`
- `frontend/package.json`, `vitest.config.ts`, `tsconfig.json`, `eslint.config.mjs`
