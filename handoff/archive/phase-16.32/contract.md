---
step: phase-16.32
title: ESLint no-restricted-imports for @phosphor-icons/react
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.32

## Research-gate summary

`handoff/current/phase-16.32-research-brief.md`. tier=simple, 6 in-full, 16 URLs, recency scan, gate_passed=true.

## Key research findings

1. **21 files currently violate the rule.** `grep -rln '@phosphor-icons/react' frontend/src/` returns 22 hits: 1 legitimate (`lib/icons.ts` itself) + 21 direct-import violators incl. Sidebar.tsx, SignalDashboard.tsx, MacroDashboard.tsx, RiskDashboard.tsx, etc.

2. **Recommended level: `"warn"` not `"error"`.** If rule lands as error, `npm run lint` immediately fails on all 21 files (would block builds + fail this cycle's verification). `"warn"` makes violations visible without breaking, and `no_lint_errors` criterion is satisfied because warnings don't cause non-zero exit.

3. **Exact config shape** (per researcher): 2 new flat-config blocks in `eslint.config.mjs`:
   - The `no-restricted-imports` rule with both `paths` (exact name) and `patterns` (subpath glob)
   - An override for `**/lib/icons.ts` setting the rule `off` (so the centralized barrel can still import from phosphor)

4. **Tree-shaking impact: zero**. `next.config.ts:10` already has `optimizePackageImports: ["@phosphor-icons/react"]`. The barrel re-export is safe.

## Hypothesis

Insert ~15-line ESLint block + ~5-line override block into `frontend/eslint.config.mjs`. After insertion, `npm run lint` shows ~21 NEW warnings (and the existing pre-rule warnings) but 0 errors. Verification command `npm run lint 2>&1 | tail -5` exits 0.

## Success Criteria (verbatim, immutable)

```
cd frontend && npm run lint 2>&1 | tail -5
```

- eslint_rule_present
- no_lint_errors
- no_test_regression

## Plan steps

1. Read `frontend/eslint.config.mjs` end-to-end (need exact insertion point)
2. Add `no-restricted-imports` rule (level: warn) + override block exempting `**/lib/icons.ts`
3. Run `cd frontend && npm run lint 2>&1 | tail -10` — confirm exit 0, warnings present, 0 errors
4. Run `cd frontend && npm run test` to confirm vitest unaffected
5. Spawn Q/A with explicit "is `warn` defensible vs `error` for this scope" judgment ask

## What Q/A must audit

1. ESLint config has the rule (grep `@phosphor-icons/react` in eslint.config.mjs)
2. icons.ts is exempted via override
3. `npm run lint` exits 0 (no errors)
4. Approximate warning count is ~21 + pre-existing
5. vitest still passes 7 files / 34 tests
6. Decision-rationale: is `warn` honest scope-discipline or buck-passing? (researcher recommends warn; user previously declined scope expansion)
7. Follow-up ticket filed for 21-file cleanup (NOT auto-closed by this cycle)
