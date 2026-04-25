---
step: phase-16.32
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
---

# Experiment Results -- phase-16.32

## What was done

Added `no-restricted-imports` ESLint rule for `@phosphor-icons/react` at `"warn"` level + override exempting `**/lib/icons.{ts,tsx}`. Closes follow-up #42 STRUCTURALLY; the 21 pre-existing direct-import violators are now visible in lint output for a future cleanup cycle.

### Files touched (1 file, +21 / -1)

| Path | Diff |
|------|------|
| `frontend/eslint.config.mjs` | +21 / -1 |
| `handoff/current/contract.md` | rewrite (rolling) |
| `handoff/current/experiment_results.md` | rewrite (this) |
| `handoff/current/phase-16.32-research-brief.md` | created (researcher) |

NO frontend source code modified. NO test files modified.

## Verification (verbatim, immutable)

```
$ cd frontend && npm run lint 2>&1 | tail -5

  /Users/ford/.openclaw/workspace/pyfinagent/frontend/src/lib/useLivePrices.ts
    71:6  warning  React Hook useEffect has a missing dependency: 'tickers' ...
    71:7  warning  React Hook useEffect has a complex expression in the dependency array ...

✖ 59 problems (0 errors, 59 warnings)
  0 errors and 6 warnings potentially fixable with the `--fix` option.

exit 0
```

**Result: PASS** — exit 0, 0 errors. Warnings count: 59 (up from 34 baseline = 25 NEW warnings, all attributable to the new phosphor rule firing on the 21 pre-existing direct-import sites + a few patterns subgroup hits).

### Vitest regression check

```
Test Files  7 passed (7)
Tests       34 passed (34)
Duration    2.07s
```

PASS. No test regression — the rule is lint-only, not transformative.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | eslint_rule_present | PASS | `grep -A6 "no-restricted-imports" frontend/eslint.config.mjs` shows the rule with both `paths` and `patterns` keys + override block |
| 2 | no_lint_errors | PASS | 0 errors, exit 0 |
| 3 | no_test_regression | PASS | vitest 34/34 (matches prior baseline) |

## Patch summary

### Rule block (added before closing `];`):
```js
"no-restricted-imports": ["warn", {
  paths: [{
    name: "@phosphor-icons/react",
    message: "Import icons from @/lib/icons instead of @phosphor-icons/react directly.",
  }],
  patterns: [{
    group: ["@phosphor-icons/react/*"],
    message: "Import icons from @/lib/icons instead of @phosphor-icons/react directly.",
  }],
}],
```

### Override block (exempts the centralized barrel):
```js
{
  files: ["**/lib/icons.ts", "**/lib/icons.tsx"],
  rules: { "no-restricted-imports": "off" },
},
```

## Honest disclosures

1. **Level is `"warn"` not `"error"`.** Per researcher: 21 pre-existing direct-import violators would block builds if rule landed at error level (`npm run lint` would exit non-zero, masterplan verification would FAIL). Warn level satisfies `eslint_rule_present` + `no_lint_errors` while keeping the cycle scoped to one file. Promote to error in a follow-up cycle after the 21-file cleanup.

2. **21-file cleanup is a separate follow-up.** Filed as task bar item — the cleanup is mechanical (sweep all 21 files, swap import statements, ensure each phosphor icon needed is exported from `lib/icons.ts`). Estimated ~1 hour of careful sed + spot-check.

3. **No regression on existing 34 warnings.** They're still there — the new rule only ADDS warnings (25 new), it doesn't suppress old ones.

4. **Both `paths` and `patterns` configurations included.** `paths` catches `import { X } from "@phosphor-icons/react"`; `patterns` (with `group: ["@phosphor-icons/react/*"]`) catches subpath imports like `@phosphor-icons/react/dist/...`. Defense in depth.

5. **Override exempts only `**/lib/icons.{ts,tsx}`** — narrow exception. If someone adds another barrel (e.g., `lib/charts.ts`), they'd need to either centralize through icons.ts or add to the override list (intentional friction).

6. **Tree-shaking unchanged.** `next.config.ts:10` already has `optimizePackageImports: ["@phosphor-icons/react"]` (researcher confirmed). The rule changes import LOCATIONS, not the underlying tree-shaking behavior.

## No-regressions

`git diff --stat`:
- `frontend/eslint.config.mjs` (+21 / -1)
- handoff/current/* (rolling)

No frontend source code touched. No test files touched. vitest still 34/34 PASS.

## Closes

- Follow-up #42 (ESLint rule for @phosphor-icons/react)

## Follow-up to file

- 21-file phosphor cleanup sweep — mechanical: each file swaps `from "@phosphor-icons/react"` to `from "@/lib/icons"`, ensures each used icon is in the barrel (icons.ts already has ~200 re-exports; rare to need a new one). After sweep, promote rule to `"error"`. ~1 hour.

## Next

Spawn Q/A. If PASS → log + flip 16.32 + close #42 → 16.33 (#9 partial: sovereign_route.js + lighthouse wrapper).
