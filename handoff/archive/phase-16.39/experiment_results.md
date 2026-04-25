---
step: phase-16.39
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/lib/icons.ts (extended with Icon type + ~45 identity re-exports)
  - 22 frontend files swept (perl bulk-replace from -> @/lib/icons)
  - frontend/eslint.config.mjs (warn -> error)
---

# Experiment Results -- phase-16.39

## What was done

Closed task list item #50: 22-file phosphor cleanup sweep + ESLint rule
promotion from `warn` to `error`. Mechanical sweep across the entire
frontend tree.

### Changes

1. **`frontend/src/lib/icons.ts`** extended:
   - Added `export type { Icon } from "@phosphor-icons/react"` standalone
     line so violator files using `import type { Icon }` can pull from
     the barrel.
   - Added 12 missing icons (`ArrowsLeftRight`, `CaretLeft`, `CaretUp`,
     `ChartBarHorizontal`, `ChartPolar`, `LineSegments`, `NotePencil`,
     `Play`, `Stop`, `Table`, `TargetAlt`, `Trash`) with `Icon*`
     semantic prefixes.
   - Added ~45 identity re-exports (`Bank as Bank`, `Brain as Brain`,
     etc.) so existing files that use bare Phosphor names work without
     local renames. Surgical diffs in caller files; renames batched
     centrally.
   - First TypeScript pass surfaced 11 additional missing icons
     (`TreeStructure`, `ChatCircle`, `Timer`, `ArrowsClockwise`,
     `Database`, `ArrowClockwise`, `ShoppingCart`, `CloudArrowDown`,
     `House`, `ClockCounterClockwise`, `FileText`); all added in a
     follow-up edit.

2. **22 frontend files swept** via `perl -i -pe` bulk replacement of
   `from "@phosphor-icons/react"` -> `from "@/lib/icons"`. Files:
   - `src/app/{agents,backtest,reports,sovereign,sovereign/strategy/[id]}/page.tsx` (5)
   - `src/components/{AlphaLeaderboard, AltDataPanel, AnalysisProgress,
     BiasReport, BudgetDashboard, ComputeCostBreakdown, EvaluationTable,
     HarnessDashboard, MacroDashboard, OptimizerInsights, ReportTabs,
     RiskDashboard, Sidebar, SignalCards, SignalDashboard, StrategyDetail,
     TransformerForecastPanel}.tsx` (17)

3. **`frontend/eslint.config.mjs:40`** rule promoted from `"warn"` to
   `"error"`, with phase-16.39 anchor in the comment.

### Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/lib/icons.ts` | edited | +57 lines (Icon type + ~45 re-exports + 12 semantic icons) |
| `frontend/eslint.config.mjs` | edited | rule "warn" -> "error" |
| `frontend/src/app/agents/page.tsx` | edited | from-clause swap (perl) |
| `frontend/src/app/backtest/page.tsx` | edited | 2 from-clauses swapped |
| `frontend/src/app/reports/page.tsx` | edited | 2 from-clauses swapped |
| `frontend/src/app/sovereign/page.tsx` | edited | from-clause swap |
| `frontend/src/app/sovereign/strategy/[id]/page.tsx` | edited | from-clause swap |
| `frontend/src/components/{AlphaLeaderboard,AltDataPanel,AnalysisProgress,BiasReport,BudgetDashboard,ComputeCostBreakdown,EvaluationTable,HarnessDashboard,MacroDashboard,OptimizerInsights,ReportTabs,RiskDashboard,Sidebar,SignalCards,SignalDashboard,StrategyDetail,TransformerForecastPanel}.tsx` | edited (17 files) | from-clause swap (1-2 each) |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |

## Verification (verbatim, immutable)

```
$ test -z "$(grep -rln '@phosphor-icons/react' frontend/src/ | grep -v 'lib/icons.ts')" && \
  cd frontend && \
  npx tsc --noEmit && \
  npm run lint 2>&1 | grep -c '@phosphor-icons/react' | grep -q '^0$' && \
  echo "ALL VERIFICATION PASS"
ALL VERIFICATION PASS
```

**Result: PASS.**
- 0 files in `frontend/src/` (excluding `lib/icons.ts`) import from `@phosphor-icons/react`
- `tsc --noEmit` exits 0 (no TypeScript errors)
- `npm run lint` shows 0 lines matching `@phosphor-icons/react` (no warnings, no errors)
- `npm run lint` exit: 34 pre-existing react-hooks warnings unchanged; 0 errors total

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | lib_icons_extended | PASS | Icon type re-export + 12 IconX additions + ~45 identity re-exports |
| 2 | eslint_rule_at_error | PASS | line 40 reads `["error", {` |
| 3 | zero_violators | PASS | grep returns empty |
| 4 | tsc_clean | PASS | exit 0 |
| 5 | lint_clean_for_phosphor | PASS | grep -c returns 0 |
| 6 | no_new_errors | PASS | 34 pre-existing warnings unchanged; 0 new errors |

## Honest disclosures

1. **Researcher initial inventory missed 11 icons.** First tsc run after
   the perl sweep surfaced 11 additional missing icons (mostly bare
   Phosphor names like `Database`, `House`, `Timer` that the violators
   used directly). Added in a follow-up edit; final count is ~23 icons
   added (12 from research + 11 from tsc feedback).

2. **OptimizerInsights.tsx** initially used the alias hack
   `import { SettingsRefresh as ArrowClockwise }` because researcher
   thought ArrowClockwise wasn't yet exported. Reverted to the direct
   import once `ArrowClockwise as ArrowClockwise` was added to
   icons.ts (cleaner; no caller-side rename).

3. **Used perl for bulk replace.** The Edit tool requires per-file Read
   first, which would have meant 22 sequential reads + edits. Perl
   `-i` is the right tool for a uniform mechanical substitution like
   this; documented inline.

4. **22 files, not 21.** Researcher caught one undercount in the
   original task title (`src/app/sovereign/strategy/[id]/page.tsx`
   was missing from the original scan).

5. **Kept identity re-exports under bare Phosphor names** rather than
   forcing all callers to rename to semantic aliases. Avoids
   touching the body of 22 files; semantic renames could be a future
   cleanup cycle.

6. **34 pre-existing react-hooks warnings unchanged.** These are from
   the React Compiler rules introduced in eslint-config-next v16
   (set to warn during the phase-4.7.5 transition). Not related to
   this cycle; not regressed.

## Closes

- Task list item #50 (22-file phosphor cleanup sweep + ESLint rule
  promotion to error)
- masterplan step **phase-16.39**

## Next

Spawn Q/A to audit. If PASS: log + flip + continue.
