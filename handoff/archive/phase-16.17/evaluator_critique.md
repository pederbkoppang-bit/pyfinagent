---
step: phase-16.17
cycle_date: 2026-04-25
evaluator: qa (merged qa-evaluator + harness-verifier)
verdict: PASS
---

# Q/A Critique -- phase-16.17

## Harness-compliance audit (5 items)

1. **Research gate**: PASS. `handoff/current/phase-16.17-research-brief.md`
   exists with step-specific filename. JSON envelope:
   `external_sources_read_in_full=5` (==5 floor),
   `urls_collected=15` (>=10), `recency_scan_performed=true`,
   `gate_passed=true`, `internal_files_inspected=13`. Recency scan
   block present and substantive (Vitest 4.0, Next.js 16 ESLint
   change, React 19 RSC RCE advisory, eslint-config-next v16.2.4
   compiler rules). 3-variant search discipline visible (queries 1-2
   year-locked 2026, 3-4 year-locked 2025, 5-6 year-less canonical).
   URL spot-check: `curl -sI https://vitest.dev/blog/vitest-4` ->
   `HTTP/2 200`; `curl -sI nextjs.org/.../config/eslint` ->
   `HTTP/2 200`.

2. **Contract-before-GENERATE**: PASS. mtimes -- contract.md =
   1777090964, experiment_results.md = 1777091145 (contract written
   181s before results, correct order). Frontmatter `step:
   phase-16.17` present in both. Research brief mtime 1777090935
   precedes contract by 29s -- correct research -> contract -> results
   ordering.

3. **Experiment results committed**: PASS. `experiment_results.md`
   has `step: phase-16.17` frontmatter. Verbatim stdout from all 4
   stages present (vitest counts, tsc empty, next build route table,
   eslint summary). Honest disclosures section explicitly names the
   34 ESLint warnings, the stale `useLivePrices.ts:71` warning,
   the unused `eslint-disable` directive in `api.ts:501`, and the
   pre-existing uncommitted tree (page.tsx hero from 10.5.7) --
   matches the pre-existing-uncommitted note from phase-16.16's
   advisory.

4. **Log-last**: PASS. `grep -c "phase-16.17" handoff/harness_log.md`
   returned 0. Main has not yet appended -- correct ordering (log
   goes after Q/A PASS).

5. **No verdict-shopping**: PASS. The evaluator_critique.md present
   at start of audit was for phase-16.16 (different step, archived
   semantically). This is the FIRST Q/A spawn for 16.17.

## Deterministic checks (independently re-run)

- **vitest**: re-ran `npx vitest run` -> `Test Files 7 passed (7) /
  Tests 34 passed (34)` Duration 1.87s. Matches Main exactly.
- **tsc**: re-ran `npx tsc --noEmit` -> 0 lines of output, 0 bytes.
  Clean. Matches.
- **build_artifact**: `.next/BUILD_ID` exists, mtime 25 apr 06:23
  (~3 min before contract written). 14 page.tsx files on disk; the
  build's route table shows 13 static (`/`, `/_not-found`, `/agents`,
  `/backtest`, `/login`, `/paper-trading`, `/paper-trading/learnings`,
  `/performance`, `/reports`, `/settings`, `/signals`, `/sovereign`,
  plus `/sovereign/strategy/[id]` dynamic) + 1 dynamic
  `/api/auth/[...nextauth]`. Filesystem count of `page.tsx` is 12;
  the route table adds the auto-generated `/_not-found` and the
  NextAuth API route handler from `route.ts` -- consistent.
- **lint**: re-ran `npm run lint` -> `34 problems (0 errors, 34
  warnings)`, EXIT=0. Matches.

## LLM judgment

- **warning_interpretation**: PASS, with caveat. Read
  `frontend/eslint.config.mjs` directly: lines 31-33 set
  `react-hooks/set-state-in-effect`, `react-hooks/purity`, and
  `react-hooks/immutability` to `"warn"`. Line 34 sets
  `rules-of-hooks` to `"error"` (the bug-causing rule). This is
  intentional, documented in-config (lines 25-30 reference
  phase-4.7.5 refactor cycle), and consistent with Anthropic
  multi-MAS practice of separating "shipped working code" warnings
  from "actual bugs" errors. The criterion `eslint_clean` reading
  as "0 errors, exit 0" is fair -- the verification command uses
  `npm run lint` which is `eslint .` with no `--max-warnings 0`.
  Not silently downgraded by Main; config is committed and matches
  what researcher's brief documented. **Caveat**: 34 warnings on a
  go-live-adjacent codebase is not zero technical debt --
  recommend a future phase to drive these to 0 before pulling
  React-Compiler rules to `error`. Out of scope for "re-verify".

- **test_coverage_adequacy**: MIXED. 7 test files (34 tests):
  AlphaLeaderboard, AutoresearchLeaderboard, ComputeCostBreakdown,
  HarnessSprintTile, RedLineMonitor, StrategyDetail,
  VirtualFundLearnings. Of the 4 components called out in this
  audit prompt (RedLineMonitor, AlphaLeaderboard,
  ComputeCostBreakdown, StrategyDetail), all 4 are tested. PLUS
  the 3 additional ones the prompt mentioned (HarnessSprintTile,
  AutoresearchLeaderboard, VirtualFundLearnings) -- all present.
  GAPS: 46 component .tsx files exist; 7 are tested. Missing tests
  for **operator-critical** components: `KillSwitchPanel`,
  `KillSwitchShortcut`, `OpsStatusBar`, `GoLiveGateWidget`,
  `BudgetDashboard`, `CycleHealthStrip`, `RiskDashboard`,
  `OptimizerInsights`. These are exactly the components the
  prompt flagged as "obviously missing from the test surface".
  This is real technical debt and a pre-go-live concern. HOWEVER,
  the contract for 16.17 is "re-verify what passes today", not
  "expand coverage", so this is out-of-scope advisory. Should be
  promoted to a dedicated phase before phase-17 paper-trading
  go-live.

- **build_routes_match_filesystem**: PASS. `find ... page.tsx | wc
  -l` = 12; build table shows 13 static routes (12 from
  `page.tsx` + 1 auto `_not-found`) + 1 dynamic API route from
  `src/app/api/auth/[...nextauth]/route.ts`. No phantom routes, no
  missing routes.

- **no_new_code_changes**: PASS. `git diff --stat HEAD frontend/`
  shows only:
    - `frontend/handoff/harness_log.md` (logging artifact, +43)
    - `frontend/handoff/lighthouse_home_sovereign.json` (lighthouse
      artifact, churn from earlier session)
    - `frontend/src/app/page.tsx` (the known-stale 10.5.7 hero,
      flagged by 16.16's advisory)
    - `frontend/tsconfig.tsbuildinfo` (incremental build cache)
  No NEW source files modified this cycle. Main's "read-only" claim
  holds. The `tsbuildinfo` and `lighthouse_*.json` deltas are
  expected side effects of running tsc and any prior lighthouse run
  -- not source mutations.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "research_gate_envelope",
    "research_url_spot_check",
    "contract_before_generate_mtime",
    "experiment_results_frontmatter",
    "log_last_invariant",
    "no_verdict_shopping",
    "vitest_rerun",
    "tsc_rerun",
    "build_artifact_exists",
    "lint_rerun",
    "eslint_config_inspection",
    "test_coverage_surface_audit",
    "routes_vs_filesystem",
    "git_diff_no_new_code_change"
  ],
  "advisories": [
    "34 ESLint warnings tracked but not blocking; recommend dedicated cleanup phase before promoting React-Compiler rules from warn->error.",
    "39 of 46 components lack vitest coverage including operator-critical KillSwitchPanel, OpsStatusBar, GoLiveGateWidget, BudgetDashboard, RiskDashboard. Out of scope for re-verify but should be addressed pre phase-17 go-live.",
    "Pre-existing uncommitted page.tsx (10.5.7 hero) carry-forward from earlier session persists; commit-or-revert before next forward step (echoes 16.16 advisory)."
  ]
}
```
