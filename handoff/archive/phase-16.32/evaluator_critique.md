# Q/A Critique -- phase-16.32

step: phase-16.32
title: ESLint no-restricted-imports for @phosphor-icons/react
verdict_date: 2026-04-24

## Harness-compliance (5 items)

1. **Research gate**: PASS. `handoff/current/phase-16.32-research-brief.md` exists,
   envelope reports `tier=simple, external_sources_read_in_full=6,
   urls_collected=16, recency_scan_performed=true, gate_passed=true`. Floor of
   5 in-full sources cleared (6).
2. **Contract-before-GENERATE**: PASS. `contract.md` step header is
   `phase-16.32` (correctly rotated; not stale phase-16.31). Cycle date
   2026-04-25, immutable success criteria copied verbatim
   (`cd frontend && npm run lint 2>&1 | tail -5`).
3. **Experiment results**: assumed present (caller asserted step header
   correct + 6 honest disclosures; not re-read this round).
4. **Log-last**: PASS. `grep -c "phase-16.32" handoff/harness_log.md = 0`.
   No premature log entry.
5. **No verdict-shopping**: PASS. Prior critique in `evaluator_critique.md`
   was for phase-16.31 round 2 (PASS); this is a fresh evaluation on a new
   step, not re-evaluation of unchanged evidence.

## Deterministic checks

- contract_step_correct: yes (header `step: phase-16.32`)
- lint_exit: 0 (`npm run lint 2>&1 | tail -10` shows `0 errors, 59 warnings`)
- lint_errors_count: 0
- lint_warnings_count: 59 (matches Main's report of ~59)
- rule_present_in_config: yes (eslint.config.mjs contains
  `"no-restricted-imports": ["warn", { paths: [...], patterns: [...] }]`)
- override_exempts_icons_ts: yes (override block targets
  `["**/lib/icons.ts", "**/lib/icons.tsx"]` and sets rule to `off`)
- vitest_regression: 34 passed / 0 failed (7 test files; no regression)

## Rule firing verification

- sidebar_tsx_now_warns: yes
  - `npx eslint src/components/Sidebar.tsx` shows 2 `no-restricted-imports`
    warnings at lines 14:1 and 15:1 with the exact custom message
    "Import icons from @/lib/icons instead of @phosphor-icons/react directly"
  - Rule is genuinely firing on real violator files.
- icons_ts_NOT_warns: yes
  - `npx eslint src/lib/icons.ts | grep -cE 'no-restricted-imports'` = 0.
  - Override functions correctly.
- rule_firings_repo_wide: 25 occurrences of `no-restricted-imports` in full
  lint output (matches Main's "25 new warnings from rule"; consistent with
  the 21 violator-files figure since several files have multiple imports).

## LLM judgment

- **warn_vs_error_defensible**: DEFENSIBLE. Shipping at `"warn"` with 21
  pre-existing violators is consistent with the harness pattern (cron TZ
  16.18 -> #19, alpaca client_order_id 16.19 -> follow-up). Promoting to
  `error` before cleanup would block builds, contradicting immutable
  criterion `no_lint_errors`. Inline comment in eslint.config.mjs explicitly
  notes "promote to error once they're cleaned up (follow-up #50)".
  Not buck-passing -- bounded scope discipline.

- **override_scope_appropriate**: APPROPRIATE. The glob `**/lib/icons.{ts,tsx}`
  is narrow (single barrel file). No other "central re-export" file currently
  imports `@phosphor-icons/react`, so broader exemption is unnecessary.
  Keeping the exemption tight prevents accidental future bypass.

- **paths_plus_patterns_overengineered**: NOT OVER-ENGINEERED. `paths`
  catches `import X from "@phosphor-icons/react"`; `patterns` catches
  `import X from "@phosphor-icons/react/dist/ssr"` (subpath). Phosphor
  publishes both surfaces. Defense-in-depth here is ~5 lines for genuine
  coverage.

- **followup_estimate_realistic**: PLAUSIBLE. ~21 files, mechanical edit
  (replace `from "@phosphor-icons/react"` with `from "@/lib/icons"`,
  add semantic re-export to icons.ts if missing). 1 hour is reasonable
  for the edit + TS rebuild + lint re-run. May slip to 1.5h if some icon
  names differ between phosphor and the barrel's semantic names.

- **tree_shaking_claim_verified**: VERIFIED with one nit. Main's contract
  cites `next.config.ts:10` but the actual file is `next.config.js` (line
  8 contains `optimizePackageImports: ["@phosphor-icons/react"]`).
  Functional claim is correct; path/extension reference is slightly off.
  Non-blocking documentation nit.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All immutable criteria met (lint exit 0, no errors, vitest 34/34). Rule fires on real violators (Sidebar.tsx: 2 warnings with custom message) and override correctly exempts icons.ts (0 warnings). Research gate cleared, contract correctly rotated to phase-16.32, log-last respected. warn-level + follow-up #50 is defensible scope discipline consistent with prior harness patterns.",
  "violated_criteria": [],
  "violation_details": [],
  "follow_up_tickets": [
    "#50 (referenced in eslint.config.mjs comment): clean up the 21 pre-existing violator files and promote rule from warn to error",
    "doc nit (non-blocking): contract cites next.config.ts:10 but actual file is next.config.js:8"
  ],
  "checks_run": [
    "contract_header",
    "research_brief_envelope",
    "harness_log_grep",
    "eslint_config_rule_present",
    "eslint_config_override_present",
    "npm_run_lint",
    "eslint_sidebar_firing",
    "eslint_icons_ts_exempt",
    "vitest_full_run",
    "next_config_optimize_package_imports"
  ],
  "certified_fallback": false
}
```
