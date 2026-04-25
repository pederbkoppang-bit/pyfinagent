step: phase-16.41
verdict: PASS
date: 2026-04-25
counter: 0 (first verdict this cycle)

# Q/A Critique -- phase-16.41 (authenticated-home lighthouse harness, #8)

## Step 1: Harness-compliance audit (5 items)

1. `handoff/current/phase-16.41-research-brief.md` exists. PASS
   (gate_passed asserted in contract: tier=moderate, 6 in-full, 14 URLs,
   recency scan present.)
2. `handoff/current/contract.md` line 2 = `step: phase-16.41`. PASS
3. `handoff/current/experiment_results.md` line 2 = `step: phase-16.41`. PASS
4. `grep -c "phase-16.41" handoff/harness_log.md` = 0. PASS (log-last
   discipline preserved; will be appended after this PASS.)
5. Prior `evaluator_critique.md` carried `step: phase-16.40` -- confirmed
   pre-overwrite. PASS

## Step 2: Deterministic checks

| Check | Result |
|-------|--------|
| Immutable verification cmd (4 file/grep checks) | exit 0, "ALL VERIFICATION PASS" |
| `--probe-only` smoke test (server lacks bypass env) | exit 2 (expected) |
| `frontend/src/middleware.ts:24` LIGHTHOUSE_SKIP_AUTH clause | present |
| git scope | `M frontend/package.json`, `?? lighthouse_auth_home.js`, `?? docs/runbooks/lighthouse-auth-home.md` -- exactly the 3 expected; no middleware/auth drift |

All deterministic checks PASS.

## Step 3: LLM judgment

- **Approach**: Option C (env-var bypass) is correctly justified in the
  contract ("research-gate summary" cites middleware.ts:24 and saves
  ~60 lines of crypto). Mirrors researcher recommendation.
- **Reuse**: `runLighthouse()` invokes existing `lighthouse-wrapper.js`
  via spawnSync at line 135 -- no chrome-path / argv duplication.
- **Safety net**: `--probe-only` mode (line 52, 194-204) emits an
  operator-friendly error + exits 2 when the dev server isn't in
  bypass mode. Live smoke-test in experiment_results.md confirms this
  behavior (302 -> /login, exit 2, helpful "Restart it with: ..."
  message). Prevents the "Lighthouse landed on /login" footgun.
- **Pattern consistency**: JSON envelope shape (`audit`, `mode`,
  `timestamp`, `overall`, `checks[]`, `output_report`) matches
  sovereign_route.js convention. `pass()` / `fail()` helpers same
  shape. Exit-code semantics documented in header.
- **Defensive coding**: `checkFinalUrl()` tries 3 field names
  (`finalUrl`, `finalDisplayedUrl`, `lhr.finalUrl`) for lighthouse
  v13 + older drift. `runLighthouse()` checks both exit status AND
  `fs.existsSync(OUTPUT_PATH)` (lighthouse can return 0 with no
  report on warnings).
- **Runbook**: `docs/runbooks/lighthouse-auth-home.md` (95 lines)
  documents purpose, mechanism, 2-terminal how-to-run, probe-only,
  output reading, troubleshooting. Security note that
  `LIGHTHOUSE_SKIP_AUTH=1` MUST never be set in production is
  present (per experiment_results §3 of "Changes").
- **Honest disclosures**: 6 disclosed in experiment_results
  (no live lighthouse run this cycle, output path location,
  wrapper reuse, exit-code nuance, finalUrl flexibility, no
  probe-only test in CI). Each is appropriate scope-honesty,
  not overclaim.
- **Mutation-resistance**: implicit -- if the bypass clause were
  removed from middleware.ts, the probe would return 302 and the
  script would exit 2 with a clear error. Live smoke-test
  effectively demonstrates this (the dev server has no env var,
  so it's the same observable state as "bypass clause missing").

## Step 4: Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 5 immutable criteria met (script_exists, npm_script_added, script_aware_of_bypass, runbook_exists, script_probe_works). Deterministic verification cmd exit=0; probe-only smoke test exit=2 as expected; git scope clean (3 expected files, no middleware drift). Pattern mirrors sovereign_route.js; reuses lighthouse-wrapper.js. Honest disclosures cover scope (infrastructure-only, full audit is operator-action). Closes task-list item #8.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5",
    "verification_command",
    "probe_only_smoke",
    "middleware_bypass_clause_present",
    "git_scope",
    "contract_alignment",
    "pattern_consistency",
    "scope_honesty"
  ]
}
```

**Verdict: PASS.** Proceed to log-last (`harness_log.md` append) then
flip masterplan `phase-16.41` to `done`.
