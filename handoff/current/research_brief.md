# Research Brief — Step 75.15 (CI gates made real)

**Tier:** moderate | **Researcher:** Layer-3 | **Date:** 2026-07-24
**Step:** 75.15 — advisory flip, requires_live migration, lock-count re-audit,
coverage runner, honest seed-stability/visual lanes, vitest + npm-audit.
**Boundary:** CI-config + test edits only; GitHub-runner lanes; NOTHING touches
operator :3000. Executor: sonnet-4.6/high.

> WRITE-FIRST scaffold. Sections fill incrementally as sources are read /
> internal files measured. `gate_passed` stays honest.

---

## 0. Legs (from the 75.15 node)

- (a) qa-tests-01: e2e-smoke.yml backend lane advisory since 2026-06-10, stale
  6-file `--ignore` list superseded by phase-56.2 `requires_live` marker.
- (b) qa-tests-02: lock-count guard — step text says 16 vs 15; MEASURE current.
- (c) qa-tests-04: coverage tier runner (scripts/qa/coverage_tier_check.py).
- (d) qa-tests-05: seed-stability-check.yml re-scope (frozen JSON).
- (e) qa-tests-06: visual-regression.yml baseline gating.
- (f) qa-tests-10: vitest unit suite into e2e-smoke frontend step.
- (g) deps-06: npm-audit.yml weekly + lockfile-change.

---

## Internal code inventory (ALL measured 2026-07-24, this session)

### Workflow lanes (`.github/workflows/`)
| File | Lines | Runs | Advisory? | Notes |
|------|-------|------|-----------|-------|
| e2e-smoke.yml | 99 | ast -> backend pytest (`--ignore` 6 files) -> frontend tsc+build -> harness dry-run -> intel/phase6 e2e | **YES** `continue-on-error: true` :36 (since 2026-06-10) | ONLY lane running backend/tests/. `--ignore` list :73-78. Frontend step does `npm ci && npx tsc --noEmit && npm run build` — NO vitest yet. |
| pip-audit.yml | 76 | `pip-audit --requirement .lock/.txt --strict` | no (enforcing) | 75.13 modified. push+PR(paths)+weekly Mon 07:00 + dispatch. **This is the mirror template for npm-audit.yml (leg g).** |
| seed-stability-check.yml | 43 | `seed_stability_test.py` reads FROZEN `handoff/seed_stability_results.json` | no (enforcing on PR) | comment :7-9 "A failing drill / blocks the PR" — the overclaim. Recomputes std over a frozen JSON → cannot catch a code regression. |
| visual-regression.yml | 94 | `npx playwright test` on push/PR (frontend paths) | no | **ZERO baselines** — `frontend/tests/visual-regression/snapshots/chromium/` exists but has 0 PNGs → every push/PR run guaranteed-red. |
| ascii-logger-lint / claude-code-review / claude / env-syntax-lint / governance-lint / limits-tag-enforcement | — | lint/governance | mixed | not in 75.15 scope. |

### `requires_live` marker (leg a)
- Registered in **`pytest.ini`:9** (phase-56.2). `requires_live: ... skipped unless PYFINAGENT_LIVE_TESTS=1`.
- `backend/tests/conftest.py` sets `PYFINAGENT_TEST_NO_BQ=1` at import (BQ isolation) — orthogonal to requires_live; there is **no auto-skip** of requires_live in conftest, so `-m "not requires_live"` is the ONLY deselector.
- The marker is applied **per-test-function**, NOT per-file. Files carrying it (some on only ONE of several tests): test_phase_23_2_9, test_64_4 (x3), test_phase_23_2_5, test_phase_62_4 (x2), test_phase_23_2_12, test_phase_23_2_11. Collection: **1462/1474 selected, 12 deselected** under `-m "not requires_live"`.

### The 6-file `--ignore` list vs reality (leg a)
| Ignored file | Marked? | Runs green under `-m "not requires_live"`? | Action |
|---|---|---|---|
| test_phase_23_2_10_watchdog_no_fire_7d.py | NO | **RED** (watchdog log 1016h stale — needs live backend) | **MARK requires_live** |
| test_phase_23_2_12_layer1_pipeline_active.py | YES | deselected | already handled |
| test_phase_23_2_14_no_reentrant_locks.py | NO | **GREEN (passes)** | un-ignore (runs) |
| test_phase_23_2_16_shortlist_doc_presence.py | NO | GREEN | un-ignore |
| test_agent_map_live_model.py | NO | GREEN | un-ignore |
| test_rainbow_canary.py | NO | GREEN (BQ-free confirmed) | un-ignore |

### THE CRITICAL WORKLIST — full red baseline under `-m "not requires_live"` (10 failed, 1444 passed, 12 deselected, measured 2026-07-24)
The step's "5 unmarked files" is a serious UNDERCOUNT. The real red set spans **8 files**, only ONE of which (test_phase_23_2_10) is in the current `--ignore` list. Categorized by root cause:

**A. Genuine live/backend.log dependency → MARK `requires_live` (per-test):**
1. test_phase_23_2_10::watchdog_log_present_and_fresh — watchdog log freshness < 24h (got 1016h).
2. test_phase_23_2_6::backend_log_has_skipping_buy_evidence — greps backend.log for 'Skipping BUY' (got 0).
3. test_phase_23_2_9::backend_log_has_prewarm_evidence — greps backend.log for prewarm line (got 0). *(file already marks the DIFFERENT latency test; mark this one too.)*
4. test_phase_23_2_15::known_pass_scripts_still_pass — shells out to `verify_phase_23_1_22.py` (exit=1 in this env). *(triage: env-sensitive verify script; mark or fix.)*

**B. LOCAL-.env pollution → GREEN on a fresh CI runner; do NOT mark (would lose coverage):**
5. test_phase_57_1::reject_binding_main_path_off — asserts default OFF; operator `.env` has `paper_risk_judge_reject_binding` ON.
6. test_phase_57_1::reject_binding_swap_path_off — same flag.
7. test_phase_57_1::off_identity_prompts_are_verbatim_constants — same flag (also an `is`-identity fragility; strings equal, not same object).
8. test_phase_60_3::flag_defaults_off — asserts `paper_data_integrity_enabled` default False; operator `.env` has it ON.
   - **Both flags default `False` in code** (settings.py:47 & :309); `model_config` hardcodes `env_file=_ENV_FILE` (:613). `.env` is **gitignored/untracked** → CI runner uses False defaults → these 4 PASS on CI. Confirmed: forcing `PAPER_DATA_INTEGRITY_ENABLED=false PAPER_RISK_JUDGE_REJECT_BINDING=false` → all 4 pass (20 passed / 2 failed).
   - `Settings(_env_file=None)` does NOT work as a robust-fix (required fields gcp_project_id/rag_data_store_id/… come from .env → ValidationError). Robust path = run the local green-baseline with the two flags forced to defaults, documented as CI-equivalent.

**C. Genuine drift/logic → FIX, do NOT mark (not live-dependent):**
9. test_phase_60_1::claude_code_rail_declares_latency_profile — `assert ClaudeCodeClient.recommended_step_timeout(=150 class attr) > client._timeout_s(=150)` → `150 > 150` False. phase-61.2 moved `recommended_step_timeout` to an INSTANCE attr (`timeout_s+30`) at claude_code_client.py:487 but the test reads the stale CLASS attr (:478). Fails at defaults, independent of .env. **Test-vs-code drift; fix the assertion (read instance attr) or the class default.**
10. test_portfolio_swap::swap_framework_fills_zero_buy_gap — pure-logic TradeOrder test, expects 2 swap SELLs got 1. Fails even at flag defaults. phase-70 "non-atomic swap" area. **Triage: real logic/expectation drift; fix.**

> **Net for a GREEN enforcing lane on CI:** mark A(4) requires_live; fix C(2); B(4) are green on CI at defaults. The local "green" capture (criterion #3) must be run at flag defaults (CI-equivalent), NOT against the operator's live `.env`, or it will show B's 4 as red. This tension must be disclosed in experiment_results.md.

### Lock-count guard (leg b) — STEP TEXT IS STALE
- `test_phase_23_2_14_no_reentrant_locks.py` **PASSES TODAY** (5 passed, 0.09s). `EXPECTED_LOCK_COUNT = 18`; measured real count = **18** (exact match).
- **phase-75.10 already bumped 17→18 on 2026-07-23** (added `mas_events.py:108` `_remote_worker_lock`, with a full audit note :23-34). phase-75.5 bumped 15→17 (added claude_code_client.py:103 as 16th + spend.py:39 as 17th).
- Step text "16 real hits vs EXPECTED_LOCK_COUNT=15 … fails TODAY" is **~2 bumps stale** (matches the pre-75.5 snapshot). **Nothing to bump.**
- claude_code_client.py:**103** (`_RAIL_GUARD_LOCK`) IS in the roster (16th, audited). Step says :102 (off-by-one).
- api/job_status_api.py:87 (`_lock`) IS counted, but it is **NOT a new lock** — added way back in phase-10.x (commit 1122a021), one of the oldest locks. Step's claim it "postdate[s] the roster" is FALSE. It is not individually named in the audit note (nor are the other pre-75.5 locks), which is fine since the count matches. Residual leg-(b) action = just ensure the file is collected (the `-m "not requires_live"` migration does this; it's unmarked and passes).

### Coverage (leg c) — measured 2026-07-24 (this session)
- `scripts/qa/coverage_tier_check.py` — **ABSENT** (must create). `pytest-cov 7.1.0` installed (75.13 lock).
- `docs/coverage_tier_overrides.md` — exists, **authored 2026-05-25 (cycle 53) = stale**. Tier-1 EXTENDED bar = >=75% (STRICT line+branch). Doc numbers: paper_trader 79.1%, portfolio_manager 81.2%, perf_metrics 81.2%.
- **Refreshed (full non-live suite `--cov`):** paper_trader **78.3%** (DRIFTED DOWN 0.8pp, still > 75% bar), perf_metrics **84.8%** (up), portfolio_manager **83.9%** (up). TOTAL 81.8%. All 3 above bar today → runner PASSES today (so the executor must PROVE the runner CAN fail — mutation/temporary-bar-raise — else it is a vacuous guard).
- Doc also defines a Tier-1 **STRICT** set (kill_switch/cycle_lock/factor_correlation/factor_loadings) with the same 75% bar. Step scopes the runner to the 3 EXTENDED modules; recommend the runner read bars from the doc (single source of truth) and gate at least those 3.

### Vitest lane (leg f) — BOUNDARY-SAFE
- `frontend/package.json`: `"vitest": "^4.1.4"` (devDep); `"test": "node scripts/run-test.mjs"`. `run-test.mjs` is a thin wrapper → `spawn(vitest, ["run", ...])` — **NO server, NO next dev, NO :3000.** `vitest.config.ts`: `environment: "jsdom"` — pure component tests, no live backend.
- **29** `*.test.ts(x)` files under frontend/src (step says "10 money-display" — stale/undercount). Includes PortfolioAllocationDonut, SectorBarList, ComputeCostBreakdown, AlphaLeaderboard, etc. + lib tests (api, auth.config, paper-trading-utils).
- Verification asserts `('vitest' in y or 'npm run test' in y)`. Adding `npm run test` (or `npx vitest run`) to the e2e-smoke frontend step after `tsc` satisfies it AND is CI/BOUNDARY-safe.

### npm-audit (leg g)
- `.github/workflows/npm-audit.yml` — **ABSENT** (must create). Mirror pip-audit.yml: schedule weekly + `pull_request`/`push` on `frontend/package-lock.json` change; `npm ci && npm audit --audit-level=high` in `frontend/`; upload artifact on failure. NO server.

### Visual-regression (leg e)
- `visual-regression.yml` fires on push+PR (frontend paths). ZERO committed baselines → guaranteed-red. Fix: gate the push/PR triggers on baseline presence, OR remove push/PR triggers (leave `workflow_dispatch` for the operator first-run capture). The immutable verification command does NOT assert anything about this file — covered only by success_criterion #5.

### Immutable verification command (read verbatim from node)
```
python3 -c "... y=open('.github/workflows/e2e-smoke.yml').read();
  assert 'continue-on-error: true' not in y;
  assert 'not requires_live' in y;
  assert ('vitest' in y or 'npm run test' in y);
  assert os.path.exists('.github/workflows/npm-audit.yml');
  assert os.path.exists('scripts/qa/coverage_tier_check.py');
  s=open('.github/workflows/seed-stability-check.yml').read();
  assert ('blocks the PR' not in s) or ('run_seed_stability' in s)"
&& .venv/bin/python -m pytest backend/tests/test_phase_23_2_14_no_reentrant_locks.py -q
```
- The lock test **already passes** (leg b is a no-op).
- `'continue-on-error: true'` substring: note the current file has the exact string at :36 AND in a comment at :13/:36. The executor must remove/false BOTH the active key and ensure the literal substring `continue-on-error: true` is gone (comment at :13 "flip to false once green" is fine; comment on :36 trails the key). Verify the assertion after editing.
- `'blocks the PR'` is currently PRESENT in seed-stability-check.yml:8 and `run_seed_stability` is absent → command FAILS today on that leg until reworded.
- **The command does NOT gate legs (e) visual-regression, (f) beyond the vitest substring, or the requires_live-migration correctness** (only that the substring `not requires_live` appears). A vacuous `-m "not requires_live"` with a still-red suite would pass the command but fail success_criterion #3 — Q/A must check the pasted green tail, not just the command.

## Research: External CI/testing best practices

### Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 1 | https://docs.pytest.org/en/stable/how-to/mark.html | 2026-07-24 | official doc | `-m "not MARKER"` **deselects** marked tests (they don't run, not skipped). Register markers in pytest.ini to avoid warnings; `--strict-markers` makes unknown markers an error. |
| 2 | https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/troubleshooting-required-status-checks | 2026-07-24 | official doc | "Successful check statuses are `success`, `skipped`, and `neutral`." Required check "must have completed successfully … during the past seven days" and "must pass on the latest commit SHA." |
| 3 | https://docs.npmjs.com/cli/v10/commands/npm-audit | 2026-07-24 | official doc | `--audit-level` = "minimum level of vulnerability for npm audit to exit with a non-zero exit code" (info/low/moderate/high/critical/none). Exit 0 if none found; needs a package-lock; `npm audit fix` mutates (do NOT use in gate); `--omit=dev` for prod-only. |
| 4 | https://coverage.readthedocs.io/en/latest/config.html | 2026-07-24 | official doc | `fail_under` is **project-wide ONLY** — "If the total coverage measurement is under this value, then exit with a status code of 2." **No native per-file/per-module threshold.** `precision` needed for non-integer bars. |
| 5 | https://www.kenmuse.com/blog/how-to-handle-step-and-job-errors-in-github-actions/ | 2026-07-24 | practitioner (ex-GitHub) | Step-level `continue-on-error:true`: step `outcome`=failure but `conclusion`=**success** → "the job and step will succeed (green bubble)" → satisfies branch protection. Job-level differs (workflow still reports failure). |
| 6 | https://community.sonarsource.com/t/ratcheting-quality-gate-conditions.../15317 | 2026-07-24 | vendor community | Ratchet / "Clean as You Code": migrate advisory→enforcing by gating **new/changed** code first, incrementally raise bars, never regress — avoids the all-or-nothing legacy-debt dilemma. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched |
|-----|------|-----------------|
| github.com/orgs/community/discussions/15452 | forum | continue-on-error PR-UI display — corroborates #5 |
| github.com/pytest-dev/pytest-cov/issues/444, /issues/728 | issue | "set minimum coverage per file" / multi-dir fail — confirms per-module gate is NOT built-in (validates leg c) |
| github.com/Botpy/modcov | tool | per-module coverage checker (a reference impl for coverage_tier_check.py) |
| til.simonwillison.net/github-actions/continue-on-error | blog | continue-on-error idiom |
| notes.kodekloud.com/.../Using-continue-on-error-expression | course | conclusion=success semantics |
| nerdleveltech.com/vitest-coverage-thresholds-fail-ci | blog | vitest `coverage.thresholds` (frontend analog) |
| qaskills.sh/blog/sonarqube-quality-gates-testing-guide-2026 | blog | ratchet/quality-gate 2026 |
| oneuptime.com …status-checks (2026-01), …pytest-markers (2026-02), …pytest-cov (2026-03) | blog | recency corroboration |
| shattered.io/npm-audit-nodejs, pkgpulse.com/guides/…2026, safeguard.sh/…npm-audit | blog | npm audit CI 2026 practice (`--omit=dev`, commit lockfile) |
| github.com/orgs/community/discussions/{26698,54877,54965,167194,26668} | forum | required-check "waiting"/skipped-job pitfalls |
| poseidon/wait-for-status-checks | repo | placeholder-job pattern for skipped required checks |

## Recency scan (2024-2026)
Searched 2026-scoped variants for all five topics. **Result: no NEW finding supersedes the canonical semantics** — pytest `-m`, coverage `fail_under`, GH `continue-on-error`/required-checks, and `npm audit --audit-level` are stable across the window. 2026-dated practitioner sources (oneuptime Jan-Mar 2026, pkgpulse/safeguard 2026, qaskills 2026) reaffirm the same guidance: `--audit-level=high` + commit lockfile + `npm ci`; per-module coverage still needs a custom runner (pytest-cov #444 open); "Clean as You Code" remains the recommended advisory→enforcing migration. The Ken Muse continue-on-error reference is 2024 but the behavior is unchanged in 2026 (corroborated by 2026 forum threads).

## Key findings
1. **The advisory lane literally cannot block merges.** Step-level `continue-on-error:true` forces `conclusion=success` regardless of the step failing (Ken Muse; community #15452). e2e-smoke.yml:36 has been reporting green on every red run since 2026-06-10. Flipping to `false` is the whole point of leg (a) — but it means EVERY collected-and-failing test now reddens the lane.
2. **`-m "not requires_live"` deselects; it does not skip.** Deselected tests don't run, so they can't fail the job; a passing job = `success` = satisfies branch protection (pytest docs #1; GH docs #2). The migration is mechanically sound. BUT the marker is per-TEST — marking one test in a file does not exclude the file's other tests (measured: test_phase_23_2_9 has a marked latency test AND an unmarked log-evidence test that fails).
3. **Per-module coverage bars require a custom runner — the tooling has no native support.** `coverage fail_under` is a single project total (exit 2); pytest-cov issue #444 (per-file) is still open (coverage docs #4). This directly validates leg (c): `scripts/qa/coverage_tier_check.py` MUST parse per-module coverage (e.g. `coverage json` → `files[path].summary.percent_covered`) and compare each to its `docs/coverage_tier_overrides.md` bar. Reference impl: `modcov`.
4. **`npm audit --audit-level=high` fails only on high/critical, exit non-zero, needs package-lock** (npm docs #3). Mirroring pip-audit.yml's shape (weekly + lockfile-path trigger + artifact-on-failure) is correct; use `npm ci` (deterministic) not `npm install`; never `npm audit fix` in a gate (it mutates deps).
5. **Ratchet / Clean-as-You-Code** (Sonar #6): the disciplined advisory→enforcing path is "make the enforced subset green, THEN flip" — never flip onto a red suite. `requires_live` is the project's quarantine lane (the legacy-debt carve-out); the credential-free subset is the enforced "new code" gate. This is exactly the pattern the step prescribes.

## Consensus vs debate
Consensus across all sources: quarantine environment-coupled tests behind a marker, enforce only the deterministic subset, gate per-module coverage with a bespoke script, and never enable a blocking gate on a red baseline. Minor debate: some practitioners run npm audit as **advisory** (report-only) because transitive advisories appear overnight and can red a lane with no code change — mitigated here by weekly + lockfile-change triggers (not every push) and artifact-on-failure. The step chooses enforcing `--audit-level=high`; acceptable given the narrow trigger set, but flag for operator awareness.

## Pitfalls (from literature + measurement)
- **Flip onto red = instant lane failure.** 10 tests fail under `-m "not requires_live"` locally today; on CI ~6 fail (the 4 .env-pollution ones pass at defaults). Do the full worklist BEFORE flipping.
- **Over-marking hides coverage.** Marking the 4 .env-pollution tests `requires_live` would deselect them on CI where they SHOULD pass and guard the shipped default — semantic error. Run the local green baseline at flag defaults instead.
- **Vacuous coverage guard.** All 3 Tier-1 modules are above bar today (78.3/83.9/84.8%), so a fresh runner passes trivially. Prove it CAN fail (mutation: temporarily raise a bar above current, or point it at a known-low module) — else it's a guard that can't fail (auto-memory `feedback_mutation_test_guards_and_fixtures`).
- **Immutable command is a weak check.** It only asserts the substring `not requires_live` exists and the lock test passes — a still-red suite would pass the command. Q/A must verify the pasted green tail (criterion #3), not just exit 0.
- **Visual-regression is NOT in the verification command.** Playwright's first run with zero baselines FAILS on CI (writes baseline, reports failure). Gate push/PR triggers on baseline presence, or keep only `workflow_dispatch`.

## Application to pyfinagent (mapping to file:line anchors)
- Leg (a): e2e-smoke.yml:36 remove `continue-on-error`; e2e-smoke.yml:71-78 replace `--ignore` block with `-m "not requires_live"`. Mark per-test `requires_live` on the 4 Category-A failures; FIX the 2 Category-C (test_60_1 assertion vs claude_code_client.py:478/487; test_portfolio_swap logic). Un-ignore the 4 green files incl. test_rainbow_canary. Capture the green tail at flag defaults into experiment_results.md BEFORE the flip.
- Leg (b): **no-op** — test_phase_23_2_14 already passes at EXPECTED_LOCK_COUNT=18 (phase-75.10). Just ensure it's collected (the migration does this).
- Leg (c): create scripts/qa/coverage_tier_check.py (custom per-module, since fail_under can't); wire a nightly workflow leg; refresh docs/coverage_tier_overrides.md with 2026-07-24 numbers (paper_trader 78.3, portfolio_manager 83.9, perf_metrics 84.8); prove it can fail.
- Leg (d): seed-stability-check.yml:7-9 reword (drop "blocks the PR"); satisfies `'blocks the PR' not in s`.
- Leg (e): visual-regression.yml:18-28 gate push/PR triggers on baseline presence (or drop them; keep workflow_dispatch).
- Leg (f): add `npm run test` (→ vitest run, jsdom, no server) to e2e-smoke.yml frontend step after tsc.
- Leg (g): new .github/workflows/npm-audit.yml mirroring pip-audit.yml.

## Internal code inventory (file:line anchors)
| File | Lines | Role | Status |
|------|-------|------|--------|
| .github/workflows/e2e-smoke.yml | 36, 71-78 | advisory backend lane + 6-file ignore | flip + migrate |
| .github/workflows/pip-audit.yml | 19-67 | mirror template for npm-audit | reference |
| .github/workflows/seed-stability-check.yml | 7-9 | "blocks the PR" overclaim | reword |
| .github/workflows/visual-regression.yml | 18-28 | zero-baseline red trigger | gate |
| pytest.ini | 8-9 | requires_live marker registration | reference |
| backend/tests/conftest.py | 21 | PYFINAGENT_TEST_NO_BQ isolation | reference |
| backend/tests/test_phase_23_2_14_no_reentrant_locks.py | 22-34 | lock guard (PASSES, count=18) | no-op |
| backend/config/settings.py | 47, 309, 613 | flag defaults False + hardcoded env_file | root cause of .env-pollution reds |
| backend/agents/claude_code_client.py | 478, 487 | class-vs-instance recommended_step_timeout | fix target (test_60_1) |
| backend/api/job_status_api.py | 87 | _lock (phase-10.x, NOT new) | already in roster |
| frontend/scripts/run-test.mjs, vitest.config.ts, package.json | — | vitest run / jsdom / no server | BOUNDARY-safe |
| docs/coverage_tier_overrides.md | 25-46 | Tier-1 bars, stale 2026-05-25 | refresh |

## Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (~32)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (all 7 workflows + marker + settings + tests + coverage doc measured live)
- [x] Contradictions / consensus noted (npm audit advisory-vs-enforcing)
- [x] All claims cited per-claim

## JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 26,
  "urls_collected": 32,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Step 75.15 makes CI gates real. MEASURED: the advisory backend lane cannot block (step-level continue-on-error forces conclusion=success). Leg (b) is a NO-OP -- the lock guard already passes at EXPECTED_LOCK_COUNT=18 (phase-75.10 bumped it 2026-07-23); step text's '16 vs 15' is 2 bumps stale and api/job_status_api.py:87 is an OLD lock, not new. THE critical worklist: under -m 'not requires_live', 10 tests fail across 8 files (only 1 is in the current 6-file ignore). 4 are genuine live/backend.log deps -> MARK; 4 are operator-.env pollution (flags default False in code, .env is gitignored) -> GREEN on CI, do NOT mark, run local baseline at defaults; 2 are genuine drift (test_60_1 class-vs-instance timeout; test_portfolio_swap logic) -> FIX. coverage fail_under is project-wide only -> the custom coverage_tier_check.py runner is REQUIRED (validated); refreshed numbers paper_trader 78.3/portfolio_manager 83.9/perf_metrics 84.8. Vitest lane is BOUNDARY-safe (run-test.mjs->vitest run, jsdom, no server). npm-audit mirrors pip-audit. gate_passed true.",
  "brief_path": "handoff/current/research_brief_75.15.md",
  "gate_passed": true
}
```
