# Contract -- Step 75.15: CI gates made real

- **Step id**: 75.15 (phase-75, Audit75 S15) -- P1, executor sonnet-tier -> **GENERATE delegated to a Sonnet-4.6 executor** (6th delegated step; Main reviews + re-measures before Q/A).
- **Date**: 2026-07-24
- **BOUNDARY (step text + Main sharpening)**: CI-config + test edits only; all lanes run on GitHub runners; NOTHING touches the operator :3000 (second-next-dev memory binding); local triage = pytest + `vitest run` only, no servers. **Main sharpening**: if either genuine-drift red (leg-a class C) turns out to be a PRODUCTION bug rather than test drift, the production fix is OUT of scope -> queue it as its own step and handle the test honestly (xfail with reference), never a silent production edit.

## Research-gate summary (gate PASSED)

Workflow `wf_ae42985c-98c` (researcher, opus/max, moderate). Envelope: `6 read-in-full (GH Actions continue-on-error/branch-protection, pytest markers, coverage.py fail_under + pytest-cov #444, npm audit, ratchet-pattern), snippet=26, urls=32, recency=true, internal=20, gate_passed=true`. Brief: `research_brief_75.15.md`.

**Corrections adopted (binding):**
1. **Leg (b) is a NO-OP**: the lock guard PASSES today at EXPECTED_LOCK_COUNT=18 (75.10 bumped 17->18 with the audit note); the step's "16 vs 15 fails TODAY" is two bumps stale; job_status_api's lock is one of the OLDEST (phase-10.x), not new; the rail-guard lock is :103 and already audited. Verify-only: ensure the file is collected post-migration (it is unmarked and green).
2. **The real leg-(a) worklist is 10 reds / 8 files** (measured), classified: **(A) mark requires_live x4** -- test_phase_23_2_10 (watchdog freshness), 23_2_6 + 23_2_9's unmarked backend.log-evidence test (the marker is PER-TEST; its latency sibling is already marked), 23_2_15 (shells verify scripts). **(B) do NOT mark x4** -- test_phase_57_1 x3 + test_phase_60_3: operator-.env pollution (both flags default False; .env gitignored -> CI runs defaults and these PASS; marking them would un-guard the shipped defaults on CI). **(C) FIX x2** -- test_60_1 (class-attr recommended_step_timeout 150 vs instance _timeout_s 150 at claude_code_client.py:478/487 -- diagnose which side drifted) and test_portfolio_swap (swap-SELL expects 2 got 1, phase-70 non-atomic-swap area -- diagnose; if production is wrong, QUEUE per the boundary sharpening).
3. Of the step-named 5 unmarked files, only 23_2_10 is red; 23_2_14/16, agent_map_live_model, rainbow_canary all PASS -> un-ignoring them is safe.
4. Leg (f): 29 frontend test files (not 10); `npm test` = run-test.mjs -> `vitest run` (jsdom, serverless).
5. The immutable command's checks are SUBSTRING-level (a red suite would still pass it; leg (e) not covered at all) -> the green `-m "not requires_live"` tail + the non-vacuous coverage runner are Q/A-verified evidence, not command-verified.
6. Visual-regression.yml has ZERO committed baselines -> every push/PR run guaranteed-red -> gate triggers on baseline presence or workflow_dispatch.

**Key findings**: step-level continue-on-error:true forces conclusion=success (the lane structurally CANNOT block); coverage.py fail_under is project-wide only (per-file unsupported, pytest-cov #444) -> the bespoke runner is genuinely required; current Tier-1 coverage all above bar (paper_trader 78.3% vs 75, portfolio_manager 83.9%, perf_metrics 84.8%) -> the runner passes trivially and MUST be proven can-fail; npm audit can red on overnight advisories (mitigated by lockfile+weekly triggers; never `npm audit fix`).

## Hypothesis

The advisory lane can flip to enforcing on a PROVEN-green selection (never flip onto red), the coverage policy gains a real, can-fail runner, and the frontend/dependency lanes become honest -- all as CI-config + test edits with zero operator-machine impact.

## Immutable criteria + command

Verbatim in the masterplan node (executor + Q/A read there). Q/A-critical beyond the command: the pasted green `-m "not requires_live"` tail at SHIPPED defaults; the coverage runner proven non-vacuous; :3000 untouched.

## Plan steps

1. **Leg (a)**: resolve the 10-red worklist per the A/B/C classification (mark x4 per-test; leave x4 with the .env-pollution disclosure; fix-or-queue x2 after diagnosis). Then e2e-smoke.yml: continue-on-error -> false; --ignore list -> `-m "not requires_live"`. Capture the green tail at CI-equivalent env (`PAPER_DATA_INTEGRITY_ENABLED=false PAPER_RISK_JUDGE_REJECT_BINDING=false`) + disclose the operator-.env delta.
2. **Leg (b)**: verify-only (guard green at 18; collected post-migration). No bump.
3. **Leg (c)**: scripts/qa/coverage_tier_check.py -- `coverage json` per-module parser; bars read from docs/coverage_tier_overrides.md (single source of truth); exit non-zero below bar; refresh the doc's numbers to 2026-07-24 measurements; nightly workflow leg (not PR); PROVE can-fail by mutation.
4. **Legs (d)/(e)**: seed-stability wording fix; visual-regression triggers gated on baseline presence or workflow_dispatch (uncovered by the command -- do NOT skip silently).
5. **Leg (f)**: `npm run test` added to the e2e-smoke frontend step after tsc.
6. **Leg (g)**: npm-audit.yml mirroring pip-audit.yml (weekly cron + lockfile-path triggers; `npm ci && npm audit --audit-level=high` in frontend/; artifact on failure; NEVER `npm audit fix`).
7. **Tests/mutations**: prove the coverage runner fails when a bar exceeds current coverage; prove the marker migration selects/deselects correctly (collection counts); the 2 fixed tests get their fix rationale documented (which side drifted + evidence); standard matrix (un-flip continue-on-error; drop the -m selection; un-mark one requires_live test; break the runner's parser -- each KILLED or hard-fails).
8. **live_check_75.15.md**: verbatim command output + green tail + collection counts + git diff --stat + the :3000 untouched proof (no server started; curl before/after optional read-only).

## NOT in scope
Production code changes (queue if diagnosis demands); visual-regression baseline creation; branch-protection settings (GitHub-side admin, operator-owned -- document as a runbook note); the operator's .env flags.

## References
research_brief_75.15.md; audit_phase75/confirmed_findings.json (qa-tests-01/02/04...); feedback_second_next_dev_breaks_operator_3000; the 75.13 pip-audit.yml precedent.
