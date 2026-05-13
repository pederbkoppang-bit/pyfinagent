# Live-check placeholder -- phase-25.B6

**Step:** 25.B6 -- Seed-stability test run + baseline commit + CI gate
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "CI run completes seed-stability test; baseline checked into repo"

## Pre-deployment evidence
- 6/6 verifier PASS.
- Direct drill invocation (`python scripts/go_live_drills/seed_stability_test.py`)
  exits 0 with the committed baseline (std_sharpe=0.0094 << 0.1).
- Workflow YAML lints clean.
- Baseline JSON at `handoff/seed_stability_results.json` already tracked in git.

## Post-deployment operator workflow
1. Pull main:
   ```
   git pull origin main
   ```
2. Trigger the workflow manually to verify (or open any PR):
   ```
   gh workflow run seed-stability-check.yml
   gh run watch
   ```
3. Expected output:
   ```
   SEED STABILITY DRILL -- Phase 4.4.1.3
   ============================================================
   [PASS] S0 results file exists -- seed_stability_results.json
   [PASS] S1 correct seeds tested ...
   [PASS] S5 std Sharpe < 0.1 (checklist gate) -- std=0.0094 ...
   ...
   DRILL PASSED
   ```

## Regression-prevention behavior
If a future change to the optimizer / engine causes std_sharpe to exceed
0.1, the next PR will fail the seed-stability-check workflow with
`[FAIL] S5 std Sharpe < 0.1`. The drill exits 1 and the PR is blocked.

## Closes audit basis
bucket 24.6 F-2 RESOLVED.

**Audit anchor for next bucket:** 25.B10.1 (lesser-secret cleanup),
25.D7 / 25.E7 / 25.F3 (P2 backlog), follow-ups.
