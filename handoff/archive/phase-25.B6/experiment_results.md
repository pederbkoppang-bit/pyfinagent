---
step: phase-25.B6
cycle: 97
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.B6

## What was built/changed

Closed audit bucket 24.6 F-2 by wrapping the existing seed-stability
drill in a CI gate:

1. **`.github/workflows/seed-stability-check.yml`** -- NEW workflow:
   - Triggers: `pull_request` to main + `workflow_dispatch`.
   - Job: ubuntu-latest, Python 3.14.
   - Runs `python scripts/go_live_drills/seed_stability_test.py`.
   - Uploads the committed baseline JSON as a workflow artifact.
   - Timeout: 5 minutes (drill is stdlib-only, sub-second).
2. **Baseline JSON** at `handoff/seed_stability_results.json` already
   committed with the current shape: 5 seeds, mean_sharpe=0.589,
   std_sharpe=0.0094 (well below the 0.1 gate threshold).
3. **Drill script** at `scripts/go_live_drills/seed_stability_test.py`
   already enforces `STD_THRESHOLD = 0.1` at check S5 -- the failing
   case is `std_sharpe >= 0.1` which triggers `failed += 1` and
   `sys.exit(1)` at the bottom.

## Files changed

| File | Action |
|------|--------|
| `.github/workflows/seed-stability-check.yml` | NEW (43 lines) |
| `tests/verify_phase_25_B6.py` | NEW verifier (6 claims) |

(No code changes; the drill + baseline were already in place.)

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_B6.py

=== phase-25.B6 verification ===

[PASS] 1. seed_stability_results_json_committed_with_baseline
        -> exists=True schema_keys_ok=True seeds_count=5
[PASS] 2. baseline_std_sharpe_below_threshold
        -> std_sharpe=0.0094 (threshold=0.1)
[PASS] 3. github_actions_seed_stability_check_yml_passes
        -> exists=True
[PASS] 4. workflow_invokes_seed_stability_drill
        -> Found script reference in workflow
[PASS] 5. stddev_threshold_enforced_in_ci
        -> STD_THRESHOLD_const=True std_lt_threshold_check=True
[PASS] 6. drill_passes_std_gate_check_on_current_baseline
        -> S5 gate present in drill output=True exit=0

ALL 6 CLAIMS PASS
```

## Success criteria -> evidence

1. `seed_stability_results_json_committed_with_baseline` -- Claims 1 + 2 PASS:
   baseline JSON tracked in git with canonical schema; std_sharpe=0.0094.
2. `github_actions_seed_stability_check_yml_passes` -- Claims 3 + 4 + 6 PASS:
   workflow exists, invokes the drill, drill exits 0 on current baseline.
3. `stddev_threshold_enforced_in_ci` -- Claim 5 PASS: STD_THRESHOLD=0.1 constant
   plus `std_sharpe < STD_THRESHOLD` gate check both in the drill.

## Out-of-scope / deferred

- Path-filtered triggers (only run on backtest-related PRs): the drill
  is sub-second so unconditional triggering is acceptable.
- Backtest engine re-run in CI: the drill is stdlib-only -- it reads
  the committed baseline. A "fresh-baseline" CI step would be heavy
  and is intentionally NOT here; local `scripts/harness/run_seed_stability.py`
  is the canonical path for regenerating the baseline.

## References

- `handoff/current/research_brief.md`
- `scripts/go_live_drills/seed_stability_test.py` (drill)
- `handoff/seed_stability_results.json` (baseline)
- `.github/workflows/seed-stability-check.yml` (NEW)
- `.claude/masterplan.json::25.B6`
