---
step: 25.B6
slug: seed-stability-ci-gate
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.B6

## Step ID + masterplan reference

`25.B6` -- "Seed-stability test run + baseline commit + CI gate"
(P2, harness_required, no dep).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`. Existing infrastructure is 90% complete; this
cycle adds the CI workflow wrapper.

## Hypothesis

The seed-stability drill script + baseline JSON already exist and pass
the std<0.1 gate. Adding a `.github/workflows/seed-stability-check.yml`
that invokes the drill on every PR locks the reproducibility guarantee
into CI -- a future change that bumps std above 0.1 will fail the gate.

## Success criteria (verbatim from masterplan.json)

> `seed_stability_results_json_committed_with_baseline`
>
> `github_actions_seed_stability_check_yml_passes`
>
> `stddev_threshold_enforced_in_ci`

## Plan steps

1. **`.github/workflows/seed-stability-check.yml`** -- new workflow:
   - Triggers: `pull_request`, `workflow_dispatch`.
   - Job: ubuntu-latest, Python 3.14.
   - Step: `python scripts/go_live_drills/seed_stability_test.py`.
   - Exits 1 on drill failure (which includes std>=0.1).
2. **Verifier** `tests/verify_phase_25_B6.py` with 5 claims:
   - Claim 1: baseline JSON exists with the canonical schema.
   - Claim 2: baseline JSON has std_sharpe < 0.1 (gate satisfied).
   - Claim 3: workflow file exists.
   - Claim 4: workflow invokes the seed_stability_test.py drill.
   - Claim 5: drill script enforces `STD_THRESHOLD = 0.1` (regex on source).

## Files

| File | Action |
|------|--------|
| `.github/workflows/seed-stability-check.yml` | NEW |
| `tests/verify_phase_25_B6.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_B6.py
```

## Live-check

`CI run completes seed-stability test; baseline checked into repo`.
The verifier covers the static + behavioral checks; CI itself will
exercise the workflow on the first PR.

## Risks + mitigations

- **Risk**: The seed-stability drill needs the backtest engine to run,
  which is heavy and slow in CI.
  **Mitigation**: The drill is STDLIB-ONLY -- it reads the JSON baseline
  and computes std. No backtest run in CI. The actual run is local +
  manual via `scripts/harness/run_seed_stability.py`.
- **Risk**: PRs that don't touch backtest code re-run this gate needlessly.
  **Mitigation**: Acceptable -- the drill is <1s. If load becomes an issue,
  add a path-filter on backtest-touching paths.

## References

- `handoff/current/research_brief.md`
- `scripts/go_live_drills/seed_stability_test.py`
- `handoff/seed_stability_results.json`
- `.github/workflows/visual-regression.yml` (template)
- `.claude/masterplan.json::25.B6`
