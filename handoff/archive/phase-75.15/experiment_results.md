# Experiment results -- Step 75.15 (CI gates made real)

Date: 2026-07-24. **Execution model: Sonnet executor GENERATE (6th
delegated step); Main review found and CORRECTED one executor
misclassification (below), re-measured every figure, and finalized.
Executor draft at `experiment_results_75.15_draft.md`.**

## MAIN-REVIEW CORRECTION (the load-bearing delta from the executor's report)

The executor reclassified `test_phase_23_2_15` as "fixed, needs NO
requires_live mark" and reported a 0-fail CI-equivalent tail. **That did
not reproduce for Main**: the test shells 8 verify scripts and 6 failed
in Main's shell -- root-caused to the scripts invoking a bare `python`
binary whose PATH presence varies by shell/runner (the executor's PATH
had one; Main's did not; CI runners are unknowable-before-probing, and
the scripts' real probes target live machine state anyway). Resolution:
the RESEARCH gate's original category-A classification was reinstated --
`test_phase_23_2_15` is now marked `requires_live` (with the measured
rationale in its docstring), the executor's legitimate sub-script mock
fix is KEPT (see below), and the collection-count pin updated
(1474/1490, 16 deselected). The executor's own collection-count guard
CAUGHT the marker change (it went red until the pin was updated) --
working exactly as designed.

## Worklist resolution (the 10 pre-step reds; final state)

| Category | Tests | Action |
|---|---|---|
| A: marked requires_live | 23_2_10 (watchdog freshness), 23_2_6 + 23_2_9 (backend.log evidence; per-TEST marks -- 23_2_9's latency sibling was already marked), **23_2_15 (Main correction)** | Deselected on CI; still run under PYFINAGENT_LIVE_TESTS |
| B: no change (.env pollution) | 57_1 x3, 60_3, **portfolio_swap (executor reclassification -- a third instance: paper_swap_churn_fix_enabled True in the operator .env flips the swap-delta formula; defaults False in code)** | GREEN on CI defaults; marking would un-guard the shipped defaults |
| C: fixed (test-side drift, production correct) | test_60_1 (class-attr vs instance-attr 150-boundary drift at claude_code_client.py:478/487); tests/api/test_pause_resume_timeout.py mock patched the CLASS while production calls the 75.9 `get_bq_client()` singleton -- patch targets fixed (a REAL cross-step interaction: 75.9's lru_cache silently no-op'd the mock) | No production change; no queue needed |

NO production bug found -> nothing queued from leg (a).

## What shipped (legs a-g)

- **(a)** e2e-smoke.yml backend lane: `continue-on-error` REMOVED
  (enforcing), `--ignore` list -> `-m "not requires_live"`.
- **(b)** verify-only NO-OP confirmed (lock guard green at 18 since the
  75.10 audited bump; collected under the new selection).
- **(c)** `scripts/qa/coverage_tier_check.py` (bars parsed from
  docs/coverage_tier_overrides.md, zero hardcoded; exit non-zero below
  bar; exit 2 on missing inputs -- never silent) + nightly
  coverage-tier-check.yml + doc refreshed to 2026-07-24 measurements
  (all 7 Tier-1 modules above bar: paper_trader 78.3, portfolio_manager
  83.7, perf_metrics 84.8, kill_switch 88.2, cycle_lock 83.0,
  factor_correlation 85.1, factor_loadings 78.1). Can-fail PROVEN
  (99% bar mutation -> exit 1).
- **(d)** seed-stability-check.yml "blocks the PR" overclaim removed.
- **(e)** visual-regression.yml gated on committed-baseline presence
  (0 PNGs today -> explanatory skip instead of guaranteed-red).
- **(f)** vitest lane added to e2e-smoke after tsc (local: 30 files /
  201 tests green, serverless).
- **(g)** npm-audit.yml (pip-audit mirror; weekly cron + lockfile
  triggers; `npm ci && npm audit --audit-level=high`; never audit fix).
  **DISCLOSED: currently exits 1 locally -- 42 vulnerabilities (19 high,
  3 critical: tmp/undici/vite/ws transitives) -- the lane WILL be red on
  its first CI run. That is the honest signal, not hidden by raising the
  level; remediation is its own follow-up.**
- NEW backend/tests/test_phase_75_ci_gates.py (16 tests) guarding lane
  config shape; two of the executor's own first-pass mutation SURVIVORS
  (its workflow header comments satisfied its substring guards) were
  caught by its matrix and fixed by anchoring to the actual `run:` lines
  -- the comment-token trap, self-caught.

## Verification (Main-measured, final tree)

- Immutable command: **exit 0** (re-run after every Main edit).
- **CI-equivalent green tail** (3 env overrides at shipped defaults):
  `1466 passed, 0 failed, 2 skipped, 16 deselected, 5 xfailed, 1 xpassed`.
- **Raw local suite**: **9 failed / 1463 passed** -- the pre-step 10-red
  baseline MINUS the fixed test_60_1; the 9 = exactly categories A (4,
  live-state, deselected on CI) + B (5, green on CI defaults). The
  standing local baseline SHRINKS 10 -> 9 and the CI lane is
  deterministically green.
- Ruff: clean over the git-derived scope + new files -- after Main
  removed the executor-disclosed pre-existing F401 in the touched
  23_2_6 file (proven pre-existing via git-show-HEAD lint; the
  touched-file precedent applies -- the executor's out-of-scope caution
  was overruled for consistency with 75.9/75.10/75.14).
- All 5 touched/new workflow YAMLs: yaml.safe_load clean.
- Mutations: executor 7/7 KILLED; Main independently spot-checked
  **M2 (drop -m) KILLED** and **M6 (remove audit step) KILLED**;
  guard suite 16/16 post-restore.
- Operator :3000: read-only curl 200 (executor) -- no server started.

## Not verified live

The enforcing lane + nightly/npm-audit/visual gating first EXERCISE on
the next GitHub push -- npm-audit red is EXPECTED (disclosed above);
e2e-smoke should be green (the CI-equivalent tail is the local proof).
Branch-protection required-checks remain GitHub-admin (operator-owned;
runbook note in the draft).

## Cycle-2 addendum (Q/A cycle-1 CONDITIONAL -- the one violation fixed)

The seed-lane durability guard was VACUOUS (Q/A mutation-proven): its
OR-clause `('blocks the PR' not in s) or ('run_seed_stability' in s)` was
permanently satisfied by a comment token this step itself added. Fixed to
two independent can-fail assertions: the overclaim ABSENT and the honest
re-scoped sentence ("structurally cannot enforce") PRESENT; the guard-file
docstring's protection overclaim reworded. The omitted leg-d mutations now
run and KILL: **M8** (re-introduce 'blocks the PR' with the comment token
still present -- the exact Q/A-proven vacuity, now 1 failed) and **M9**
(strip the honest sentence -- 1 failed); 16/16 green post-restore.
Cycle-2 re-measurement: immutable command exit 0; CI-equivalent tail
identical (1466 passed / 0 failed / 16 deselected).
