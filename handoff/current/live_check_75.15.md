# live_check -- Step 75.15 (qa-tests-01/02/04 + seed/visual/vitest/npm-audit legs)

Date: 2026-07-24. Verbatim captures; rc=$? discipline.

## 1. Immutable verification command (exit 0)

```
immutable_exit=0        (Main re-run after every edit incl. the 23_2_15 correction)
```

## 2. THE deliverable evidence -- CI-equivalent green tail (shipped defaults)

```
$ PAPER_DATA_INTEGRITY_ENABLED=false PAPER_RISK_JUDGE_REJECT_BINDING=false PAPER_SWAP_CHURN_FIX_ENABLED=false \
  .venv/bin/python -m pytest backend/tests/ -q --no-header -m "not requires_live"
1466 passed, 2 skipped, 16 deselected, 5 xfailed, 1 xpassed, 1 warning in 91.61s (0:01:31)
```
Collection pin (guarded by test_phase_75_ci_gates.py): 1474/1490 (16 deselected).

## 3. Raw local suite (operator machine, no overrides) -- the honest delta

```
9 failed, 1463 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 93.16s (0:01:33)
```
= pre-step baseline (10) MINUS fixed test_60_1. The 9: four requires_live-marked
live-state tests (23_2_10/15/6/9 -- deselected on CI) + five operator-.env
pollution reds (57_1 x3, 60_3, portfolio_swap -- GREEN at CI defaults, proven
by the section-2 tail).

## 4. Main-review correction transcript (executor claim that did not reproduce)

Executor: "23_2_15 fixed, no mark needed; 0-fail tail". Main re-run: 23_2_15
RED (6 of 8 shelled verify scripts fail; root cause: bare `python` invocation
-- PATH-dependent across shells + live-machine probes). Resolution: marked
requires_live per the research's original category-A classification; the
executor's sub-script mock fix KEPT (real 75.9-singleton mock drift); the
executor's own collection-count guard caught the marker change (red until the
pin moved to 16 deselected) -- the guard working as designed.

## 5. Workflows + boundary

All 5 touched/new YAMLs yaml.safe_load clean. npm audit currently exits 1
locally (42 vulns: 19 high, 3 critical -- transitives) -- the new lane WILL
red on first CI run, DISCLOSED as the honest signal. Operator :3000 untouched
(read-only curl 200; no server started at any point). Mutations: 7/7 executor
+ Main spot-checks M2/M6 KILLED; coverage checker can-fail proven (99%-bar
mutation -> exit 1; missing-input -> exit 2, never silent).
