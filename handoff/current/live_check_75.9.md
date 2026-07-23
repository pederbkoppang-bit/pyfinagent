# live_check -- Step 75.9 (data-bq-01/02/03/06, py-core-03, gap3-08, gap6-09, perf-11)

Date: 2026-07-23. Verbatim captures; exit codes via `rc=$?` immediately
after the command (never through a pipe).

## 1. Immutable verification command (exit 0) -- Main's independent re-run

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_bq_discipline.py -q
.............................................                            [100%]
45 passed in 1.47s
pytest_exit=0
```

## 2. Change surface

```
$ git diff --stat HEAD | tail -1
 38 files changed, 1091 insertions(+), 501 deletions(-)
```
(30 .py code/test files; the remainder are this step's handoff artifacts +
the masterplan 75.9.1 queue insert. Full per-file stat in
experiment_results_75.9_draft.md section 3.)

## 3. Lint over the git-derived scope (exit 0)

```
$ files=$(git diff --name-only HEAD -- '*.py'); n=$(echo "$files" | grep -c .); echo "scope_files=$n"
scope_files=30
$ uvx ruff check --select F821,F401,F811 <the 30 files> backend/tests/test_phase_75_bq_discipline.py
All checks passed!
ruff_exit=0
```

## 4. Full-suite regression (Main's independent re-run)

```
10 failed, 1370 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 90.89s (0:01:30)
```
The 10-test FAILED set is byte-identical to the pre-75.9 measured baseline
(the standing live-environment red set); 1370 = 1325 + the 45 new tests.

## 5. Flag-gated / UI note

No flag-gated live-loop behavior introduced (fail-closed dedup, query
parameters, client-side timeouts, and the maximum_bytes_billed cap are
unconditional mechanical hardening; the cost cap fails a >5GiB-estimate
query BEFORE any charge -- inert on every normal query in this repo, all of
which are partition/LIMIT-bounded). No live BQ query was executed in this
step (offline mocks only). No UI surface -> no Playwright capture.
