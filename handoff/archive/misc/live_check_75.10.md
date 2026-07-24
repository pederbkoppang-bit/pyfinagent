# live_check -- Step 75.10 (perf-01/10, api-design-01/02/04/05/06/08/09, py-core-01/05, pysvc-09)

Date: 2026-07-24. Verbatim captures; exit codes via `rc=$?` immediately
after the command.

## 1. Immutable verification command (exit 0) -- Main's independent re-run

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_event_loop.py -q
.....................                                                    [100%]
21 passed in 0.93s
pytest_exit=0
```

## 2. Change surface

```
$ git diff --stat HEAD | tail -1
 20 files changed, 863 insertions(+), 291 deletions(-)
```
13 modified .py + 2 new (`backend/tests/test_phase_75_event_loop.py`,
`backend/utils/asyncio_tasks.py`). Masterplan diff = ONLY the 75.10.1
queue insert (+21 lines, verified). Full per-file stat in
experiment_results_75.10_draft.md.

## 3. Lint over the git-derived scope (exit 0)

```
$ files=$(git diff --name-only HEAD -- '*.py'); echo "scope_files=$(echo "$files" | grep -c .)"
scope_files=13
$ uvx ruff check --select F821,F401,F811 <the 13 files> backend/tests/test_phase_75_event_loop.py backend/utils/asyncio_tasks.py
All checks passed!
ruff_exit=0
```

## 4. Full-suite regression (Main's independent re-run, symmetric diff)

```
10 failed, 1391 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 91.02s (0:01:31)
FAIL SET IDENTICAL TO 75.9 BASELINE   (comm diff of sorted FAILED lines: empty)
```

## 5. Criterion-4 decision-line proof

`git diff HEAD -- backend/services/autonomous_loop.py` = 48 +/- lines,
all execution plumbing (to_thread wraps + Semaphore-bounded gather). The
`compute_peer_leadlag_signals(...)` call, every threshold getattr, and
all gate/sizing lines are byte-identical. The apparent `or {}` semantic
seam was verified BYTE-EQUIVALENT to HEAD:641 (guard pre-existed). The
executor draft carries the diff hunks.

## 6. Flag-gated / UI note

Flag-gated paths touched (peer_leadlag etc.) default OFF and the wraps
are output-identical ON or OFF ($0 no-op by construction -- same values,
same dict shapes, same skip semantics; proven by the crit-4 behavioral
test comparing wrapped vs unwrapped returns under mocks). The running
backend still executes the OLD code until the next operator restart. No
UI surface -> no Playwright capture.
