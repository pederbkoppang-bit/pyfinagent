# Experiment Results (DRAFT) — Step 75.10: event-loop hygiene sweep

**Executor**: Sonnet GENERATE agent (per contract's executor tag). This file
reports ONLY measured figures from commands actually run in this session.
Main/Q-A: please independently re-run the verification command and the
mutation script before certifying.

## What was built / changed (file list)

13 files modified + 2 new files. Full `git diff --stat` for the intentional
change set (isolated from ambient session/hook noise on `.claude/masterplan.json`,
`handoff/audit/*`, `handoff/current/contract.md`, `handoff/current/research_brief.md`
etc. -- those are pre-existing/concurrent modifications this session did NOT
author; I never called Write/Edit on any of them):

```
 backend/agents/mas_events.py                       |  59 +++++++++--
 backend/agents/multi_agent_orchestrator.py         |  15 ++-
 backend/agents/task_bus.py                         |   2 +-
 backend/api/analysis.py                            |  16 ++-
 backend/api/backtest.py                            | 108 +++++++++++++++++++--
 backend/api/cron_dashboard_api.py                  |  43 +++++++-
 backend/api/mas_events.py                          |  10 +-
 backend/api/paper_trading.py                       |  11 ++-
 backend/api/performance_api.py                     |   4 +-
 backend/api/portfolio.py                           |  31 ++++--
 backend/main.py                                    |  42 +++++++-
 backend/services/autonomous_loop.py                |  46 ++++++---
 backend/tests/test_phase_23_2_14_no_reentrant_locks.py |  15 ++-
 13 files changed, 341 insertions(+), 61 deletions(-)

New (untracked):
 backend/tests/test_phase_75_event_loop.py   (21 tests)
 backend/utils/asyncio_tasks.py              (shared track_task() helper)
```

### Per-file summary

- **`backend/agents/multi_agent_orchestrator.py`** (py-core-05): deleted the
  DEAD `loop = asyncio.get_event_loop()` at the top of `_execute_full_flow`
  (verified unused across the whole 415-544 method body -- no `loop.`
  reference anywhere in it). Replaced the remaining 7
  `asyncio.get_event_loop()` sites (all feed `loop.run_in_executor`, all
  inside `async def`) with `asyncio.get_running_loop()`.
- **`backend/agents/task_bus.py`** (py-core-05): `asyncio.get_event_loop().create_future()`
  -> `asyncio.get_running_loop().create_future()` at the `delegate()` site.
- **`backend/agents/mas_events.py`** (py-core-01): deleted the dead
  `self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None`
  (no in-file reader). Replaced `_forward_remote`'s thread-per-event pattern
  with a lazily-started single daemon worker (`_ensure_remote_worker`, guarded
  by a `threading.Lock` double-checked-locking init) draining a `queue.Queue`
  through one shared `httpx.Client`. Replaced `make_run_id`'s meaningless
  `id(asyncio.get_event_loop)` term with `uuid.uuid4().hex`.
- **`backend/services/autonomous_loop.py`** (perf-01, perf-10): `get_sp500_tickers()`,
  `build_sector_map(universe)`, and `screen_universe(...)` are now
  `await asyncio.to_thread(...)`-wrapped with byte-identical kwargs. The
  peer_leadlag per-ticker loop (already `to_thread`'d since 2026-05-18,
  commit `6ceeb10ff` -- I did NOT claim it "unblocks the loop", only fixed
  the serial-latency defect) is now a `Semaphore(8)`-bounded `asyncio.gather`
  preserving the exact per-ticker try/except/continue-on-failure semantics
  and the exact `lookup` dict shape.
- **`backend/api/mas_events.py`** (api-design-01): `get_dashboard`'s 2x
  `subprocess.run` + `list_openclaw_sessions()` are now `asyncio.to_thread`-wrapped.
  The async httpx health checks were already compliant (untouched).
- **`backend/api/performance_api.py`** (api-design-02): `get_llm_p95_latency`'s
  `bq.client.query(sql).result(timeout=30)` is now
  `await asyncio.to_thread(lambda: list(...))`; `timeout=30` unchanged.
- **`backend/api/backtest.py`** (api-design-04, api-design-06, api-design-09):
  - `run_data_ingestion` is now 202-immediate: creates `_ingestion_state`,
    launches `_run_ingestion_async` as a tracked background task, returns
    `{"status": "started", "run_id": ...}` immediately. New
    `GET /api/backtest/ingest/progress` polls `_ingestion_state` (kept
    distinct from the pre-existing `GET /ingest/status`, which reports
    historical-table row counts, not task progress -- do not conflate, per
    the contract's own warning).
  - `_run_ingestion_async` (new): `screen_universe()` is now
    `await asyncio.to_thread(screen_universe)`; rest mirrors the original
    body verbatim (same tickers-empty fallback list, same `run_full_ingestion`
    call/kwargs), with self-contained try/except flipping `_ingestion_state`.
  - `get_optimizer_status`: `async def` -> plain `def` (confirmed zero
    `await` in its body before the edit; FastAPI auto-threadpools plain-def
    routes per `fastapi.tiangolo.com/async/`).
  - `run_backtest`'s existing `asyncio.create_task(_run_backtest_async(...))`
    and the new ingestion task are now registered through the shared
    `track_task()` keep-set helper (api-design-09). The already-saved
    `_optimizer_task` (:299, single module ref, no keep-set needed) got
    error-callback parity via a new `_on_optimizer_task_done` done-callback.
- **`backend/api/portfolio.py`** (api-design-05): `list_positions` and
  `get_portfolio_performance`'s N+1 sequential `_enrich_position` calls are
  now `Semaphore(8)`-bounded `asyncio.gather` of `to_thread`-wrapped calls via
  a new `_enrich_position_async` helper. `add_position`'s single-call site
  (not N+1) is unchanged.
- **`backend/api/cron_dashboard_api.py`** (api-design-08): `get_log_tail`'s
  whole-file `deque(f, maxlen=n)` scan replaced with a new `_tail_lines()`
  seek-from-end block reader (reads backwards in 64KB chunks until enough
  newlines are seen). **Query cap kept at `le=10000`** per the contract's
  own correction (lowering to `le=1000` would reject 1001-10000 requests
  that succeed today -- a behavior change out of the mechanical-only
  boundary). Removed the now-unused `from collections import deque` import.
  `get_all_jobs`'s per-entry `_launchctl_state(...)` probes are now
  `asyncio.gather`'d over `to_thread`-wrapped calls.
- **`backend/api/analysis.py`** (api-design-09): the `_run_sync_analysis`
  fire-and-forget task is now tracked via `track_task()` with an
  `_on_error` callback that flips `_tasks[task_id]` to `AnalysisStatus.FAILED`
  if an exception somehow escapes the coroutine's own internal try/except
  (which already self-catches everything -- this is defense-in-depth, not
  the primary error path).
- **`backend/api/paper_trading.py`** (api-design-09): `_run_cycle_background`'s
  fire-and-forget task is now tracked via `track_task()`; its `_on_error`
  callback sets `_last_cycle_error` (same defense-in-depth framing as
  analysis.py -- `_run_cycle_background` also already self-catches).
- **`backend/utils/asyncio_tasks.py`** (NEW, api-design-09): shared
  `track_task(task, tasks, on_error, label)` helper implementing the official
  asyncio "save a reference to tasks" pattern (keep-set + discard-on-done)
  plus exception propagation via a caller-supplied `on_error` callback
  (deliberately not dict-shape-specific, since analysis.py/backtest.py/
  paper_trading.py each have a different state shape -- an enum, two
  different dicts, and a bare module string respectively).
- **`backend/main.py`** (pysvc-09): lifespan `finally` now also shuts down
  the paper-trading `scheduler` (`if 'scheduler' in locals(): scheduler.shutdown(wait=False)`,
  mirroring the existing `queue_scheduler` guard), cancels + awaits the
  prewarm task (now saved as `prewarm_task = asyncio.create_task(...)`,
  previously discarded) under `contextlib.suppress(asyncio.CancelledError)`,
  and stops the Slack monitor via `get_slack_monitor()`.
- **`backend/tests/test_phase_23_2_14_no_reentrant_locks.py`** (out-of-scope
  fix, see Deviations below): bumped `EXPECTED_LOCK_COUNT` 17 -> 18 +
  documented audit paragraph for the one new lock this step introduces
  (`mas_events.py`'s `_remote_worker_lock`).

## Verbatim verification command output

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_event_loop.py -q
.....................                                                    [100%]
21 passed in 0.93s
```
Exit code: **0**

## Ruff (F821, F401, F811) over the complete git-derived scope

Scope re-derived AFTER the last edit (per the 75.9 lesson about stale
hand-typed lists), via:
```
files=$(git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py')
```
**Scope size: 15 files** (14 touched .py files + the new test file; verified
non-empty). Ran via `xargs` (NOT bare `$files` interpolation -- this shell is
zsh, which does not word-split an unquoted variable the way bash does; a
first naive attempt silently passed a single malformed multi-line path to
ruff and printed a false "All checks passed!" -- caught by re-running with
`xargs` and getting real per-file diagnostics instead).

```
$ xargs uvx ruff check --select F821,F401,F811 < scope.txt
F401 backend/agents/task_bus.py:22   `time` imported but unused
F401 backend/agents/task_bus.py:26   `typing.Any` imported but unused
F401 backend/api/mas_events.py:95    `backend.agents.agent_definitions.AgentType` imported but unused
Found 3 errors.
```
Exit code: **1** (3 findings)

**All 3 are pre-existing, verified via `git show HEAD:<file>`** — I never
touched the import lines in either file (my task_bus.py edit is a single
line inside `delegate()`; my mas_events.py edit only touches the
`subprocess.run` and `list_openclaw_sessions()` call sites, not the
`AgentType` import a few lines above). Confirmed by direct diff inspection
before writing this. Zero NEW ruff findings introduced by this step's
changes. (My own new test file initially had one genuinely-mine F401 --
an unused `AsyncMock` import -- which I fixed; it's not in the 3 above.)

## AST sanity (all touched + new files)

`ast.parse()` clean on all 15 files in scope (looped, one call per file):
all printed `OK`, zero `FAIL`.

## Full-suite regression comparison

```
$ .venv/bin/python -m pytest backend/tests/ -q --no-header
10 failed, 1391 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 90.27s
```

Failures match the **exact named baseline set, no more no less**:
`test_phase_23_2_10_watchdog_no_fire_7d.py::test_phase_23_2_10_watchdog_log_present_and_fresh`,
`test_phase_23_2_15_verify_23_1_smoke.py::test_phase_23_2_15_known_pass_scripts_still_pass`,
`test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence`,
`test_phase_23_2_9_ticker_meta_latency.py::test_phase_23_2_9_backend_log_has_prewarm_evidence`,
`test_phase_57_1_reject_binding.py` x3,
`test_phase_60_1_deep_pipeline.py::test_60_1_claude_code_rail_declares_latency_profile`,
`test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off`,
`test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap`.

**One NEW failure surfaced on the first full-suite run and was fixed before
this handoff** (see Deviations #1 below):
`test_phase_23_2_14_no_reentrant_locks.py::test_phase_23_2_14_threading_lock_count_matches_roster`
went from 17 -> 18 because of the new `_remote_worker_lock` in
`mas_events.py`. After the sanctioned re-audit-and-bump (the test's own
docstring prescribes exactly this flow, already used 3x in its history),
the full suite re-ran clean at the named-baseline-only 10 failures.

## Criterion-4 decision-line proof (no gate/sizing/threshold edits)

`git diff backend/services/autonomous_loop.py` (full, reproduced above in
this file) shows exactly 4 hunks, none touching a gate/sizing/threshold
line:
1. `get_sp500_tickers()` call site -- wrap only.
2. `build_sector_map(universe)` call site -- wrap only.
3. `screen_universe(...)` call site -- wrap only, all 5 kwargs byte-identical
   (`tickers=universe, period="6mo", sector_lookup=_sector_lookup,
   short_interest_lookup=short_interest_lookup or None,
   short_interest_threshold=getattr(settings, "short_interest_threshold", 0.10)`).
4. peer_leadlag per-ticker fetch loop -- restructured for bounded
   concurrency; the DOWNSTREAM `compute_peer_leadlag_signals(...)` call
   (leader_threshold / laggard_threshold / max_analyst_count /
   min_market_cap_usd / boost -- all the actual decision thresholds) is
   **untouched**, confirmed by a dedicated test
   (`test_decision_lines_untouched_boundary_markers_present`) asserting
   all 6 `getattr(settings, "...", <value>)` threshold markers are present
   verbatim in the file.

## Mutation matrix (8/8 killed)

Script: `/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/0a35ec0b-2832-4744-a9ae-fab6b46f19bb/scratchpad/mutation_matrix_75_10.py`
(pattern mirrors `mutation_matrix_75_9.py`: exact-one-occurrence assert,
apply, run immutable test, record, byte-exact restore + verify).

| # | Mutation | Killed |
|---|---|---|
| M1 | Revert one `get_running_loop()` -> `get_event_loop()` (orchestrator `_think_plan`) | YES (2 failed) |
| M2 | Restore thread-per-event `_forward_remote` | YES (2 failed) |
| M3 | Un-thread `screen_universe` (remove the `to_thread` wrap) | YES (2 failed) |
| M4 | Neuter `track_task`'s `add_done_callback` wiring | YES (2 failed) |
| M5 | Remove the paper-scheduler shutdown line in `main.py` | YES (1 failed) |
| M6 | Break `run_data_ingestion`'s 202 (restore synchronous await-inline completion) | YES (2 failed) |
| M7 | STUB: point crit-3 route lookup at a renamed route in the TEST file | YES (1 failed -- hard error, not skip-green) |
| M8 | STUB: neuter the crit-5 exception fixture so the task succeeds | YES (1 failed -- proves the fixture can represent the failure) |

**8/8 killed, 0 survivors.** All files verified restored byte-exact
(`git status --short` identical before/after; the script's own
`assert path.read_text(...) == backup` also passed for every mutation, or
the script itself would have raised).

Ran the full matrix twice (once mid-session, once as the final
confirmation pass after the phase-23.2.14 fix) with identical 8/8 results
both times.

## NOT verified live

Everything in this step is verified **offline only**: unit/AST/source-scan
tests + mocked ingestion. Nothing here was run against the live paper-trading
loop, a real BigQuery client, real yfinance, or a real `subprocess`/`launchctl`
call. The contract's `live_check_75.10.md` (verbatim pytest output +
`git diff --stat` + the criterion-4 proof + flag ON/OFF diff) is Main's
artifact to produce after independent re-verification, not something I
(the GENERATE executor) am authorized to write per my instructions.

The peer_leadlag / `get_sp500_tickers` / `build_sector_map` flag-gated paths
are asserted to be **output-identical by construction** (the wrap only
moves WHERE the call executes, never changes args or return value) and
covered by the `test_screen_universe_wrap_runs_off_main_thread_and_preserves_kwargs_and_return`
behavioral proof for the primary `screen_universe` call, but no live
ON-vs-OFF cycle was actually run against real yfinance data in this session
-- that's the kind of evidence `live_check_75.10.md` should carry.

## Deviations from the contract (every one named explicitly)

1. **Fixed an out-of-scope regression I caused**: my new
   `_remote_worker_lock` in `mas_events.py` bumped the real
   `threading.Lock()` count under `backend/` from 17 to 18, breaking
   `test_phase_23_2_14_no_reentrant_locks.py`'s pinned-count regression
   guard on the first full-suite run. This is exactly the class of
   "discovered defect while working a step" the operator's queuing
   doctrine cares about, EXCEPT it's not a pre-existing defect I found --
   it's a consequence of MY OWN new code, so the correct action is to fix
   it in this same step (not queue it as a separate step) per the test's
   own documented mechanism ("Re-audit required + bump EXPECTED_LOCK_COUNT
   in same commit," already exercised 3x in that file's history for
   exactly this scenario). I audited the new lock against the file's own
   phase-23.2.14 re-entrancy criteria (single-acquire, never nested, no
   `_*_locked` helper re-acquires it -- confirmed clean), bumped the count
   to 18, and added a documented audit paragraph in the same style as the
   existing 16th/17th entries. Full suite now shows only the named
   10-test baseline.

2. **Slack-monitor capture uses `get_slack_monitor()`, not "capture the
   return value of `init_slack_monitor`"** as the contract's step-text
   literally describes. Measured: `init_slack_monitor` (slack_monitor.py:96-101)
   does NOT return the monitor instance (implicit `None` return) -- it's
   `get_slack_monitor()` (slack_monitor.py:103-105) that returns `_monitor`.
   The contract's own citation ("init_slack_monitor RETURNS the monitor,
   slack_monitor.py:105") conflated the two adjacent functions. I used the
   accessor that actually exposes the instance rather than literally
   "capturing" a `None` return, which would not have worked. Functionally
   achieves the same shutdown goal the contract asked for. NOT a criterion-6
   blocker -- criterion 6 only requires "shuts down both schedulers and
   cancels the prewarm task" (source assert); the Slack-monitor stop is a
   pysvc-09 nice-to-have from the plan steps / step-text, not an immutable
   criterion, so this deviation cannot fail the gate either way, but I'm
   naming it per "measure, don't assert."

3. **`get_log_tail`'s new `_tail_lines()` reader is NOT wrapped in
   `asyncio.to_thread`.** The contract's plan step 4 describes only "seek-
   from-end block reader... cap STAYS le=10000" with no to_thread mention,
   and criterion 3's AST scan explicitly covers only subprocess.run/sync-
   httpx/.result( -- file I/O isn't in that category. I judged wrapping it
   would be unrequested scope creep on a mechanical-only step and left it
   as a direct synchronous call (matching this file's existing convention
   of inline sync file reads in other routes, e.g. `_load_plist`). If Q/A
   or the operator wants it threaded too, that's a one-line follow-up.

4. **`list_positions` was also fixed alongside `get_portfolio_performance`**
   in `portfolio.py`, even though the contract's plan step 4 names only
   "_enrich_position N+1" without specifying both call sites. Both
   `list_positions` (:60) and `get_portfolio_performance` (:114-115) had the
   identical N+1 serial-blocking pattern; fixing only one would have left
   the other one half-fixed for no principled reason, so I applied the same
   `_enrich_position_async` wrapper to both. `add_position`'s single-call
   site (not N+1) was deliberately left untouched.

5. **A new shared helper module was added** (`backend/utils/asyncio_tasks.py`)
   rather than duplicating the keep-set + done-callback logic 3x across
   analysis.py/backtest.py/paper_trading.py. The contract's plan step 6 says
   "keep-set or module ref + done-callback... at analysis.py:364,
   backtest.py:134, paper_trading.py:1023" without mandating a shared vs.
   per-file implementation; a shared, independently-testable helper seemed
   like the better-engineered choice and is what the criterion-5 behavioral
   test in `test_phase_75_event_loop.py` directly exercises.

## Not incomplete / no other known gaps

All 6 immutable criteria have both a source-level assert and (where the
criterion text says "behavioral") a behavioral test in
`backend/tests/test_phase_75_event_loop.py`. `ticket_queue_processor.py:423`
(the 10th `get_event_loop` site) was correctly left untouched per the
contract's explicit "queued as 75.10.1" instruction -- not fixed here,
not silently expanded into this step's surface.
