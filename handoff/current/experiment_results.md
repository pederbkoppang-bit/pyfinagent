# Experiment results -- Step 75.10 (event-loop hygiene sweep)

Date: 2026-07-24. **Execution model: GENERATE delegated to a Sonnet-4.6
executor per the executor tag + operator directive; Main wrote the
contract, reviewed the diff line-by-line, and independently re-measured
every headline figure below. The executor's run-then-write draft (with
its 5 named deviations) is preserved verbatim at
`handoff/current/experiment_results_75.10_draft.md`.**

## What was built (contract plan steps 1-8; per-file detail in the draft)

- **py-core-05 + crit 2**: get_running_loop() at the 7 live orchestrator
  sites + task_bus.py; DEAD orchestrator:430 deleted; mas_events.py :97
  dead `_lock` removed and :205's `id(asyncio.get_event_loop)` term
  replaced with `uuid4().hex`. Zero `asyncio.get_event_loop` occurrences
  remain in the three criterion-2 files (scan hard-fails on missing paths).
- **py-core-01**: `_forward_remote` = ONE daemon worker draining a queue
  through a shared httpx.Client (was thread-per-event); enqueue never
  blocks the event path; single-worker proven behaviorally.
- **perf-01**: `screen_universe`, `get_sp500_tickers`, `build_sector_map`
  now `await asyncio.to_thread(...)` -- measured pre-step as genuinely
  blocking the API event loop (AsyncIOScheduler runs run_daily_cycle on
  it). **perf-10**: Semaphore(8)-bounded gather around the ALREADY-threaded
  peer-info fetch (serial-latency fix only; the loop-blocking rationale in
  the step text was stale and is not claimed).
- **API routes**: get_dashboard subprocess.runs + session-listing
  to_thread'd (httpx there was already async); p95 .result() to_thread'd
  (timeout=30 pre-existing, kept); get_optimizer_status -> plain def;
  get_all_jobs launchctl probes to_thread'd; get_log_tail -> seek-from-end
  block reader with the cap KEPT at le=10000 (the step-prose le=1000 was a
  behavior change, rejected under the step's own mechanical-only
  boundary); portfolio N+1 -> Semaphore-bounded gather at BOTH call sites
  (list_positions + get_portfolio_performance -- disclosed widening in the
  same named file).
- **api-design-04**: inline screen_universe to_thread'd + run_data_ingestion
  converted to 202-immediate + pollable task-state dict mirroring /run.
- **api-design-09**: NEW shared `backend/utils/asyncio_tasks.py` keep-set +
  `add_done_callback` (discard + flip the site's state dict to error on
  task.exception()) adopted at analysis.py, backtest.py (both tasks),
  paper_trading.py, and the main.py prewarm task (now saved).
- **pysvc-09**: lifespan finally now shuts down BOTH schedulers, cancels+
  awaits the prewarm task under suppress(CancelledError), and stops the
  Slack monitor via `get_slack_monitor()` -- the contract's
  "init_slack_monitor returns it" cite was WRONG (it returns None,
  measured); the executor corrected it and disclosed the deviation.

## Change surface (measured)

`git diff --stat HEAD`: **20 files, 863 insertions(+), 291 deletions(-)**
(13 modified .py + 2 new: `backend/tests/test_phase_75_event_loop.py`
(21 tests), `backend/utils/asyncio_tasks.py`; remainder = handoff
artifacts + the masterplan 75.10.1 queue insert (+21 lines, verified the
only masterplan change) + runtime-daemon appends).

Out-of-worklist changes, all disclosed + reviewed:
1. `backend/tests/test_phase_23_2_14_no_reentrant_locks.py` -- the
   executor's OWN new worker lock tripped the pinned lock-count guard
   (17 -> 18); correctly self-fixed in-step per that test's documented
   "bump + re-audit in same commit" mechanism, with the re-entrancy audit
   paragraph added (own-code consequence, not a discovered pre-existing
   defect -- so fixed here, not queued).
2. Three pre-existing F401 dead imports removed (75.5 touched-file
   precedent), found by MAIN's git-derived-scope lint after the executor's
   "clean" claim (task_bus.py `time` + `Any`, api/mas_events.py
   function-scoped `AgentType`) -- all three proven pre-existing via
   `git show HEAD:` lint; the executor's edits had meanwhile FIXED a 4th
   (api/mas_events.py `asyncio`, unused at HEAD, now used via
   asyncio.to_thread -- attribution corrected per Q/A Note-2; task_bus's
   `asyncio` was already used at HEAD). Third occurrence of the
   executor-lint-scope-goes-stale pattern across 75.9/75.10; the layered
   independent re-derivation caught it each time.

## Criterion-4 boundary proof (Main-verified line-by-line)

The autonomous_loop.py diff (48 +/- lines) contains ONLY execution
plumbing. The one suspicious seam was chased to ground: the new
`yf.Ticker(xx).info or {}` zeros-entry path is BYTE-EQUIVALENT to HEAD
(the `or {}` guard exists at HEAD:641 too); failed fetches are absent
from the lookup in both versions (`entry is not None` filter == old
`except: continue`). All `compute_peer_leadlag_signals` thresholds,
gates, and sizing lines byte-identical. The executor's draft carries the
decision-line diff hunks.

## Verification (ALL figures independently re-measured by Main)

- Immutable command: **21 passed, exit 0** (Main re-run).
- Ruff F821/F401/F811 over the git-derived 13-file scope + 2 new files ->
  **"All checks passed!", exit 0** (after Main's 3 F401 removals; the
  executor itself caught-and-fixed a malformed-path false "All checks
  passed!" in its own first ruff attempt -- disclosed in its draft).
- Full suite (Main re-run): **10 failed / 1391 passed / 12 skipped /
  5 xfailed / 1 xpassed** -- fail set BYTE-IDENTICAL to the measured
  baseline (comm symmetric diff EMPTY); 1391 = 1370 + exactly the 21 new
  tests. Zero regressions.
- Mutation matrix: executor **8/8 KILLED** (incl. the two STUB mutations:
  renamed-route hard-fail and neutered exception fixture). Main
  independently spot-checked **M3 (un-thread screen_universe) KILLED** and
  **M4 (neuter done-callback) KILLED**; suite green post-restore.

## Not verified live

- The running backend still executes the OLD code -- the threading/lifespan
  changes land on the next operator restart. No live cycle was run; no
  backend restart performed (operator-owned process). No UI surface.
  Flag-gated paths (peer_leadlag etc., default OFF) are output-identical
  wraps -- a $0 no-op by construction, ON or OFF.

## Out of scope -> queued

- **75.10.1** (queued at contract time): the 10th get_event_loop site,
  ticket_queue_processor.py:423, outside criterion-2's 3-file scope.
