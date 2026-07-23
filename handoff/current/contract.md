# Contract -- Step 75.10: event-loop hygiene sweep (to_thread blocking paths, get_running_loop, task refs, lifespan drain)

- **Step id**: 75.10 (phase-75, Audit75 S10) -- P1, executor sonnet-tier
- **Date**: 2026-07-23
- **Author**: Main (contract + review). **GENERATE delegated to a Sonnet-4.6 executor agent** per the executor tag + operator session directive (same model as 75.9: Main reviews the diff and independently re-measures before Q/A).
- **BOUNDARY (from step text)**: mechanical execution changes ONLY -- zero decision/threshold changes in the live loop.

## Research-gate summary (gate PASSED)

Workflow `wf_4dffaba2-d81` (researcher, opus/max, tier=complex).
Envelope: `external_sources_read_in_full=6 (all official docs: asyncio-eventloop, asyncio-task, fastapi/async, apscheduler, asyncio-sync, httpx/async), snippet_only=11, urls=17, recency_scan=true, internal_files=15, gate_passed=true`.
Brief: `handoff/current/research_brief_75.10.md`.

**Step-text corrections adopted (binding -- four findings are PARTIALLY ALREADY FIXED; stale audit anchors):**
1. **perf-10 already threaded**: autonomous_loop.py:641 is ALREADY `await asyncio.to_thread(lambda x=t: yf.Ticker(x).info)` (since 2026-05-18, commit 6ceeb10ff). It does NOT block the loop; the remaining defect is SERIAL latency only. Fix = Semaphore(8)-bounded gather; the "blocks the event loop" rationale must NOT be claimed.
2. **api-design-01**: get_dashboard's httpx is ALREADY AsyncClient/awaited (:116/:122). Real blockers = 2x subprocess.run (:146 timeout=10, :153 timeout=15, ~25s worst-case not ~35s) + list_openclaw_sessions() (:173). Only those need to_thread.
3. **api-design-02**: the p95 query ALREADY carries `.result(timeout=30)` (:82) -- criterion-3's timeout clause is already satisfied; remaining work = to_thread the sync call.
4. **api-design-04**: run_full_ingestion is ALREADY to_thread'd (:200). Remaining = to_thread the inline `screen_universe()` (:195) + convert to 202-immediate + pollable task-state dict mirroring run_backtest (:112-135).
5. **get_log_tail cap**: actual is `Query(200, ge=1, le=10000)` (cron_dashboard_api.py:533). The step-prose "le=1000" would REJECT 1001-10000 -- a behavior change that violates this step's own mechanical-only boundary. **Decision: keep le=10000** (the immutable criteria do not mention the cap; only the seek-from-end reader is required).
6. **Incident cite drift**: the 2026-05-25 misfire-grace comment is paper_trading.py:1301-1311.
7. **A 10th get_event_loop site** exists at ticket_queue_processor.py:423, OUTSIDE criterion-2's 3-file scope -> queued as step **75.10.1**, not folded in.
8. **mas_events.py has TWO occurrences**: :97 (the crash path) AND :205 (`id(asyncio.get_event_loop)` inside make_run_id -- matches the criterion-2 grep without calling). Both must go (replace :205's term with `uuid4().hex`).
9. **orchestrator:430 is DEAD** (`loop` never used in _execute_full_flow 415-544) -- DELETE, don't convert.
10. **A 4th unsaved fire-and-forget**: the prewarm `asyncio.create_task(_prewarm_ticker_meta())` at main.py:406 (return discarded -- both a pysvc-09 cancel target and an api-design-09 GC hazard).

**Key measured findings:**
- Python 3.14 official docs: get_event_loop() now "Raises a RuntimeError if there is no current event loop" -- confirms the MASEventBus:97 crash from sync callers (slack_bot app_home.py:41/455/543/559).
- perf-01 context MEASURED: both schedulers are AsyncIOScheduler (main.py:301/:348) -> run_daily_cycle (async, autonomous_loop.py:252) runs ON the API event loop; screen_universe (:576 -> screener.py:137, yf.download ~500 tickers/6mo) genuinely blocks it. to_thread is warranted, output-identical.
- MASEventBus._lock is DEAD (no in-file reader) -- delete. _forward_remote = thread-per-event with fresh httpx import; single queue + daemon worker + shared client is strictly better (FIFO, connection reuse); no consumer relies on thread-per-event.
- lifespan finally (main.py:410-419) shuts down ONLY queue_scheduler. Missing: paper scheduler (:301), prewarm task (:406), Slack monitor -- which DOES expose sync `stop()` (slack_monitor.py:88; init returns it at :105 but main.py:333 discards the return).
- py-core-05: all 8 orchestrator sites are inside async def (get_running_loop safe; 7 feed run_in_executor); task_bus.py:140 same.
- Fire-and-forget inventory for criterion 5: analysis.py:364, backtest.py:134, paper_trading.py:1023 (+ backtest.py:299 already saved -- add error-callback parity; + main.py:406 prewarm).
- Test conventions: pytest-asyncio NOT installed; anyio only; `asyncio.run()` is the idiom (17 files); no conftest loop fixture. Criterion-1's no-running-loop construction = plain def test body (NOT wrapped in asyncio.run).
- Live-flag context: peer_leadlag/sector flags default False; paper_markets live = US+EU+KR so get_sp500_tickers IS live-executed -- wraps are output-identical ON or OFF.

## Hypothesis

Every blocking call can be moved off the event loop and every event-loop-API misuse corrected as pure execution-plumbing -- byte-identical decision/gate/threshold logic, identical outputs, provable offline by a test file that constructs MASEventBus without a loop, drives a task to exception, and AST-verifies the route bodies -- with zero behavior change visible to the trading loop or API consumers (202-conversion of run_data_ingestion being the single documented API-semantics change, mirroring the existing /run pattern).

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.10)

verification.command:
```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_event_loop.py -q
```

1. "New backend/tests/test_phase_75_event_loop.py passes offline and constructs MASEventBus() in a context with NO running event loop without raising (the Python-3.14 crash repro), and asserts _forward_remote no longer spawns a Thread per event (single worker/queue -- source or behavioral assert)"
2. "Scan asserts zero asyncio.get_event_loop occurrences in backend/agents/multi_agent_orchestrator.py, backend/agents/task_bus.py, backend/agents/mas_events.py"
3. "AST assert: the named async route bodies (get_dashboard, get_llm_p95_latency, run_data_ingestion, portfolio enrichment paths, get_log_tail, get_all_jobs) contain no direct subprocess.run/sync-httpx/.result( calls outside an asyncio.to_thread wrapper, and get_optimizer_status is a plain def or fully to_thread-wrapped; the p95 query carries timeout=30"
4. "services/autonomous_loop.py contains 'await asyncio.to_thread(screen_universe' and the flag-gated universe/sector/peer-leadlag fetches are threaded/gathered -- with the cycle's decision logic diff-verified unchanged (no edits to gate/sizing/threshold lines; diff file list in experiment_results.md)"
5. "Backtest, analysis, and paper-cycle create_task results are stored with add_done_callback error propagation into their state dicts (source assert per site); a test drives one task to exception and sees the state flip to error"
6. "main.py lifespan finally shuts down both schedulers and cancels the prewarm task (source assert), and run_data_ingestion returns 202-immediately semantics with a pollable status (test with mocked ingestion)"

verification.live_check: "handoff/current/live_check_75.10.md: verbatim output of this step's verification command (exit 0) + git diff --stat proving the change surface; for any flag-gated live-loop behavior an ON-vs-OFF $0 diff, and for UI-touching parts a Playwright/curl capture. Findings covered: perf-01, perf-10, api-design-01, api-design-02, api-design-04, api-design-05, api-design-06, api-design-08, api-design-09, py-core-01, py-core-05, pysvc-09"

## Plan steps

1. **py-core-05 + criterion 2**: get_running_loop() at the 7 live orchestrator sites (566,586,690,723,738,862,1065) + task_bus.py:140; DELETE dead orchestrator:430. mas_events.py: fix :97 (delete dead _lock outright) AND :205 (`uuid4().hex` replaces the id() term). Grep reaches zero in all 3 files.
2. **py-core-01**: _forward_remote -> single daemon worker draining a queue.Queue through one shared httpx.Client (FIFO preserved, connection reuse). Behavioral single-worker assert + source assert.
3. **perf-01**: `await asyncio.to_thread(screen_universe, ...)` at autonomous_loop.py:576 + to_thread get_sp500_tickers (:524) + build_sector_map (:572). **perf-10**: Semaphore(8)-bounded gather around the EXISTING to_thread call (latency fix only -- no loop-blocking claim).
4. **API routes**: get_dashboard -> to_thread the 2 subprocess.run + list_openclaw_sessions; get_llm_p95_latency -> to_thread the .result() (timeout=30 already there); get_optimizer_status -> plain def (no awaits in body); get_all_jobs -> to_thread launchctl probes; get_log_tail -> seek-from-end block reader, **cap stays le=10000**; _enrich_position -> Semaphore-bounded gather of to_thread.
5. **api-design-04**: to_thread screen_universe (:195) + 202-immediate + task-state dict + pollable status mirroring run_backtest (:112-135).
6. **api-design-09**: keep-set + add_done_callback (discard + flip state to error on task.exception()) at analysis.py:364, backtest.py:134, paper_trading.py:1023; error-callback parity on backtest.py:299; save + later cancel the prewarm task (main.py:406).
7. **pysvc-09**: lifespan finally adds paper `scheduler.shutdown(wait=False)`, cancel+await prewarm under suppress(CancelledError), capture the Slack monitor from init_slack_monitor (main.py:333) and call its sync stop().
8. **Tests** (plain def + asyncio.run idiom; NO pytest-asyncio): criterion-1 constructs MASEventBus() in a plain def body; criterion-3 AST hard-fails if any named route node is missing/renamed + allow-lists awaited AsyncClient; criterion-5 drives a real task to exception (asyncio.run) and asserts the state dict flips.
9. **Mutation matrix** (executor runs; Main spot-checks): revert :97 to get_event_loop; restore thread-per-event; un-thread screen_universe; drop one done-callback; remove the paper-scheduler shutdown line; break the 202 semantics; point the criterion-3 AST at a renamed route (must hard-fail); mutate the test stub (loop-fixture injection) where applicable.
10. **Queue 75.10.1** (ticket_queue_processor.py:423 get_event_loop, same 3.14 class) -- own research-gated step.
11. **live_check_75.10.md**: verbatim pytest exit 0 + git diff --stat + the criterion-4 decision-line diff proof (no gate/sizing/threshold edits). No UI capture (no UI surface); flag-gated paths are output-identical wraps (documented as $0 no-op by construction).

## Explicitly NOT in scope

- ticket_queue_processor.py:423 (queued 75.10.1)
- get_log_tail cap change to le=1000 (behavior change; violates the mechanical-only boundary)
- Any decision/gate/threshold line in autonomous_loop.py (byte-identical; enumerated in the brief for diff-verification)
- Client-construction or BQ changes beyond the named to_thread wraps (75.9 territory)

## References

- `handoff/current/research_brief_75.10.md` (6 official-doc sources read in full; Python 3.14 whatsnew/asyncio docs; FastAPI async guidance; APScheduler shutdown; create_task save-a-reference guidance)
- `handoff/current/audit_phase75/confirmed_findings.json` (the 12 findings)
- CLAUDE.md Harness Protocol; feedback_queue_discovered_defects_in_masterplan; feedback_mutation_test_guards_and_fixtures
