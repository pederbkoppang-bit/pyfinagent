# Research Brief — Step 75.10 (event-loop hygiene sweep)

**Tier:** complex | **Audit-class:** false | **Executor:** sonnet-4.6/high
**BOUNDARY:** mechanical execution changes only — ZERO decision/threshold changes in the live loop.
Findings: perf-01, perf-10, api-design-01/02/04/05/06/08/09, py-core-01, py-core-05, pysvc-09.

---

## STEP-TEXT CORRECTIONS (drift since the 2026-07-19 audit — every prior phase-75 research found real drift; here it is)

1. **perf-10 is ALREADY `to_thread`'d — the "blocks the event loop" rationale is FALSE.** `autonomous_loop.py:641` is `info = await asyncio.to_thread(lambda x=t: yf.Ticker(x).info or {})` — added 2026-05-18 (commit `6ceeb10ff`), i.e. it PRE-DATES the audit. The per-ticker `.info` fetch does NOT block the loop today; the only remaining defect is **serial latency** (N sequential `await`s in the `for t in target_tickers` loop, :639). The prescription (bounded `gather`) is still a valid *latency* win, but the executor/Q/A must NOT describe it as "unblocking the event loop" — it is already non-blocking. (`lambda x=t:` already fixes the late-binding closure bug — keep it.)

2. **api-design-01 (get_dashboard): httpx is ALREADY async; "sync httpx" is stale.** `mas_events.py:116` and `:122` use `async with httpx.AsyncClient(timeout=3)` + `await c.get(...)`. The real blockers are the **2× `subprocess.run`** at `:146` (`openclaw gateway status`, timeout=10) and `:153` (`openclaw cron list`, timeout=15) → worst-case ~25s, plus `list_openclaw_sessions()` at `:173` (verify blocking). Criterion-3 targets `subprocess.run/sync-httpx/.result(` — the async httpx already complies; wrap the two `subprocess.run` (and `list_openclaw_sessions`) in `asyncio.to_thread`.

3. **api-design-02 (get_llm_p95_latency): ALREADY carries `timeout=30`; "untimed" is stale.** `performance_api.py:82` is `rows = list(bq.client.query(sql).result(timeout=30))`. Criterion-3's "p95 query carries timeout=30" is ALREADY satisfied. Remaining work = wrap the sync `.result(...)` in `await asyncio.to_thread(...)` (it is a blocking BQ call inside `async def`).

4. **api-design-04 (run_data_ingestion): the heavy call is ALREADY `to_thread`'d; two smaller items remain.** `backtest.py:200` already does `await asyncio.to_thread(service.run_full_ingestion, ...)`. Remaining: (a) `screen = screen_universe()` inline at `:195` is sync (~500-ticker `yf.download`) — wrap in `to_thread` OR move into the background task; (b) the route returns `{"status":"completed", "result":result}` synchronously at `:207` — criterion 6 wants **202-immediate + pollable status** mirroring `/run`. The `/run` pattern to copy (`run_backtest` :112-135): set a module state dict `{"status":"running","run_id":...}`, `asyncio.create_task(_run_ingestion_async(...))`, `return {"status":"started","run_id":...}`; poll via a status route (there is already `GET /ingest/status` :213 returning row counts — add a task-progress state dict, don't overload it).

5. **get_log_tail cap mismatch.** Step says "lines le=1000"; actual is `lines: int = Query(200, ge=1, le=10000)` (`cron_dashboard_api.py:533`) plus a second clamp `n = max(_LINES_MIN, min(_LINES_MAX, ...))` (:543). Lowering to `le=1000` is a **behavior change** (rejects requests 1001-10000 that succeed today) — DECISION POINT for the operator/Q/A, not a silent mechanical edit. Recommend: keep the seek-from-end rewrite; leave the cap at `le=10000` unless the operator wants it lowered.

6. **2026-05-25 incident line cite drift.** Step cites `paper_trading.py:1312-1320`; the actual misfire-grace comment is `:1301-1311` (`:1312-1320` is now the `reschedule_paper_job` docstring). The incident narrative (cron fired, event-loop contention pushed dispatch 2.10s late, APScheduler skipped, `misfire_grace_time` raised to 3600) is CONFIRMED at :1301-1311 and directly supports perf-01.

7. **A 10th in-coroutine `get_event_loop()` exists OUTSIDE criterion-2's 3-file scope.** `ticket_queue_processor.py:423` (`loop = asyncio.get_event_loop()` inside an async method). Criterion 2 scans only orchestrator/task_bus/mas_events, so this will NOT fail the gate — but it is the same latent Python-3.14 class. Per `feedback_queue_discovered_defects_in_masterplan`, queue it as its OWN step; do NOT silently expand 75.10's surface.

8. **`mas_events.py` has TWO `get_event_loop` occurrences, both must go for criterion 2.** `:97` (the crash path) AND `:205` (`id(asyncio.get_event_loop)` in `make_run_id`). Line 205 does NOT call it (takes `id()` of the function object → a process-constant, meaningless term; no crash) but the criterion-2 grep matches the string, so both must be removed.

9. **Orchestrator `:430` `loop` is DEAD.** In `_execute_full_flow` (415-544) the `loop = asyncio.get_event_loop()` at :430 is never used (no `loop.` reference in the method body). Cleanest criterion-2 fix = delete the line (the other 7 orchestrator sites feed `loop.run_in_executor` and need `get_running_loop()`).

10. **A 4th unsaved fire-and-forget task:** the prewarm task `asyncio.create_task(_prewarm_ticker_meta())` at `main.py:406` (return discarded — GC-able). pysvc-09 already calls for cancelling it; note it is *also* an api-design-09-class GC hazard.

---

## Part A — Internal re-anchoring (verbatim current file:line)

### py-core-05 — the 9 `get_event_loop` sites (criterion 2 = ZERO in 3 files)
All 8 orchestrator sites are inside `async def` (running loop guaranteed → `get_running_loop()` is safe & preferred):

| File:line | Enclosing `async def` | Use |
|---|---|---|
| multi_agent_orchestrator.py:430 | `_execute_full_flow` | **DEAD var → delete** |
| :566 | `_think_plan` | `loop.run_in_executor` |
| :586 | `_iterative_parallel_research` | `loop.run_in_executor` |
| :690 | `_check_research_complete` | `loop.run_in_executor` |
| :723 | `_synthesize` | `loop.run_in_executor` |
| :738 | `_single_with_delegation` | `loop.run_in_executor` |
| :862 | `_quality_gate` | `loop.run_in_executor` |
| :1065 | `_classify_via_llm` | `loop.run_in_executor` |
| task_bus.py:140 | `delegate` (async) | `asyncio.get_event_loop().create_future()` → `get_running_loop().create_future()` |

Criterion-2 grep must ALSO be clean in `mas_events.py` → remove :97 and :205 (items 8 above).

### py-core-01 — MASEventBus (`agents/mas_events.py`)
- **`:97`** `self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None`. `_lock` is **DEAD** — no in-file reader (`emit`/`subscribe` never use it; the many other `_lock` grep hits are unrelated `threading.Lock` classes: cost_tracker, kill_switch, api_cache…). **Safe to delete outright.** If kept, use `try: asyncio.get_running_loop() except RuntimeError: None`.
- **Construction contexts** (singleton `get_event_bus()` :192 → `MASEventBus()` :196): async callers = orchestrator :431/:588/:1336, `api/mas_events.py` routes (running loop present → fine today). **Sync/dangerous callers = the Slack-bot process** `slack_bot/app_home.py:41/455/543/559` (these can run with NO running loop → this is the Python-3.14 crash surface, and even pre-3.14 raises from a non-main thread). This is why `get_running_loop()`+try/except (or deleting `_lock`) is the fix.
- **`_forward_remote` :132-144** spawns `threading.Thread(target=_send, daemon=True).start()` **per event**, each doing a fresh `import httpx` + sync `httpx.post(timeout=3)`. No consumer relies on the thread-per-event behavior for ordering — the remote sink is `/api/mas/events/ingest`, best-effort, buffered by arrival. A single daemon worker draining a `queue.Queue` through one shared `httpx.Client` is strictly BETTER (FIFO ordering + connection reuse — httpx docs: "avoid instantiating multiple client instances in hot loops"). Lifetime: make the worker daemon so it dies with the process.
- **`make_run_id` :205** — output is an opaque 12-hex hash used only as an event `run_id` (grouping); no consumer parses the `id()` term. Replace `id(asyncio.get_event_loop)` with e.g. `uuid.uuid4().hex` or `os.getpid()` — any unique-ish term.

### perf-01 — run_daily_cycle Step-1 (execution context MEASURED)
`run_daily_cycle` is `async def` (`autonomous_loop.py:252`), invoked via `await` from `_scheduled_run` (`paper_trading.py:1344`, APScheduler) and `_run_cycle_background` (:1268). **Both schedulers are `AsyncIOScheduler`** (`main.py:301` paper, `:348` queue) → jobs run **as coroutines ON the API event loop**. So `screen_universe(...)` at `:576` (→ `screener.py:137` `yf.download(tickers, period="6mo", threads=True)`, ~500 tickers) runs SYNC on the event loop → **genuinely blocks it**. Fix per criterion 4: `screen_data = await asyncio.to_thread(screen_universe, tickers=universe, period="6mo", sector_lookup=_sector_lookup, short_interest_lookup=..., short_interest_threshold=...)`. (`yf.download`'s internal `threads=True` only parallelizes the download; the outer call still blocks the caller.)

**Siblings + gating flags (measured defaults, all `False`):**
| Site | Call | Gate flag (settings.py) | Default |
|---|---|---|---|
| :524 | `get_sp500_tickers()` | inside `if _intl_markets:` (paper_markets≠[US]) | paper_markets default `["US"]` (:78) — **but live .env = US+EU+KR, so this IS live-executed** |
| :572 | `build_sector_map(universe)` | `sector_neutral_momentum_enabled` OR `multidim_momentum_enabled` OR `paper_soft_sector_diversity_enabled` | all `False` (:429/:439/:448) |
| :639 | peer_leadlag `yf.Ticker().info` | `peer_leadlag_enabled` | `False` (:515) |

Wrapping any of these in `to_thread`/`gather` is **output-identical** whether the flag is ON or OFF (it moves WHERE the same call runs, not WHAT it computes) — this is the BOUNDARY argument. Executor must still verify live flag state at implementation and record ON-vs-OFF `$0` diff for any flag that is live.

### perf-10 — see correction #1. Fix = bounded `gather`:
```python
sem = asyncio.Semaphore(8)
async def _one(t):
    async with sem:
        try: return t, await asyncio.to_thread(lambda x=t: yf.Ticker(x).info or {})
        except Exception: return t, None
results = await asyncio.gather(*[_one(t) for t in target_tickers])
# then build lookup from results (same dict shape as today)
```

### api-design-09 — fire-and-forget create_task (GC hazard)
| Site | Task | State dict for done-callback |
|---|---|---|
| analysis.py:364 | `_run_sync_analysis` | analysis task-state (keyed by `task_id`) |
| backtest.py:134 | `_run_backtest_async` | `_backtest_state` (`status`/`error`/`traceback`) |
| paper_trading.py:1023 | `_run_cycle_background` | `_last_cycle_error` (coro already try/excepts, but task ref still GC-able) |
| backtest.py:299 | `_optimizer_task` | ALREADY held in a module var — reference exists; add `add_done_callback` for error-state parity if not present |

Pattern (official): module `set()` keep-alive + `task.add_done_callback` that (a) `discard`s from the set and (b) on `task.exception()` flips the state dict to `error`.

### pysvc-09 — lifespan `finally` (main.py:410-419)
Today shuts down ONLY `queue_scheduler` (`:416-417`, `shutdown(wait=False)`). MISSING:
- **paper `scheduler`** (:301, started :303) — add `if 'scheduler' in locals(): scheduler.shutdown(wait=False)`.
- **prewarm task** (:406, unsaved) — capture `t = asyncio.create_task(...)`; in finally `t.cancel()` + `await` under `contextlib.suppress(asyncio.CancelledError)`.
- **Slack monitor** — `init_slack_monitor` RETURNS the monitor (`slack_monitor.py:105`), and `SlackMonitor` HAS a **sync** `def stop(self)` (`:88`). But `main.py:333` **discards the return** (`await init_slack_monitor(slack_client)`). Capture it (`monitor = await init_slack_monitor(...)`) and call `monitor.stop()` in finally (or add a module-level `get_slack_monitor()`/`stop_slack_monitor()`). MEASURED: the hook exists and is sync — no `await`.
APScheduler `shutdown(wait=False)` "Does not interrupt any currently running jobs" and won't block startup teardown (apscheduler base docs).

### api-design-05/06/08 — the remaining routes
- **_enrich_position (portfolio.py:152, plain `def`)** — calls blocking `yfinance_tool.get_comprehensive_financials()` (:156); invoked in a `for` loop inside `async def get_portfolio_performance` (:114-115) → N+1 serial blocking on the loop. Fix: `await asyncio.gather(*[asyncio.to_thread(_enrich_position, pid, p) for pid, p in _positions.items()])` (optionally Semaphore-bounded). (Legacy in-memory `_positions` — verify still routed.)
- **get_optimizer_status (backtest.py:352, `async def`, NO awaits in body)** — `subprocess.run(["pgrep", -f, pattern])` in a 3-iteration loop (:363) + `open(tsv)`/`open(json)` reads (:375/:391). Criterion-3 wants "plain def or fully to_thread-wrapped". Cleanest = **convert to plain `def`** (FastAPI runs it in the threadpool — fastapi/async: plain def "is run in an external threadpool that is then awaited"). TTL cache is an optional add.
- **get_all_jobs (cron_dashboard_api.py:411, `async def`)** — `_launchctl_state(entry["id"])` (:455) → `_probe_launchctl` (:258) → `subprocess.run(["launchctl","print",...])` (:267), 30s TTL cache but a miss forks sync (~400ms/entry). Wrap the `_launchctl_state` calls in `to_thread` (or gather).

---

## Part B — External research

### Read in full (6; ≥5 gate met)
| # | URL | Kind | Key verbatim finding |
|---|---|---|---|
| 1 | https://docs.python.org/3/library/asyncio-eventloop.html | official | `get_event_loop()`: **"Changed in version 3.14: Raises a RuntimeError if there is no current event loop."** `get_running_loop()` "is preferred to `get_event_loop()` in coroutines and callbacks." run_in_executor example uses `loop = asyncio.get_running_loop()`. |
| 2 | https://docs.python.org/3/library/asyncio-task.html | official | create_task: **"Save a reference… The event loop only keeps weak references to tasks. A task that isn't referenced elsewhere may get garbage collected at any time, even before it's done."** → `background_tasks.add(task)` + `task.add_done_callback(background_tasks.discard)`. to_thread: **"Due to the GIL, asyncio.to_thread() can typically only be used to make IO-bound functions non-blocking."** |
| 3 | https://fastapi.tiangolo.com/async/ | official | plain `def` path op "is run in an external threadpool that is then awaited, instead of being called directly (as it would block the server)"; avoid blocking I/O inside `async def`. |
| 4 | https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html | official | `shutdown(wait=True)`: waits for running jobs; `wait=False` returns immediately; **"Does not interrupt any currently running jobs."** |
| 5 | https://docs.python.org/3/library/asyncio-sync.html | official | `asyncio.Semaphore(n)` + `async with sem:` bounds concurrent operations (counter blocks at 0). |
| 6 | https://www.python-httpx.org/async/ | official | sync `httpx.get/Client` "should be avoided in async contexts, as they will block the event loop"; use `httpx.AsyncClient` + `await`; reuse one client (no hot-loop instantiation). |

### Snippet-only (URLs collected, not read in full)
docs.python.org/3/whatsnew/3.14.html; docs.python.org/3/library/asyncio-policy.html; github.com/browser-use/browser-use#4447; github.com/grpc/grpc#39507; github.com/run-llama/llama_index#18058; github.com/micropython/micropython#12299; superfastpython.com asyncio-disappearing-task-bug; mkennedy.codes fire-and-forget-asyncio; stackoverflow #136168 (tail seek-from-end); dpdzero/leapcell/sentry FastAPI threadpool posts. (**URLs collected ≥16.**)

### Recency scan (2024-2026)
The load-bearing finding IS the newest: `asyncio.get_event_loop()` raising `RuntimeError` landed in **Python 3.14** (deprecated since 3.10) — confirmed by the official "Changed in version 3.14" note and multiple 2025-2026 downstream breakage reports (browser-use #4447, grpc #39507). The create_task weak-ref note and to_thread GIL caveat are current 3.14-docs wording. No newer guidance supersedes the `get_running_loop()` replacement. No contradicting source found (the change is unambiguous in the reference).

---

## Part C — Application (external → pyfinagent)
- **[1]** → py-core-05 (all 9 sites) + py-core-01:97: `get_running_loop()` inside coroutines; the 3.14 RuntimeError is exactly the MASEventBus:97 crash class. run_in_executor sites already have a running loop.
- **[2] save-reference** → api-design-09 (+ prewarm main.py:406): keep-set + add_done_callback. **[2] to_thread GIL** → perf-01/perf-10 + every route wrap are IO-bound (yfinance/BQ/subprocess/file) so to_thread is correct; none are CPU-bound.
- **[3]** → api-design-06 get_optimizer_status → convert to plain `def`; the other async routes stay `async def` with `to_thread` wrappers around the blocking calls.
- **[4]** → pysvc-09 use `shutdown(wait=False)` (matches existing queue_scheduler call) so teardown never blocks.
- **[5]** → perf-10 + _enrich_position bounded `gather` (Semaphore(8)).
- **[6]** → get_dashboard's async httpx is already compliant; only subprocess needs to_thread.

## Part D — Boundary invariants (must stay byte-identical; diff-verify)
The executor edits ONLY *where* code runs (thread vs loop), never *what* it decides. Keep byte-identical in `autonomous_loop.py`: the rank/gate/sizing/threshold lines around the touched fetches — the flag gates (`if getattr(settings, "..._enabled", False)`), the calendar `_open_today` gate (:536-549), the `short_interest_threshold`/`peer_leadlag_*`/`ma_preannounce_*` numeric args, and every `screen_data[...]`/`paper_screen_top_n` slice. Wrapping `screen_universe`/`build_sector_map`/peer-leadlag in `to_thread`/`gather` must return the SAME objects into the SAME variables. `experiment_results.md` must carry `git diff --stat` + an explicit "no decision-line edits" diff of `autonomous_loop.py`. For any live-ON flag, record an ON-vs-OFF `$0` behavior diff.

## Test guidance (`backend/tests/test_phase_75_event_loop.py`)
- **pytest-asyncio is NOT installed** (only `anyio`); `@pytest.mark.asyncio` used in 0 tests; **`asyncio.run(...)` is the convention (17 files)**. conftest.py has NO event_loop fixture (only an import-time llm_call_log guard) → nothing interferes.
- Criterion 1 (construct `MASEventBus()` with NO running loop): do it in a **plain `def test_...`** body (no running loop there naturally) — do NOT wrap in `asyncio.run()`. Assert no raise. For `_forward_remote`, assert single-worker/queue via source-scan AND/OR behavioral (set `remote_url`, emit N, assert ≤1 worker thread) — mutate the stub too (guard-can-fail per `feedback_mutation_test_guards_and_fixtures`).
- Criterion 2 grep: assert `get_event_loop` count == 0 across the 3 files (this WILL catch mas_events:205 — make sure the fix removes it).
- Criterion 3 AST: to hard-fail on missing routes, resolve each `async def` node by name and `assert` it was found (fail if the route was renamed/removed), then walk `ast.Call` nodes for `subprocess.run`/`httpx` sync/`.result(` NOT lexically inside an `asyncio.to_thread(...)` call; allow-list the async `httpx.AsyncClient().get` (it is `await`ed, not sync). Avoid false positives on `.result(` by checking the attribute chain is a BQ query, not e.g. a Future inside to_thread.
- Criterion 5: drive one create_task to raise; assert the state dict flips to `error` via the done-callback (use `asyncio.run` around a small harness).
- Criterion 6: mock ingestion; assert the route returns immediately (`status: started` + run_id) and the status route reports progress.

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 11,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0, "K_required": 2, "new_findings_last_round": 0, "dry": false},
  "summary": "All 12 findings verified against current source with material drift: perf-10 (:641) is ALREADY to_thread'd (serial-latency only, not loop-blocking); get_dashboard already uses async httpx (real blockers = 2x subprocess.run); get_llm_p95 already carries timeout=30; run_data_ingestion already to_thread's run_full_ingestion (remaining = screen_universe inline + 202-conversion). Confirmed: 9 get_event_loop sites (8 orchestrator all async + task_bus:140) + mas_events has TWO occurrences (:97 crash-path dead _lock, :205 meaningless id() term) both needed for criterion-2; orchestrator:430 loop is dead (delete); a 10th get_event_loop at ticket_queue_processor:423 is OUT of the 3-file scope (queue separately). pysvc-09: finally shuts only queue_scheduler; paper scheduler + prewarm task (main:406, also GC-hazard) + Slack monitor (stop() exists, sync, slack_monitor:88, but return discarded at main:333) are unshut. perf-01 confirmed: AsyncIOScheduler runs run_daily_cycle on the event loop; gating flags all default False, paper_markets live=US+EU+KR. Test: pytest-asyncio NOT installed -> plain def + asyncio.run; criterion-1 no-loop construction natural in a def body. get_log_tail cap is le=10000 not le=1000 (behavior-change decision).",
  "brief_path": "handoff/current/research_brief_75.10.md",
  "gate_passed": true
}
```
