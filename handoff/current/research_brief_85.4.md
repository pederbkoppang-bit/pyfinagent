# Research Brief — phase-85.4 (P0 ENGINE HEALTH: cycle never completes)

Tier: **moderate**. Audit-class: **false**. Started 2026-08-08.

Status: IN PROGRESS (write-first; appended incrementally as sources are read).

## Scope

Step 85.4: the autonomous trading cycle has not COMPLETED since 2026-07-31.
Deliverables requested by Main:
- A. Internal code audit (bulk of value): autonomous_loop phase structure,
  timeout semantics, terminal-row guarantee, gather semantics, cc_rail call
  sites, cycle_history writers, existing watchdog surface, test seams.
- B. >=5 external sources read in full (asyncio gather/timeout/TaskGroup on
  Python 3.14, finally-under-cancellation, aggregate-staleness alerting,
  per-task timeout budgeting).
- C. Arithmetic feasibility of a 7200s budget at ~88s median / 150s cap /
  17.2% timeout rate for the derived ticker count.

(sections appended below as evidence lands)

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.python.org/3/library/asyncio-task.html | 2026-08-08 | Official docs (Python 3.14.7) | WebFetch (full) | "If `return_exceptions` is False (default), the first raised exception is immediately propagated to the task that awaits on `gather()`. Other awaitables in the *aws* sequence **won't be cancelled** and will continue to run." Also: `asyncio.timeout` "will cancel the current task and handle the resulting `asyncio.CancelledError` internally, transforming it into a `TimeoutError`... which means the `TimeoutError` can only be caught *outside* of the context manager." And: "`asyncio.CancelledError` directly subclasses `BaseException`". |
| 2 | https://hynek.me/articles/waiting-in-asyncio/ | 2026-08-08 | Authoritative blog (Hynek Schlawack, pub 2020-05-21, upd 2023-07-28) | WebFetch (full) | "If you can, use **`asyncio.TaskGroup`**. It's the most modern, user-friendly API with the fewest sharp edges." `gather()` has no timeout parameter; a group timeout requires `wait_for(gather(...))` or `async with asyncio.timeout(...)`. |
| 3 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-08 | Book ch.6 (Rob Ewaschuk / Betsy Beyer, O'Reilly 2017) | WebFetch (full) | "Black-box monitoring is symptom-oriented and represents active--not predicted--problems: 'The system isn't working correctly, right now.'" "Every page should be actionable." "The rules that catch real incidents most often should be as simple, predictable, and reliable as possible." NOTE: chapter contains **no** guidance on alerting on the absence of successful work -- that gap is filled by source 5. |
| 4 | https://sre.google/sre-book/data-processing-pipelines/ | 2026-08-08 | Book ch.25 (Dan Dennison / Tim Harvey, O'Reilly 2017) | WebFetch (full) | On a job whose runtime approaches its period: "each new run might stack up on the cluster scheduler because the previous run is not complete. Even worse, the currently executing and nearly finished run could be killed." Also names the Moire-load-pattern and thundering-herd failure classes. NOTE: this chapter likewise has **no** data-freshness-alerting guidance -- recorded as a negative finding rather than padded. |
| 5 | https://prometheus.io/docs/practices/alerting/ | 2026-08-08 | Official docs (Prometheus) | WebFetch (full) | "alert on symptoms that are associated with end-user pain rather than trying to catch every possible way that pain could be caused". For batch jobs: alert if they have not **completed** recently enough; thresholds "should allow for at least 2 full job cycles" (a 4h-period / 1h-runtime job warrants a ~10h threshold). For offline processing: "Page when data throughput delays risk user impact." NOTE: the page does **not** literally contain a `last_success_timestamp` expression -- claim corrected against the fetched text. |

| 6 | https://docs.python.org/3.14/whatsnew/3.14.html | 2026-08-08 | Official docs | WebFetch (full) | NEW live introspection: `python -m asyncio ps PID` "inspects the given process ID (PID) and displays information about currently running asyncio tasks... a flat listing of all tasks, their names, their coroutine stacks, and which tasks are awaiting them", and `python -m asyncio pstree PID` renders "a visual async call tree". Only asyncio API change: `create_task()` now takes arbitrary kwargs. NO documented change to `gather`/`timeout`/`to_thread`/cancellation. |
| 7 | https://oneuptime.com/blog/post/2026-02-09-monitor-cronjob-missed-schedules/view | 2026-08-08 | Industry blog (Nawaz Dhandala, OneUptime, 2026-02-09) | WebFetch (full) | **[RECENCY 2026]** Alert on **absence of success**, not on errors: track `kube_cronjob_status_last_successful_time`, alert on `(time() - last_successful_time) > threshold`, threshold set to **2-3x the normal schedule interval**. "Status fields reveal 'has never succeeded' conditions undetectable from error metrics alone." Error-only monitoring misses jobs that never execute. |
| 8 | https://training.promlabs.com/training/monitoring-and-debugging-prometheus/metrics-based-meta-monitoring/end-to-end-watchdog-alerts/ | 2026-08-08 | Training/vendor docs (PromLabs) | WebFetch (full) | Watchdog / dead-man's-switch pattern: an always-firing alert (`expr: vector(1)`) routed to an EXTERNAL service that pages when the heartbeat **stops arriving**. Catches "meta-monitoring infrastructure errors causing the monitoring system to fail", Alertmanager downtime, and delivery failure -- failure modes ordinary alerts structurally cannot catch. |
| 9 | https://anyio.readthedocs.io/en/stable/threads.html | 2026-08-08 | Official library docs (AnyIO, stable) | WebFetch (full) | **Load-bearing for criterion 2:** "There is no mechanism in Python to cancel code running in a thread." By default "tasks are shielded from cancellation while they are waiting for a worker thread to finish"; `abandon_on_cancel=True` lets the task be cancelled but "the thread will still continue running -- only its outcome will be ignored." Voluntary check via `from_thread.check_cancelled()`. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://superfastpython.com/asyncio-gather/ | Blog | Superseded by source 1 (official docs) on the same semantics |
| https://superfastpython.com/asyncio-task-cancellation-best-practices/ | Blog | Superseded by source 1 |
| https://superfastpython.com/asyncio-shield/ | Blog | Superseded by source 1 |
| https://superfastpython.com/asyncio-event-loop-exit/ | Blog | Superseded by source 9 (AnyIO) on thread-cancellation |
| https://billypoon.com/insights/structured-concurrency-in-python-with-taskgroup-... | Blog | Superseded by sources 1+2 |
| https://fixdevs.com/blog/python-asyncio-gather-error/ | Community-tier | Lowest tier; official docs cover it |
| https://dev.to/_eb7f2a654e97a60ae9f96e/asyncio-pitfalls-the-3-hour-bug-4m3e | Community-tier | Anecdotal |
| https://github.com/python/cpython/issues/136084 (`to_thread` efficiency) | Issue tracker | Perf, not cancellation semantics |
| https://github.com/python/cpython/issues/98275 (`Task.cancel` docs inaccurate) | Issue tracker | Noted; source 1 carries the corrected wording |
| https://docs.python.org/3/library/asyncio-threading.html | Official docs | 3.14 free-threading; not load-bearing here (repo runs the GIL build) |
| https://labs.quansight.org/blog/scaling-asyncio-on-free-threaded-python | Vendor lab blog | Free-threading perf; out of scope |
| https://docs.datadoghq.com/watchdog/alerts/ | Vendor docs | Proprietary ML alerting; not applicable locally |
| https://dev.to/cronmonitor/how-to-monitor-cron-jobs-in-2026-a-complete-guide-28g9 | Community-tier | Overlaps source 7 |
| https://blog.pythonlibrary.org/2025/10/09/an-intro-to-python-3-14s-new-features/ | Blog | Superseded by source 6 |
| https://betterstack.com/community/guides/scaling-python/python-3-14-new-features/ | Vendor blog | Superseded by source 6 |
| https://en.wikipedia.org/wiki/Watchdog_timer | Encyclopaedia | Background only |

**URLs collected (unique): 25.** Read in full: 9. Snippet-only: 16.

### Search-query composition (3-variant discipline)

| Variant | Query run |
|---|---|
| Year-less canonical | `asyncio.gather one hanging task blocks entire group return_exceptions timeout` |
| Year-less canonical | `Python asyncio finally block CancelledError shielded cleanup guaranteed to run` |
| Current-year (2026) | `watchdog alert job has not succeeded in N periods staleness monitoring 2026` |
| Last-2-year (2025) | `asyncio to_thread cannot be cancelled blocking subprocess thread keeps running 2025` |
| Version-scoped | `Python 3.14 asyncio changes TaskGroup free-threading what's new` |

### Recency scan (last 2 years, 2024-2026)

**Performed. Two new findings that MATTER, one that does not.**

1. **[2026] Alert on absence-of-success, not on error** (source 7, OneUptime 2026-02-09). This is the exact pattern criterion 4 needs and it is newer and sharper than the 2017 SRE-book chapters, which -- measured, not assumed -- contain **no** guidance on staleness/absence alerting (both ch.6 and ch.25 were fetched in full and checked). Threshold guidance (2-3x schedule interval) corroborates Prometheus's older "at least 2 full job cycles" (source 5).
2. **[2025-10, Python 3.14] `python -m asyncio ps PID` / `pstree PID`** (source 6). Brand-new, directly applicable: the backend is a long-lived 3.14 process, so a hung cycle can now be introspected **live, read-only, without a restart** -- it prints the coroutine stack and the awaiter chain for every task. There was no equivalent before 3.14. This supersedes "add more logging" as the first diagnostic move.
3. **No supersession** of the core asyncio semantics: the 3.14 whatsnew documents **no** change to `gather`, `timeout`, `to_thread`, or cancellation (only `create_task` kwargs). The canonical year-less prior art (source 2, 2020/2023) still stands: prefer `TaskGroup`; `gather` has no timeout of its own.

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | 3418 | THE trading cycle. `run_paper_trading_cycle` = steps 1-10 | live |
| `backend/services/cycle_health.py` | 602 | `cycle_history.jsonl` writer + `cycle_heartbeat_alarm` | live, **defective** (see F4) |
| `backend/agents/orchestrator.py` | ~2200+ | per-ticker 28-agent pipeline; `_generate_with_retry` | live |
| `backend/agents/claude_code_client.py` | ~800 | cc_rail subprocess wrapper + rail guard | live |
| `backend/slack_bot/scheduler.py` | 800+ | watchdog cron; calls `cycle_heartbeat_alarm` at :775-807 | live |
| `backend/services/observability/alerting.py` | -- | `AlertDeduper`, `raise_cron_alert_sync` | live, working |
| `backend/services/kill_switch.py` | 700+ | pause/resume latch replayed from the audit JSONL | live, **latched paused** (see F6) |
| `backend/tests/test_cycle_heartbeat_alarm.py` | ~200 | 8 tests of the alarm | live |
| `backend/tests/test_phase_38_2_cycle_start_logging.py` | ~160 | started-row + orphan-row tests; monkeypatches `_HISTORY_PATH` **and** `_HEARTBEAT_PATH` | live -- **the fault-injection seam criterion 3 needs** |
| `tests/services/test_cycle_failure_alerts.py` | ~200 | phase-23.2.18 regression guard for exactly this failure class | live |
| `backend/tests/test_autonomous_loop_integration.py` | -- | **FALSE FRIEND**: tests `backend/autonomous_loop.py` (the harness orchestrator), NOT `backend/services/autonomous_loop.py` (the trading cycle). Do not cite it as cycle coverage. | live |

### A1. The cycle timeout -- what actually happens when it fires

- `_cycle_timeout = float(getattr(settings, "paper_cycle_max_seconds", 1800.0))` at `backend/services/autonomous_loop.py:439`; the default is **7200.0** at `backend/config/settings.py:33`.
- `async with asyncio.timeout(_cycle_timeout):` at `backend/services/autonomous_loop.py:446` wraps steps 1-10.
- `except asyncio.TimeoutError:` at `:1687-1691` sets `summary["status"] = "timeout"`, `error = f"cycle exceeded {_cycle_timeout:.0f}s"`, and **returns** the summary.
- `finally:` at `:1706` -> releases the cycle lock, then at `:1746-1768` calls `_cycle_log().record_cycle_end(...)` with `status=summary.get("status", "unknown")`.

**Criterion 3, answered: the terminal row IS written today, and the premise "no terminal row" is REFUTED by the file itself.** `handoff/cycle_history.jsonl` carries a real terminal row for every one of the three timeouts:

```
{"cycle_id": "ab116cd1", ..., "duration_ms": 7200968, "status": "timeout", ...}   # 08-04
{"cycle_id": "fdd19797", ..., "duration_ms": 7201503, "status": "timeout", ...}   # 08-06
{"cycle_id": "0c2ffd64", ..., "duration_ms": 7201746, "status": "timeout", ...}   # 08-07
```

The `finally` runs because `asyncio.timeout` converts `CancelledError` into `TimeoutError` **inside** the context manager (source 1: "transforming it into a `TimeoutError` which can only be caught *outside* of the context manager") -- the writer is downstream of that conversion and is never itself cancelled.

**The real criterion-3 defect is status FIDELITY, not row absence.** Two holes:

- **F3a -- `"running"` leaks out as a terminal status.** `summary = {"status": "running", "steps": []}` at `:362`. The kill-switch halt path at `:1313-1327` logs `"kill-switch active (%s) -- skipping decide/execute"`, appends `kill_switch_halted`, and `return summary` at `:1327` **without ever setting a terminal status**. So `record_cycle_end` writes `status: "running"`, and the post-finally block at `:1777` (`if _final_status not in ("completed", "skipped")`) fires a P1 literally titled **"Autonomous trading cycle running"**. Measured, both halves:
  - `cycle_history.jsonl` 08-03 and 08-05 rows: `"completed_at": "...", "duration_ms": 5709941, "status": "running"`.
  - backend.log `2026-08-05 21:35:10,456 WARNING alerting alert bot-token fallback delivered=True source=autonomous_loop title='Autonomous trading cycle running'`.
- **F3b -- SIGKILL still leaves an orphan.** A hard kill skips the `finally` entirely, leaving a lone `started` row. `cycle_health.orphan_rows()` (`cycle_health.py:~366-400`) detects exactly this, but **nothing schedules it and nothing pages on it** -- grep for callers returns tests only.

### A2. Analysis-phase dispatch -- can one ticker stall the gather?

Dispatch is at `backend/services/autonomous_loop.py:1109-1168`:

```
_concurrency = 3 if _std_model.startswith("claude-") else 8      # :1098-1103
_analysis_semaphore = asyncio.Semaphore(_concurrency)            # :1109
candidate_results = await asyncio.gather(
    *[_run_and_persist_one(t, "new") for t in analyze_tickers],
    return_exceptions=True,                                      # :1157-1160
)
holding_results = await asyncio.gather(..., return_exceptions=True)  # :1164-1167
```

`_std_model` is `settings.gemini_model` = **`claude-sonnet-4-6`** (`settings.py:31`), so **concurrency = 3**, confirmed live: `2026-08-07 20:09:16 Paper trading: per-provider concurrency cap = 3 (standard=claude-sonnet-4-6)`.

**Criterion 2, answered honestly -- the EXCEPTION half is already safe; the HANG half is not.**

- **Exceptions: SAFE.** `return_exceptions=True` is set on both gathers, and `_run_and_persist_one` additionally wraps `_run_single_analysis` in `try/except Exception` at `:1134-1138` (`logger.error(f"Failed to analyze {kind} {ticker}: {exc}"); return None`). Per source 1, only the `return_exceptions=False` default propagates early -- that is not this code. So "one unhandled per-ticker failure kills the gather" is **REFUTED**.
- **A never-resolving await: UNBOUNDED.** I swept `_run_single_analysis` (`:1859-2520`) for `asyncio.timeout` / `wait_for` / `TimeoutError` / `timeout=` and got **zero matches**. There is **no per-ticker timeout anywhere**. The only bounds are (a) the cycle-level 7200s and (b) per-LLM-step `future.result(timeout=...)` inside the orchestrator. Per source 1, `gather` waits for ALL tasks, so one non-returning ticker would consume the entire remaining cycle budget -- and because the tasks hold `_analysis_semaphore` slots, one stuck ticker also permanently removes 1 of 3 concurrency slots.
- **The cycle timeout cannot stop the thread.** Every rail call runs under `asyncio.to_thread` (e.g. `:2543`, `:2655`, `:2680`). Source 9 (AnyIO): "There is no mechanism in Python to cancel code running in a thread... the thread will still continue running -- only its outcome will be ignored." **Measured**: after the 08-07 cycle died at 22:00:01, `risk_debate` kept working and logged `Risk debate complete: decision=APPROVE_REDUCED` at **22:03:11** -- 3m10s of orphan work, including 1 further `claude_code_invoke` that itself timed out, all spent on a cycle that no longer existed. Bounded here, but the mechanism is unbounded in principle.

### A3. cc_rail call sites and retry wall-clock multiplication

- Subprocess: `subprocess.run(..., timeout=timeout_s)` at `claude_code_client.py:409-419`; on `TimeoutExpired` -> `raise ClaudeCodeError(f"claude CLI timeout after {timeout_s}s")` at `:423-429`.
- `ClaudeCodeClient.__init__(self, model_name, timeout_s: int = 150)` at `:593`; `recommended_step_timeout = timeout_s + 30` at `:600` (=180).
- **The error is swallowed into a successful-looking empty response**: `except ClaudeCodeError` at `:762-777` returns `LLMResponse(text="", thoughts=f"errored: {exc}")`. It never raises.
- **Retry, and it DOES multiply wall clock.** `_generate_with_retry(..., max_retries: int = 3, timeout: int = 90)` at `orchestrator.py:811`. `timeout = _resolve_step_timeout(model, timeout, is_grounded)` at `:824` lifts it to `recommended_step_timeout` = **180s**. The loop at `:878`:
  - each attempt builds a **fresh `concurrent.futures.ThreadPoolExecutor(max_workers=1)`** and does `future.result(timeout=180)` (`:880-883`);
  - phase-61.2 result-based retry at `:925-943`: if `text == "" and thoughts.startswith("errored:")` and budget remains -> `time.sleep(random.uniform(0, min(15, 2*2**n)))` then `continue`.
  - Budget: `_empty_retry_budget = claude_code_empty_retry_max` (default **2**, `settings.py:192`) but **only when `paper_synthesis_integrity_enabled` is True** (`settings.py:206`, documented "DARK until operator promotion") -- so verify the live flag value before assuming retries are armed.
  - **Worst case per agent step: 3 x 150s subprocess + up to 2 x <=15s jitter = ~480s for ONE of ~29 calls in a ticker.**
  - Secondary hazard: the executor is a context manager, so leaving the `with` calls `shutdown(wait=True)` -- on a `concurrent.futures.TimeoutError` at `:944-948` the `continue` must first **block until the abandoned 150s subprocess finishes**. Same class as source 9's finding.

### A4. Who writes `cycle_history.jsonl`

Sole writer: `backend/services/cycle_health.py::CycleHealthLog`, `threading.Lock` + `O_APPEND`.
- `record_cycle_start` (`:264-290`) -- writes the `started` row; called at `autonomous_loop.py:371`.
- `record_cycle_end` (`:292-341`) -- writes the terminal row; called **only** from the `finally` at `autonomous_loop.py:1746`. So yes, there IS a `finally` and it does guarantee a terminal row for every in-process exit path. It does **not** cover SIGKILL (F3b).
- Both also stamp `handoff/.cycle_heartbeat.json` via `_write_heartbeat` (`:404-407`).

### A5. Existing watchdog / alerting surface -- WHICH HALF IS TRUE

**Main's audit basis said "per-cycle P1 paging and terminal rows ALREADY work; the gap is the aggregate." Measured verdict: that is CORRECT, and the step's framing sentence "The failure is invisible unless someone hand-reads a jsonl" is FALSE.**

Per-cycle P1 paging fired and was **delivered** on all three timeout days (verbatim from backend.log):

```
2026-08-04 22:00:01,055 ERROR   autonomous_loop  Paper trading cycle TIMED OUT after 7200s
2026-08-04 22:00:01,451 WARNING alerting  alert bot-token fallback delivered=True source=autonomous_loop title='Autonomous trading cycle timeout'
2026-08-06 22:00:02,086 WARNING alerting  alert bot-token fallback delivered=True source=autonomous_loop title='Autonomous trading cycle timeout'
2026-08-07 22:00:02,224 WARNING alerting  alert bot-token fallback delivered=True source=autonomous_loop title='Autonomous trading cycle timeout'
```

Dedup is not the blocker: `AlertDeduper.should_fire` (`alerting.py:77-104`) lets P1 **bypass the consecutive threshold**, gated only by a `repeat_hours` window (default 1h) -- daily cycles are 24h apart, so every day's page fires.

**So why did nobody act? Alert fatigue, not silence.** The same log shows `cycle_health` firing `'Data freshness critical: paper_trades'` + `'... paper_portfolio_snapshots'` **every single hour** (10:30, 11:31, 12:32, 13:33, 14:34, 15:35, 16:35, 17:36, 18:37, 19:38, 20:39, 21:40 on 08-07 alone) -- ~24 P1s/day, because the 1h repeat window re-arms forever while the underlying table stays stale. The one page that mattered was one line in that flood. This is precisely what source 3 warns against ("Every page should be actionable"; "If a page merely merits a robotic response, it shouldn't be a page") and source 5 ("minimize alert count").

**F4 -- the aggregate alarm exists but is structurally blind.** `cycle_health.cycle_heartbeat_alarm` (`cycle_health.py:144-222`):

```python
if parsed.get("status") == "started":
    continue                       # :193-194  <-- ONLY 'started' is skipped
last_row = parsed
break
...
completed_at = last_row.get("completed_at")   # :203
age_sec = (now - completed_dt).total_seconds()
stale = age_sec > threshold_sec               # threshold = 93_600.0 (26h) at :61
```

A `status:"timeout"` / `"running"` / `"error"` row **has a `completed_at`**. So every failed cycle **resets `age_sec` to ~0** and `stale` is False. The alarm measures *"time since a cycle ENDED"*, never *"time since a cycle SUCCEEDED"* -- it is incapable of detecting the actual outage, and worse, it actively **masks** it. It is wired at `slack_bot/scheduler.py:775-807` with correct `_cycle_heartbeat_last_was_stale` state-transition gating; the wiring is fine, the predicate is wrong.

This is exactly the distinction source 7 (2026) draws: track last **successful** time, not error state -- "Status fields reveal 'has never succeeded' conditions undetectable from error metrics alone."

### A6. Fault-injection / test seams for criterion 3

Criterion 3 demands a fault-injected proof, not inspection. The seam already exists and is proven:

- `backend/tests/test_phase_38_2_cycle_start_logging.py:29-32` monkeypatches **both** `cycle_health._HISTORY_PATH` and `cycle_health._HEARTBEAT_PATH` to `tmp_path`. This is the safe pattern -- copy it; do not write to the real `handoff/`.
- `backend/tests/test_cycle_heartbeat_alarm.py` has a row-writing helper (`:37`) and 8 tests including a missing-file case (`:142`).
- `backend/tests/test_phase_66_1_rail_guard.py:191-219` shows `monkeypatch.setattr(ch, "_HISTORY_PATH", tmp_path / "cycle_history.jsonl")` then asserting the parsed last row -- the exact assertion shape a terminal-row proof needs.
- `tests/services/test_cycle_failure_alerts.py` is the phase-23.2.18 regression guard for this same class, with an `_isolated_kill_switch_audit` fixture (`:47-57`) that redirects `kill_switch._AUDIT_PATH` to tmp. **Reuse that fixture** -- see F6.

To fault-inject a *timeout* specifically: monkeypatch `settings.paper_cycle_max_seconds` to a small value (e.g. 0.2) and stub one step to `await asyncio.sleep(5)`, then assert the last `cycle_history` row has `status == "timeout"` and a non-null `completed_at`. No such test exists today (grep for `paper_cycle_max_seconds` in `backend/tests/` returns nothing).

---

## C. Is 7200s arithmetically sufficient? (all figures RE-DERIVED)

### Ticker count -- derived, not assumed

`paper_analyze_top_n: int = Field(5, ...)` at `backend/config/settings.py:403`. The live funnel row for 08-07 in `cycle_history.jsonl` confirms it: `"new_to_analyze": 5, "reeval_tickers": 1` -> **6 tickers**, run at concurrency **3**.

### Rail measurements over the WHOLE cycle window

Window = the full cycle 2026-08-07 18:00:00Z..20:00:02Z (= 20:00..22:00 CEST; backend.log stamps are CEST). Not a tail sample.

| Metric | Measured |
|---|---|
| `claude_code_invoke` starts | 176 |
| successes | 144 |
| `subprocess timeout after 150s` | 31 |
| **timeout rate** | **31/175 = 17.7%** |
| success duration (s) | min 45, p25 73, **median 91**, p75 115, p90 134, **max 145** |
| sum of successful subprocess time | 13,508 s |
| sum of timed-out subprocess time | 31 x 150 = 4,650 s |
| **total serial subprocess-seconds** | **18,158 s** |
| wall-clock window | 7,202 s |
| **implied average parallelism** | **2.52** |
| rail calls per ticker | 176 / 6 = ~29 |

Main's "122 outcomes, 101 success, 21 timeout, 17.2%" was a narrower window; over the full cycle it is **175 outcomes / 17.7%** -- same magnitude, confirmed. Main's "median 88s" -> **91s** measured. The 67% figure was correctly flagged as a burst artefact.

### Measured per-ticker wall clock (from the interleaved log)

| Ticker | Dispatch | Finish | Duration |
|---|---|---|---|
| CRWD | 20:09:19 | 20:45:35 | 2,176 s |
| DELL | 20:09:16 | 20:50:24 | 2,468 s |
| PANW | 20:09:17 | ~20:52 | ~2,570 s |
| HPE | 20:45:35 | ~21:24:48 (slot handoff to NTAP) | ~2,353 s |
| HUM | 20:50:24 | (not observed complete) | -- |
| NTAP | 21:24:48 | **never** (macro agent still running 21:43:50) | >2,113 s at cutoff |

Mean completed-ticker duration ~**2,330 s (38.8 min)**.

### The arithmetic

- Step 1 (screening): 20:00:01 -> 20:09:16 = **555 s**.
- Analysis started 555 s into the cycle. NTAP -- the 6th and last ticker -- did not get a semaphore slot until **21:24:48 = 4,487 s** into the cycle (ticker durations are heterogeneous, so the slots do not free in clean waves).
- NTAP at ~2,330 s would finish at ~**6,817 s**; it was still in the Enhanced-Macro stage at 21:43:50 with debate + synthesis + risk still ahead, so realistically ~**7,400-7,600 s**.
- Steps 5-10 then still have to run.

**Verdict: the 7200 s budget is arithmetically INSUFFICIENT for 6 tickers at the measured rail latency -- short by roughly 5-15%.** It is not short by 2x; it misses by minutes. All three timeouts landed at 7200.9 / 7201.5 / 7201.7 s -- the wall, not a hang.

### (a) slowness, (b) hang, or (c) unhandled failure?

**(a) legitimate slowness -- and it IS distinguishable, on this evidence.** Rejecting the alternatives:

- **Not (b) a hang.** Work was continuously progressing: 176 rail invocations spread evenly from 20:01:13 to 21:58:46, 3-4 of 6 tickers ran to completion with real verdicts (`Debate complete: consensus=SELL...`, `Risk debate complete: decision=REJECT`), and the 6th was actively mid-pipeline at the cutoff.
- **Not (c) an unhandled per-ticker failure stalling the gather.** `return_exceptions=True` plus the inner `try/except` make that path structurally impossible (A2), and no ticker's progress flatlined.

**The dominant addressable cost is the 150 s cap itself.** Note the distribution: **p90 = 134 s, max success = 145 s, cap = 150 s.** The success distribution is *truncated right at the cap* -- the 31 "timeouts" are overwhelmingly calls that would have returned at ~155-200 s, not a broken rail. They burn **4,650 s = 26% of all subprocess-seconds for zero output**, and each one that is retried costs another full attempt. At the measured 2.52 parallelism, recovering that 4,650 s is worth **~1,845 s of wall clock** -- comfortably more than the ~400-800 s the cycle was short by.

**Measurement that would settle any residual doubt, at zero risk:** Python 3.14 ships `python -m asyncio ps <backend-pid>` and `pstree <backend-pid>` (source 6). Run against the live backend during the next cycle it prints every task's coroutine stack and awaiter chain -- a hang shows as a task parked on one frame across two samples; slowness shows as the stack advancing. It is read-only and needs no restart. **This is the single highest-value diagnostic to add to the runbook.**

---

## F6 -- OUT-OF-SCOPE BUT MONEY-CRITICAL (P0, file as its own step)

**The kill switch has been latched PAUSED since 2026-08-04T11:43:31Z. Even a cycle that completes will execute no trades.**

- `KillSwitchState` docstring (`backend/services/kill_switch.py:~140`): "the most recent `pause` or `resume` line sets the resume state; if it's `pause` the system re-enters paused on restart."
- `handoff/kill_switch_audit.jsonl`: **last `resume` = 2026-07-27T06:20:38Z**. Every event after it is a `pause`. Last row overall = `{"ts": "2026-08-08T08:35:16.324544+00:00", "event": "pause", "trigger": "manual", "details": {}}`. `handoff/kill_switch_audit_archive/` is empty, so no later row exists anywhere.
- **Live proof it already cost a cycle:** 08-05 is the one day the cycle got through all 6 tickers, and it traded nothing --
  ```
  2026-08-05 21:34:47,387 WARNING autonomous_loop Paper trading: kill-switch active (paused) -- skipping decide/execute
  ```
  ...which is also the `status: "running"` row (F3a) and the nonsense P1 `title='Autonomous trading cycle running'`.
- **Probable contamination source.** 13 `pause` events with `trigger: "manual"` and empty `details` were written between **2026-08-08T07:26:28Z and 08:35:16Z** -- a 70-minute burst with zero resumes, coinciding with this morning's phase-85.5 test work. This is the documented hazard the `_isolated_kill_switch_audit` fixture (`tests/services/test_cycle_failure_alerts.py:47-57`) exists to prevent: *"a 2026-05-05 pytest run wrote 7 spurious pause events into prod, creating a latent boot-paused risk for the next backend restart."* Some suite is writing to the real `kill_switch._AUDIT_PATH`. I did **not** run any test to confirm which -- stated as a strong inference from timestamps, not a measurement.
- **Operator action needed** (not mine to take): resume the kill switch, and find the suite writing to the production audit path.

---

## Recommendation -- smallest change set for criteria 1-4

Nothing below touches order construction, position sizing, or risk logic. Flag-gating column marks criterion 5.

### R1 -> criterion 4 (aggregate staleness as a pageable signal). **EXTEND, do not rebuild.**

The alarm, its dispatcher, its cron wiring and its 8 tests all exist. The bug is one predicate.

- In `backend/services/cycle_health.py::cycle_heartbeat_alarm` (`:190-205`), select the last row with **`status == "completed"`** instead of the last row that is merely not `"started"`. Return it as `last_completed_at` / `age_sec` (names already correct) and add `consecutive_non_completions` (count of terminal rows newer than the last completed one) to the verdict + the P1 `details`.
- Keep `_CYCLE_HEARTBEAT_STALE_SEC = 93_600.0` (26h). Sources 5 and 7 independently recommend 2-3x the schedule interval; the cycle is daily-on-weekdays, so 26h = ~1.08 intervals is *aggressive* rather than lax -- it is defensible only because the existing weekday gate (`should_alarm = stale and is_weekday_et`) suppresses weekend false positives. If it proves noisy, move to 2 intervals (48h), not below 26h.
- **Do not add a new cron, a new file, or a new alert source.** `slack_bot/scheduler.py:775-807` already has correct state-transition gating.
- **Flag-gating: NOT required** (criterion 5 exemption argument). This is a monitoring predicate with no trading blast radius, and shipping it dark reproduces the exact defect -- an alarm that cannot fire. Ship it live; the risk is a page, not a trade. If Main wants belt-and-braces, gate only the *threshold* value, never the predicate.

**Bonus, ~free:** page (or at least P3) on `cycle_health.orphan_rows()` being non-empty, which covers the SIGKILL case (F3b) that no `finally` can ever reach. This is the dead-man's-switch shape from source 8.

### R2 -> criterion 3 (terminal row ALWAYS written, fault-injected proof)

- **Fix F3a**: at `backend/services/autonomous_loop.py:1326`, set `summary["status"] = "halted"` before `return summary`. Add `"halted"` to the `_final_status not in (...)` no-P1 set at `:1777` (a deliberate kill-switch halt is not a failure) -- or keep the P1 with an honest title. This removes both the bogus `"running"` terminal rows and the nonsense `'Autonomous trading cycle running'` page.
- **Belt-and-braces in the `finally`**: immediately before `record_cycle_end` at `:1746`, normalise `status == "running"` -> `"interrupted"`. Any future early-return that forgets a status then lands on a truthful terminal value instead of the initializer from `:362`.
- **Proof (this is the deliverable, not the fix)**: a new test that monkeypatches `cycle_health._HISTORY_PATH` **and** `_HEARTBEAT_PATH` to `tmp_path` (copy `test_phase_38_2_cycle_start_logging.py:29-32`), sets `settings.paper_cycle_max_seconds` small, injects a slow step, and asserts the last row has `status == "timeout"` with non-null `completed_at`. Add a sibling for the kill-switch-halt path asserting `status == "halted"` and **not** `"running"`.
- **Mutation guard**: revert the `:1326` line and confirm the halt test goes red. A guard that cannot fail does not count.
- **Flag-gating: no.** Both are status-string corrections inside an already-executing path.

### R3 -> criterion 2 (one ticker cannot stall the group)

- Wrap the `_run_single_analysis` call inside `_run_and_persist_one` (`backend/services/autonomous_loop.py:1129-1138`) in `async with asyncio.timeout(per_ticker_budget):` and catch `TimeoutError` alongside the existing `except Exception`, returning `None` exactly as the failure path already does. Budget: derive as `paper_cycle_max_seconds / 2` or a new `paper_ticker_max_seconds` defaulting to ~2700 s (measured mean 2,330 s + ~15% headroom).
- **Do NOT migrate the gathers to `TaskGroup`.** Source 1/2 rightly prefer `TaskGroup` for new code, but its defining behaviour is "the first time any of the tasks... fails with an exception... the remaining tasks in the group are cancelled" -- that is the **opposite** of what this cycle wants, where a bad ticker must not cost the good ones. `gather(return_exceptions=True)` is the correct primitive here and is already in place. Recommending `TaskGroup` would be a regression dressed as modernisation.
- **State the limit honestly in the contract**: per source 9, this bounds the *await*, not the *thread*. The abandoned ticker's subprocess keeps running to completion (measured: 3m10s of orphan work past the 08-07 timeout). The gather is unblocked; the machine still pays. Do not claim the timeout "kills" the ticker.
- **Flag-gating: YES.** New default-OFF `paper_ticker_timeout_enabled`; flag-absent must be byte-identical to flag-False.

### R4 -> criterion 1 (the cycle actually completes). Config-only; pick from measured levers.

Ranked by (wall-clock recovered) / (risk):

1. **Raise the cc_rail cap `timeout_s` 150 -> 210** in `ClaudeCodeClient.__init__` (`claude_code_client.py:593`). `recommended_step_timeout` auto-tracks at `+30` (`:600`), so the orchestrator budget follows to 240 with no second edit. Rationale is measured, not guessed: p90 = 134 s and max success = 145 s against a 150 s cap -- the distribution is truncated at the cap, so the 17.7% "timeouts" are mostly slow successes. Recovers up to **4,650 s of serial time (~1,845 s wall)** and removes the retry multiplier behind it. Highest value, lowest risk.
2. **Raise `paper_cycle_max_seconds` 7200 -> 10800** (`settings.py:33`). Directly closes the measured ~5-15% shortfall. Safe against source 4's overlap warning: the schedule is daily (86,400 s period), so 3 h is ~12.5% of the period -- nowhere near stacking. Note the `settings.py:33` description still says "Read by ... autonomous_loop.py:219"; the real site is **`:439`** -- fix the stale anchor while editing.
3. **Lower `paper_analyze_top_n` 5 -> 3** (`settings.py:403`). Largest single reduction (~40% of analysis wall clock) but it is the only lever that changes *what gets analysed*, i.e. it narrows the funnel that feeds trade selection. Treat as a last resort and flag it to the operator as a signal-quality tradeoff, not a pure perf knob.

**Recommend levers 1 + 2 together, not 3.** Together they take the required wall clock from ~7,500-8,100 s to comfortably inside a 10,800 s budget with the rail waste removed.
- **Flag-gating: partial.** Levers 1 and 2 are existing settings fields -- changing a default is reversible and observable, and criterion 5's dark-launch intent is aimed at new *behaviour*, not at a timeout constant. State the assumption explicitly in the contract rather than inventing a flag for a number.

### R5 -> the noise problem (not a numbered criterion, but it is why 3 delivered pages were ignored)

`cycle_health`'s freshness alarm re-fires hourly, forever, at P1 (~24/day measured on 08-07). Per sources 3 and 5, that is the definition of a non-actionable page and it is what buried the real signal. Recommend a separate step: escalate-once-then-back-off (or downgrade steady-state red to P3 and page only on the red transition). **Do not fold this into 85.4** -- it is a distinct blast radius and 85.4's tree should stay frozen.

### Sequencing note

R4 lever 1 + R1 are the two changes that would have prevented this outage. R6/F6 (kill switch latched paused) blocks money **regardless of anything in 85.4** -- if only one thing ships today, it is the resume.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **9**
- [x] 10+ unique URLs total (incl. snippet-only) -- **25**
- [x] Recency scan (last 2 years) performed + reported -- 2 material findings
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the request (A1-A6 all answered)
- [x] Contradictions / consensus noted (TaskGroup-vs-gather recommendation deliberately declined; both SRE chapters recorded as negative findings)
- [x] All claims cited per-claim
- [ ] Brief length exceeds the `moderate` <=700-word guide. Deliberate and disclosed: the caller specified 6 internal sub-questions plus a derived arithmetic proof plus a recommendation. Tier depth honoured; word cap overrun.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 16,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Two step premises refuted by measurement. (1) Terminal rows ARE written -- a finally at autonomous_loop.py:1746 wrote status:timeout rows for all three timeouts; the real criterion-3 defect is status FIDELITY (the kill-switch halt returns at :1327 leaving the :362 initializer 'running' as a terminal status). (2) The failure was NOT invisible -- P1 'Autonomous trading cycle timeout' was delivered=True on 08-04/06/07; it was buried under ~24 hourly freshness P1s. The genuine criterion-4 gap: cycle_heartbeat_alarm (cycle_health.py:193) skips only 'started' rows, so a timeout row's completed_at resets age to ~0 and the alarm can never see 'nothing COMPLETED in 8 days'. Criterion 2: exceptions already safe (return_exceptions=True + inner try/except); a hang is not (zero timeouts in _run_single_analysis :1859-2520), and per AnyIO a timeout cannot kill the worker thread. Arithmetic: 6 tickers at concurrency 3, 176 rail calls, 17.7% timeout, median 91s, 18,158 serial subprocess-seconds at 2.52 parallelism -- needs ~7,500-8,100s vs a 7,200s budget. Verdict (a) legitimate slowness; success max 145s vs a 150s cap means 26% of rail time is truncation waste. SEPARATE P0: kill switch latched paused since 2026-08-04, so a completing cycle still trades nothing.",
  "brief_path": "handoff/current/research_brief_85.4.md",
  "gate_passed": true
}
```
