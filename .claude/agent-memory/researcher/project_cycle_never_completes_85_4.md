---
name: project-cycle-never-completes-85-4
description: phase-85.4 P0 engine health -- BOTH step premises refuted (terminal rows exist; P1 pages were delivered); the real gap is a staleness predicate that treats a timeout row as a heartbeat, plus a kill switch latched paused
metadata:
  type: project
---

Phase-85.4 ("autonomous cycle has not completed since 2026-07-31"). Researched
2026-08-08. **Two of the step's stated premises were REFUTED by measurement.**

1. **"A non-completing cycle may not write a terminal row."** FALSE. The
   `finally` at `backend/services/autonomous_loop.py:1746` calls
   `record_cycle_end` on every in-process exit path, and
   `handoff/cycle_history.jsonl` carries `"status": "timeout"` rows for 08-04,
   08-06 and 08-07. `asyncio.timeout` converts `CancelledError` -> `TimeoutError`
   INSIDE the context manager, so the writer is downstream of the conversion and
   is never itself cancelled. The real defect is status **fidelity**: the
   kill-switch halt path `return summary` at `:1327` never sets a terminal
   status, so the `:362` initializer `{"status": "running"}` is written as a
   terminal row AND fires a P1 titled *"Autonomous trading cycle running"*.
2. **"The failure is invisible unless someone hand-reads a jsonl."** FALSE.
   `alert bot-token fallback delivered=True ... title='Autonomous trading cycle
   timeout'` fired on all three days. P1 bypasses `AlertDeduper`'s consecutive
   threshold (`alerting.py:77-104`), gated only by a 1h repeat window, and daily
   cycles are 24h apart. **It was alert FATIGUE, not silence** -- `cycle_health`
   re-fires `'Data freshness critical: paper_trades'` at P1 every single hour
   (~24/day), so the one page that mattered was one line in a flood.

**The genuine gap (criterion 4):** `cycle_health.cycle_heartbeat_alarm`
(`cycle_health.py:193`) skips only rows with `status == "started"`. A `timeout` /
`running` / `error` row HAS a `completed_at`, so every failed cycle **resets
`age_sec` to ~0**. The alarm measures "time since a cycle ENDED", never "time
since a cycle SUCCEEDED" -- structurally incapable of seeing the outage, and it
actively masks it. 26h threshold at `:61`; wiring at
`slack_bot/scheduler.py:775-807` is correct.

**Criterion-2 nuance:** exceptions are ALREADY safe (`return_exceptions=True` at
`:1157/:1164` plus an inner `try/except` at `:1134`), so "one unhandled ticker
failure kills the gather" is refuted. A *hang* is unbounded -- a sweep of
`_run_single_analysis` (`:1859-2520`) for `asyncio.timeout|wait_for|timeout=`
returns **zero matches**. And per AnyIO, a timeout cannot kill the worker thread:
measured 3m10s of orphan `risk_debate` work continuing past the 08-07 cycle death.

**Do NOT recommend `TaskGroup` here.** Its defining behaviour (cancel siblings on
first failure) is the OPPOSITE of what a per-ticker fan-out wants.

**Capacity arithmetic (re-derive; do not reuse):** 6 tickers
(`paper_analyze_top_n=5` + 1 reeval) at concurrency 3 (because
`gemini_model="claude-sonnet-4-6"` starts with `claude-`, `:1098-1103`). Full
08-07 window: 176 rail invokes, 144 success, 31 timeout = 17.7%; median 91s,
p90 134s, **max success 145s against a 150s cap** -- the distribution is
TRUNCATED AT THE CAP, so most "timeouts" are slow successes, and 26% of all
subprocess-seconds are pure waste. 18,158 serial subprocess-seconds / 7,202s
window = 2.52 avg parallelism. Needed ~7,500-8,100s vs a 7,200s budget: verdict
**(a) legitimate slowness**, short by minutes, not a hang.

**Separate P0 found:** the kill switch has been latched **paused** since
2026-08-04T11:43:31Z (last `resume` in `handoff/kill_switch_audit.jsonl` is
2026-07-27T06:20:38Z; archive dir empty). On 08-05 -- the one day all 6 tickers
finished -- the cycle logged `kill-switch active (paused) -- skipping
decide/execute` and traded nothing. 13 `manual` pauses with empty `details` were
written 2026-08-08 07:26-08:35Z, matching the documented test-contamination
hazard that `tests/services/test_cycle_failure_alerts.py:47-57`'s
`_isolated_kill_switch_audit` fixture exists to prevent.

**Traps for the next session:**
- `backend/tests/test_autonomous_loop_integration.py` is a FALSE FRIEND -- it
  tests `backend/autonomous_loop.py` (harness orchestrator), not
  `backend/services/autonomous_loop.py` (the trading cycle).
- `settings.py:33` says `paper_cycle_max_seconds` is "Read by ...
  autonomous_loop.py:219". The real site is **`:439`**. Stale anchor.
- backend.log stamps are **CEST (UTC+2)**; `cycle_history.jsonl` is UTC. A window
  mismatch silently samples the wrong two hours.
- `python -m asyncio ps <pid>` / `pstree <pid>` is NEW in Python 3.14 and this
  repo runs 3.14 -- a live, read-only hang-vs-slowness diagnostic on the running
  backend that needs no restart. Highest-value addition to the runbook.

Related: [[project_away_watchdog_p1_path]], [[project_kill_switch_36_12_traps]],
[[project_cc_rail_live_window_4000_3]], [[feedback_measure_dont_assert_claims]].
