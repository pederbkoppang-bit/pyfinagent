# Phase-23.2.18 Internal Codebase Audit
# Topic: Autonomous loop hang / watchdog kill / missing Slack alert
# Date: 2026-05-05

---

## Files Inspected

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | 865 | Daily cycle orchestrator | Active; all trader.* calls wrapped after phase-23.1.23 |
| `backend/services/cycle_health.py` | 228 | Heartbeat + cycle history writer | Active; no alerting emitter |
| `backend/services/kill_switch.py` | 188 | Pause/resume/flatten state | Active; no Slack on auto-pause |
| `backend/slack_bot/scheduler.py` | 350 | APScheduler jobs + escalation helper | Active; `send_trading_escalation` defined at line 205 |
| `backend/services/observability/alerting.py` | 152 | `raise_cron_alert` dedup wrapper | Active; wraps `send_trading_escalation` with 3-consecutive dedup |
| `scripts/launchd/backend_watchdog.sh` | 62 | External liveness watchdog | Active; no Slack hook, silent kickstart |
| `scripts/launchd/com.pyfinagent.backend-watchdog.plist` | 26 | launchd agent (60s interval) | Active |

---

## 1. All `await` points in `run_daily_cycle` (lines 108-498)

The following is every `await` in the try block. Each is a potential hang site; those marked "unbounded" have no explicit asyncio.timeout wrapper.

| Line | Call | Kind | Timeout? |
|---|---|---|---|
| 117 | `await compute_macro_regime()` | async service call (HTTP/BQ) | None — wrapped in try/except, but no time limit |
| 131 | `await fetch_pead_signals_for_recent_reporters()` | async service call | None |
| 141-146 | `await fetch_news_signals(...)` | async service call | None |
| 153 | `await fetch_sector_events()` | async service call | None |
| 179 | `await asyncio.to_thread(_fetch_ticker_meta, ...)` | sync HTTP + BQ in thread | None — thread runs until done or hangs |
| 197 | `await meta_score_candidates(candidates, regime=regime)` | async (LLM?) | None |
| 216 | `await asyncio.to_thread(trader.get_positions)` | BQ SELECT in thread | None |
| 258 | `await _run_single_analysis(ticker, settings)` | LLM + yfinance (per ticker, loops) | None |
| 270 | `await _persist_lite_analysis(analysis, bq)` | BQ INSERT in thread | None |
| 281 | `await _run_single_analysis(ticker, settings)` | same, re-eval loop | None |
| 288 | `await _persist_lite_analysis(analysis, bq)` | BQ INSERT | None |
| 300 | `await asyncio.to_thread(trader.mark_to_market)` | ~42 blocking ops (14 pos x yfinance + BQ DML) | None |
| 307 | `await asyncio.to_thread(trader.check_and_enforce_kill_switch)` | BQ + kill_switch logic | None |
| 315-320 | `await asyncio.to_thread(trader.mark_to_market)` + `save_daily_snapshot` (kill-switch path) | same | None |
| 328 | `await asyncio.to_thread(trader.get_positions)` | BQ SELECT | None |
| 346 | `await asyncio.to_thread(_fetch_ticker_meta, ...)` | legacy sector enrichment | None |
| 392-399 | `await asyncio.to_thread(trader.execute_sell, ...)` (in loop per order) | BQ DML + optional yfinance | None |
| 415-428 | `await asyncio.to_thread(trader.execute_buy, ...)` (in loop per order) | BQ DML + optional yfinance | None |
| 440 | `await asyncio.to_thread(trader.mark_to_market)` | final mark-to-market | None |
| 441 | `await asyncio.to_thread(trader.save_daily_snapshot, ...)` | BQ INSERT | None |
| 451 | `await _learn_from_closed_trades(...)` | BQ reads + optional LLM | None |

### Critical observation: `asyncio.to_thread` does NOT grant cancellability

`asyncio.to_thread` moves the sync function to a ThreadPoolExecutor thread. If that thread blocks (yfinance stalled, BQ connection hung, network timeout not set), the `await` on the coroutine side will also block indefinitely. There is NO per-call timeout on any of the 15+ `asyncio.to_thread` calls in the loop. Python cannot forcibly kill a thread — only the await on the event loop can be cancelled, and even then the thread continues running until the blocking call returns.

The phase-23.1.23 fix moved the calls out of the event loop's hot path (freeing `/api/health` to respond), but it did NOT add per-operation timeouts. If any single thread op hangs (e.g., yfinance stalls on a quote fetch, BQ connection pool exhausted), the `await asyncio.to_thread(...)` also hangs indefinitely. The watchdog probe still passes (event loop is free), but the cycle does not advance.

### Most dangerous sites (new blockers post-23.1.23)

1. **Line 300 — `trader.mark_to_market`**: Fetches live prices from yfinance for all positions (up to 14 calls) then runs BQ DML per position. If yfinance returns a stalled HTTP connection (no timeout set at the yfinance `Ticker.history()` level), the thread hangs. The event loop is free so the watchdog does NOT kick — but the cycle never progresses past Step 5.

2. **Line 258-288 — `_run_single_analysis` loops**: Each loop iteration calls `_run_claude_analysis` which does `stock.info` + `stock.history(period="3mo")` via yfinance (sync, no timeout) then `client.messages.create(...)` via Anthropic SDK (wrapped in `asyncio.to_thread` at line 653). No overall per-ticker timeout. If yfinance stalls on 1 of 10 tickers, the cycle stalls at that iteration.

3. **Line 117 — `compute_macro_regime()`**: No timeout. If this hangs the event loop IS blocked (it is a bare `await`, not `to_thread`). However the watchdog would then catch it — but only after 180s (3x60s probe).

4. **Line 440-441 — final mark_to_market + save_daily_snapshot**: These are the last writes before the `summary.update(status="completed")` at line 484. If either stalls, the cycle never writes `status=completed`, `cycle_history.jsonl` never gets a completion row, and the heartbeat stays stuck at `event=start`.

### What phase-23.1.23 fixed vs. what it missed

**Fixed**: Freed the asyncio event loop from blocking ops, so `/api/health` could respond and the watchdog would not fire SIGUSR1 + kickstart during a running cycle. This explains why 04-30, 05-01, 05-04 cycles ended with a kick: the loop was blocking the event loop directly. After 23.1.23 that specific failure mode is gone.

**Not fixed**: Added no per-operation timeouts to any `asyncio.to_thread` call. The threads themselves can still hang indefinitely if underlying network calls stall. With the fix, `/api/health` responds fine so the watchdog does NOT kick. The result: a silently hanging cycle that never completes, heartbeat stuck at `event=start`, no cycle_history row, no alert.

This matches the 05-05 symptom exactly: heartbeat stuck at start, no kick from watchdog (watchdog passes the health check), no cycle_history row.

---

## 2. Phase-23.1.23 commit analysis

Commit `4251fd1d` (2026-05-04, author Ford). Diff confirmed via `git show`.

**What it did**: Wrapped 8 specific `trader.*` call sites in `asyncio.to_thread(...)`:
- `trader.get_positions` (lines 216, 328)
- `trader.mark_to_market` (lines 300, 315, 440)
- `trader.check_and_enforce_kill_switch` (line 307)
- `trader.save_daily_snapshot` (lines 316, 441)
- `trader.execute_sell` / `trader.execute_buy` (lines 392, 415)

**What it did NOT do**:
- No timeout added to any `asyncio.to_thread` call
- `_run_single_analysis` loop calls (lines 258, 281) are also `await` points that call async functions; `_run_claude_analysis` calls yfinance sync + Anthropic SDK sync inside their own `asyncio.to_thread` (line 653), but there is no per-ticker timeout guarding the outer analysis loop
- `compute_macro_regime`, `fetch_pead_signals_for_recent_reporters`, `fetch_news_signals`, `fetch_sector_events` — all bare `await` calls with no timeout; if any stall, the event loop is blocked
- No overall cycle-level timeout (e.g., `async with asyncio.timeout(3600)` wrapping the try block)

---

## 3. Watchdog behavior: signal sequence and alerting

File: `/Users/ford/.openclaw/workspace/pyfinagent/scripts/launchd/backend_watchdog.sh`

Sequence on 3-consecutive-failure threshold (lines 47-61):
1. `pgrep -f "uvicorn backend.main"` — find uvicorn PID
2. `kill -USR1 $PID` — send SIGUSR1 (triggers faulthandler stack dump to backend.log)
3. `sleep 2` — wait for dump to land
4. `launchctl kickstart -k gui/$UID_NUM/$SERVICE_LABEL` — force-restart

The `-k` flag in `kickstart -k` is equivalent to SIGKILL then restart. It does NOT send SIGTERM first; it kills the process immediately. This is confirmed by Apple developer documentation: `kickstart -k` kills any running instance before launching.

**SIGKILL bypasses `finally`**: Python's `finally` blocks run on SIGTERM (graceful) but NOT on SIGKILL. The `cycle_health.record_cycle_end()` call at `autonomous_loop.py:509-519` is inside a `finally` block. When the watchdog fires `kickstart -k`, the process receives SIGKILL, the finally block never runs, and no cycle_history completion row is written.

**No notification before kickstart**: The watchdog emits only `echo` statements to the log file (`handoff/logs/backend-watchdog.log`). There is NO:
- `curl` call to a Slack webhook
- `terminal-notifier` invocation
- Any notification mechanism whatsoever

The kick is entirely silent to the operator unless they are actively watching the log file.

**plist configuration**: `StartInterval=60` (every 60 seconds), `RunAtLoad=true`. Logs go to `handoff/logs/backend-watchdog.log`. The plist is at the correct path and the watchdog is live (kill_switch_audit.jsonl shows kick events on 04-30, 05-01, 05-04). For 05-05 there is no kick event in kill_switch_audit.jsonl, consistent with the `asyncio.to_thread` fix having resolved the watchdog-trigger mechanism — but the cycle still hangs silently.

---

## 4. `kill_switch.py` — Slack alert on auto-pause

File: `/Users/ford/.openclaw/workspace/pyfinagent/backend/services/kill_switch.py`

The `KillSwitchState.pause()` method (line 111) writes an audit log entry with `trigger` and `details` but does NOT emit any Slack alert. There is no call to `send_trading_escalation` or `raise_cron_alert` anywhere in `kill_switch.py`.

This is a gap: if a drawdown or daily-loss breach triggers an auto-pause (trigger=`auto`), the operator learns about it only by reading `handoff/kill_switch_audit.jsonl` or the morning Slack digest.

The `evaluate_breach` function (line 153) also returns a dict of breach flags but does not emit any alert — it is purely informational for the caller (`PaperTrader.check_and_enforce_kill_switch`).

---

## 5. Slack alert infrastructure — existing helper

### `send_trading_escalation` (scheduler.py:205-260)
- **Signature**: `async def send_trading_escalation(app: AsyncApp, severity: str, title: str, details: dict, actions: list[str] | None = None)`
- **Requires**: a `slack_bolt.async_app.AsyncApp` instance as first argument — this is the initialized Bolt app object, NOT a standalone webhook client
- **Import path**: `from backend.slack_bot.scheduler import send_trading_escalation`
- **Channels**: L1 Slack channel (all severities), L2 iMessage to `+4794810537` (P0 only)
- **Issue for autonomous_loop**: `autonomous_loop.py` does not import the Slack bot's `app` object. It runs in a separate FastAPI/APScheduler context. Calling `send_trading_escalation` from `autonomous_loop.py` requires either: (a) importing the Bolt app object, which creates a circular dependency, or (b) using the `raise_cron_alert` wrapper.

### `raise_cron_alert` (observability/alerting.py:111-138)
- **Signature**: `def raise_cron_alert(source: str, error_type: str, severity: str, title: str, details: str) -> bool`
- **Important**: This is a SYNC function. It calls `send_trading_escalation` as a coroutine but does NOT await it — it calls it directly as `send_trading_escalation(severity=..., ...)` without `await` (line 129). This appears to be a bug: calling an `async def` without `await` returns a coroutine object and does nothing.
- **Dedup logic**: 3 consecutive occurrences within 5-minute window before firing; 1-hour repeat suppression; critical severities bypass dedup.
- **Import path**: `from backend.services.observability.alerting import raise_cron_alert`

**Key finding**: `raise_cron_alert` has a latent bug — it calls the async `send_trading_escalation` without `await`, meaning alerts are silently dropped. This needs to be fixed as part of phase-23.2.18.

---

## 6. Stale heartbeat detector — does anything poll and alert?

Search result: NO existing polling detector.

`cycle_health.py::compute_freshness()` (line 180) computes `hb_age_sec` and returns a `band` (green/amber/red) but:
- It is called by the API endpoint `GET /api/paper-trading/freshness` on demand (passive)
- Nothing in the codebase periodically checks `hb_age_sec` and emits an alert

`scheduler.py::_watchdog_health_check` (line 141) probes `/api/health` via httpx and posts to Slack on failure — but this checks the health endpoint, not the cycle heartbeat file. It would catch "backend is down" but NOT "backend is up but cycle is stalled."

There is no active "stale heartbeat" detector anywhere in the codebase that would alert when `.cycle_heartbeat.json` is stuck at `event=start` for more than (say) 90 minutes after the expected 18:00 UTC fire time.

---

## Summary of Gaps

| Gap | Severity | Location |
|---|---|---|
| No per-operation timeout on any `asyncio.to_thread` call | P1 | autonomous_loop.py (15+ sites) |
| No overall cycle-level timeout (e.g., 2-hour ceiling) | P1 | autonomous_loop.py |
| `raise_cron_alert` calls async `send_trading_escalation` without `await` | P0 bug | observability/alerting.py:129 |
| No stale-heartbeat detector/alerter | P1 | cycle_health.py / scheduler.py |
| Watchdog kickstart is silent (no Slack/notification before kill) | P1 | backend_watchdog.sh |
| kill_switch auto-pause emits no Slack alert | P2 | kill_switch.py |
| yfinance calls in `_run_claude_analysis` (line 589-591) have no timeout | P1 | autonomous_loop.py:589-591 |
