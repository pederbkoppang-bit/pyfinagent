---
step: phase-23.2.18
title: Silent cycle stop + missing operator notification
cycle_date: 2026-05-05
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_18.py'
research_brief: handoff/current/phase-23.2.18-external-research.md (also see phase-23.2.18-internal-codebase-audit.md)
---

# Contract — phase-23.2.18

## Hypothesis

User: "agents has paused its process without notifying me." Forensic state:
- `handoff/cycle_history.jsonl` last row is 2026-04-29; cycles on 04-30,
  05-01, 05-04, 05-05 all started but never wrote a completion row.
- `handoff/.cycle_heartbeat.json` stuck at `{event: start, updated_at:
  2026-05-05T18:00:00.089876+00:00}`.
- `handoff/kill_switch_audit.jsonl` shows `peak_update` events firing on
  each cycle start, but no `pause` events with `trigger=auto`.
- `handoff/logs/backend-watchdog.log` shows kickstart -k events on
  04-30, 05-01, 05-04 (= SIGKILL, bypasses Python `finally`). 05-05
  watchdog log shows ZERO kicks today — but heartbeat is still stuck
  at start.

**Root cause (two-level):**

1. **04-30/05-01/05-04 hangs:** the cycle blocked the asyncio event
   loop -> /api/health stopped responding -> watchdog kickstart -k -> SIGKILL
   bypasses the `try/finally` in `autonomous_loop.py:505-519`, leaving
   no completion row.

2. **05-05 hang (NEW silent mode after phase-23.1.23):** that fix wrapped
   blocking trader.* calls in `asyncio.to_thread()` to free the event
   loop (so /api/health stays responsive and watchdog stays silent).
   But it added NO per-call timeouts on the 13 to_thread sites in
   the cycle path. A stuck yfinance / BQ call now hangs the
   `await asyncio.to_thread(...)` indefinitely. Event loop alive
   (watchdog happy), cycle never advances, no completion record,
   no notification. This is observed today.

**Compounding bug:** `backend/services/observability/alerting.py:127-129`
calls `send_trading_escalation(severity=..., title=..., details=...)`,
but `send_trading_escalation` (`backend/slack_bot/scheduler.py:205`) is
`async def` AND requires `app: AsyncApp` as the first argument. The call
is missing both the `await` and the `app` argument. Any cron alert
routed through `raise_cron_alert` raises TypeError into the fail-open
`except` and is silently dropped. This is why the user has never been
notified by any of the existing alert paths.

## Research-gate summary

- External brief: `handoff/current/phase-23.2.18-external-research.md`.
  8 sources read in full (Python asyncio docs, SuperFastPython timeouts,
  AnyIO threads, OneUptime dead-man's-switch 2026, dev.to async patterns,
  asyncio pitfalls, Cronitor heartbeat, Seifrajhi). 18 URLs collected.
  Recency scan 2024-2026. `gate_passed: true`.
- Internal audit: `handoff/current/phase-23.2.18-internal-codebase-audit.md`.
  7 files inspected with file:line anchors for every claim.

## Immutable success criteria (verbatim — DO NOT EDIT)

1. The autonomous cycle fires a Slack notification on ANY non-`completed`
   terminal status (error, timeout, killed). Notification names cycle_id,
   status, and the failing step / error.
2. The cycle has an outer `asyncio.timeout(...)` ceiling so a stuck
   thread cannot hang the cycle indefinitely. On timeout, status is
   recorded and the operator is notified.
3. `raise_cron_alert` no longer drops alerts. It is fixed (or replaced)
   to call a working webhook path that does NOT depend on the
   separate-process slack_bot daemon.
4. `scripts/launchd/backend_watchdog.sh` posts a Slack alert to
   `slack_webhook_url` BEFORE issuing `launchctl kickstart -k`, so
   SIGKILL does not happen silently.
5. `kill_switch.pause()` invoked with `trigger` not in the
   manual/test/bench allowlist posts an operator alert. Manual/test
   triggers do NOT alert.
6. Regression test `tests/services/test_cycle_failure_alerts.py`
   monkey-patches the webhook helper, exercises the error / timeout /
   auto-pause paths, and asserts the helper was called with the
   correct payload on each.
7. `python -c "import ast; ast.parse(open(PATH).read())"` passes for
   every modified `.py` file.
8. `python tests/verify_phase_23_2_18.py` exits 0.

## Plan steps

1. Make `raise_cron_alert` async and route via `backend/tools/slack.send_notification`
   (already async, webhook-based, no AsyncApp coupling). Add a sync
   convenience wrapper `raise_cron_alert_sync` that schedules the
   coroutine via `asyncio.run` (no loop) or `asyncio.create_task`
   (loop running) for use from sync paths.
2. In `backend/services/autonomous_loop.py`:
   - Wrap the `try` block (line 108) with
     `async with asyncio.timeout(getattr(settings, "paper_cycle_max_seconds", 1800)):`.
   - On TimeoutError catch -> set status="timeout", continue to
     finally.
   - In the existing `except Exception` and after the finally writes
     the cycle row, if status != "completed" call
     `await raise_cron_alert(...)` with cycle_id + status + error.
3. In `backend/services/cycle_health.py.record_cycle_end`: when
   `status` is one of `{"error","timeout","killed","skipped"}`,
   defense-in-depth fire-and-forget alert. Schedules the async
   helper via the sync wrapper so this code path remains
   sync-callable.
4. In `scripts/launchd/backend_watchdog.sh`:
   - Read `SLACK_WEBHOOK_URL` from `backend/.env` (grep, not source —
     avoid sourcing the env into a wrapped shell).
   - `curl -X POST` a JSON `{text: "..."}` payload to the webhook
     BEFORE the `launchctl kickstart -k` line, with -m 5 timeout.
5. In `backend/services/kill_switch.py.pause()`: when `trigger` is not
   in `{manual, test, test-pre, bench-1, bench-2, bench-3}`, fire the
   sync wrapper alert.
6. Add `tests/services/test_cycle_failure_alerts.py`:
   - Monkey-patch `backend.tools.slack.send_notification` to a capture.
   - Drive `record_cycle_end(status="error", ...)` -> assert capture
     called once.
   - Drive `kill_switch.pause(trigger="auto")` -> assert capture
     called.
   - Drive `kill_switch.pause(trigger="manual")` -> assert capture
     NOT called.
7. Add `tests/verify_phase_23_2_18.py`: `ast.parse` the 5 modified
   files; assert `asyncio.timeout` block exists in autonomous_loop.py;
   assert `raise_cron_alert` is async; assert
   webhook curl line exists in backend_watchdog.sh; assert
   pause-allowlist guard exists in kill_switch.py.
8. Append `handoff/harness_log.md` cycle entry AFTER Q/A PASS, BEFORE
   any masterplan status flip (per `feedback_log_last.md`).

## Immutable verification command

```bash
source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_18.py
```

Must exit 0.

## Out of scope (deferred)

- Cooperative thread cancellation via AnyIO `from_thread.check_cancelled()`
  — overkill for the user's notification need; per-call timeouts at
  yfinance/BQ level are a phase-2 hardening.
- Stale-heartbeat APScheduler detector — the cycle now alerts on its
  own status, which closes the immediate gap.
- Refactor of `send_trading_escalation` to drop AsyncApp coupling — the
  webhook escape hatch is sufficient; Bolt path stays intact for the
  slack_bot process.

## Backwards compatibility

- `raise_cron_alert` was non-functional (silently dropped on TypeError);
  changing its signature to async is a strict improvement. Callers from
  sync paths use the new sync wrapper.
- Outer `asyncio.timeout` is purely additive — completed cycles
  unaffected.
- Watchdog curl is best-effort with `-m 5`; failure cannot block
  kickstart.
- kill_switch alert is gated by trigger allowlist; no spam during
  test/bench runs.

## References

- `handoff/current/phase-23.2.18-external-research.md`
- `handoff/current/phase-23.2.18-internal-codebase-audit.md`
- `backend/services/autonomous_loop.py:91-95, 505-519`
- `backend/services/cycle_health.py:79-112`
- `backend/services/observability/alerting.py:111-138` (the broken alert path)
- `backend/slack_bot/scheduler.py:205` (the AsyncApp-coupled escalation)
- `backend/tools/slack.py` (the working webhook helper)
- `scripts/launchd/backend_watchdog.sh:55-58`
- `backend/services/kill_switch.py:115`
- Anthropic, *Harness design for long-running apps*
- Python asyncio docs (3.14): `asyncio.timeout`
- AnyIO threads docs: forced cancellation impossibility
- OneUptime *Heartbeat and Dead Man's Switch* (2026-02-06)
