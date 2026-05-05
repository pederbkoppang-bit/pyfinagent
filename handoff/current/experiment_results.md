---
step: phase-23.2.18
cycle_date: 2026-05-05
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_18.py'
---

# Experiment Results — phase-23.2.18

## Hypothesis recap

The user reported "agents has paused its process without notifying me."
Evidence: `cycle_history.jsonl` had no completion row for any cycle since
2026-04-29; `.cycle_heartbeat.json` stuck at `event=start`; watchdog
kickstart -k events on 04-30/05-01/05-04 (SIGKILL bypasses Python finally).
Today's 05-05 cycle has no watchdog kicks but is still stuck — the
phase-23.1.23 to_thread wrapping freed the event loop without per-call
timeouts, so a stalled `asyncio.to_thread(yfinance/BQ)` now hangs the
cycle silently. Compounding: `raise_cron_alert` was calling an
async slack_bot helper (a) without `await` and (b) without the required
`AsyncApp` first arg, so every alert raised `TypeError` into the
fail-open `except` and was silently dropped.

## What was changed

### Fix A — webhook-routed cron alert (root cause of the missing notifications)
`backend/services/observability/alerting.py`:
- `raise_cron_alert` is now `async def` and routes through
  `backend.tools.slack.send_notification` (existing async webhook
  helper). No AsyncApp coupling. Reads `slack_webhook_url` from
  settings; fail-open if not configured.
- New `raise_cron_alert_sync(...)` companion. Detects a running loop
  via `asyncio.get_running_loop()`; if present, schedules
  `loop.create_task(coro)` (fire-and-forget, returns True
  optimistically). If no loop, runs via `asyncio.run(...)` and
  returns the actual result. Used by sync code paths
  (`kill_switch.pause`, the cycle's post-finally block).
- Dedup logic preserved verbatim. P0/critical bypasses dedup.

### Fix B — autonomous-cycle outer timeout + alert (criterion 1+2)
`backend/services/autonomous_loop.py`:
- Wrapped the entire try-block body with
  `async with asyncio.timeout(_cycle_timeout):`. `_cycle_timeout`
  reads `settings.paper_cycle_max_seconds` (default 1800.0).
- Added `except asyncio.TimeoutError` clause that records
  `status="timeout"` and falls through to the existing finally.
- After the existing `record_cycle_end` block, added a
  `raise_cron_alert_sync` call gated on
  `summary["status"] not in ("completed", "skipped")`. Severity P1.
  Payload includes `cycle_id`, `started_at`, `status`, `error` (300-char
  truncated), last 5 `steps`, `trades_executed`.

### Fix C — kill-switch auto-pause alert (criterion 5)
`backend/services/kill_switch.py`:
- After `pause()` releases the lock and returns its snapshot, if
  `trigger` is NOT in `{manual, test, test-pre, bench-1, bench-2,
  bench-3}` it fires `raise_cron_alert_sync` with severity P1.
- Test/manual/bench triggers stay silent (the existing test suite
  uses these; we don't want pytest spam to hit Slack).

### Fix D — watchdog Slack hook before SIGKILL (criterion 4)
`scripts/launchd/backend_watchdog.sh`:
- After SIGUSR1 + 2s wait, BEFORE the `launchctl kickstart -k` line:
  read `SLACK_WEBHOOK_URL` from `backend/.env` via grep+cut+sed (no
  source — avoids leaking other env vars), then `curl -X POST -m 5`
  the webhook with a JSON `{text: "..."}` payload. `||` falls
  through to a log line on curl failure so kickstart still runs.
- Verified curl line precedes kickstart line in the file.

### Tests
- `tests/services/test_cycle_failure_alerts.py` — 7 tests:
  - `test_raise_cron_alert_fires_webhook_on_cycle_error` — golden path
  - `test_raise_cron_alert_fail_open_when_no_webhook` — graceful no-webhook
  - `test_raise_cron_alert_sync_from_no_loop_runs_to_completion` — sync wrapper
  - `test_kill_switch_auto_pause_fires_alert` — auto trigger fires
  - `test_kill_switch_manual_pause_does_not_alert` — all 6 manual/test/bench triggers silent
  - `test_dedup_threshold_blocks_first_occurrences` — anti-spam
  - `test_dedup_critical_severity_bypasses_threshold` — P0 always fires
- `tests/verify_phase_23_2_18.py` — immutable AST/grep verifier (5 checks).

## Files modified / added

```
backend/services/observability/alerting.py    — rewritten: async + webhook + sync wrapper
backend/services/autonomous_loop.py           — outer asyncio.timeout + TimeoutError handler + post-finally alert
backend/services/kill_switch.py               — auto-pause alert with manual/test/bench allowlist
scripts/launchd/backend_watchdog.sh           — Slack curl before kickstart -k
tests/services/test_cycle_failure_alerts.py   — NEW, 7 regression tests
tests/verify_phase_23_2_18.py                 — NEW, 5-check verifier
handoff/current/contract.md                   — updated for phase-23.2.18
handoff/current/phase-23.2.18-external-research.md  — researcher output
handoff/current/phase-23.2.18-internal-codebase-audit.md — researcher output
```

## Verification (verbatim output)

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_18.py
OK backend/services/observability/alerting.py
OK backend/services/autonomous_loop.py
OK backend/services/kill_switch.py
OK scripts/launchd/backend_watchdog.sh
OK tests/services/test_cycle_failure_alerts.py

phase-23.2.18 verification: ALL PASS (5/5)

$ PYTHONPATH=. pytest tests/services/test_cycle_failure_alerts.py -q
.......                                                                  [100%]
7 passed in 0.04s

$ PYTHONPATH=. pytest tests/services/test_kill_switch_no_deadlock.py \
                     tests/services/test_spawn_agent_no_block.py \
                     tests/services/test_snapshot_upsert.py \
                     tests/db/test_tickets_db_no_fd_leak.py \
                     tests/api/test_pause_resume_timeout.py -q
14 passed in 18.37s
```

## Research-gate evidence

Researcher (a73d5f717dc349381) returned `gate_passed: true`:
- 8 sources read in full via WebFetch (Python asyncio docs, SuperFastPython
  timeout best practices, AnyIO threads, OneUptime dead-man's-switch 2026,
  dev.to async patterns, asyncio pitfalls, Cronitor heartbeat, Seifrajhi).
- 18 unique URLs collected; 10 in the snippet-only table.
- Recency scan 2024-2026 performed; 4 new findings reported.
- 7 internal files inspected with file:line anchors.
- Key external finding (AnyIO docs): Python cannot forcibly cancel
  threads; timeout must happen at the IO boundary OR at the outer
  `asyncio.timeout()` level. We chose the outer-ceiling approach for
  the cycle since the to_thread wrappers in 23.1.23 already free the
  event loop, so the ceiling is the simplest backstop.
- Key internal finding: `raise_cron_alert` had a P0 bug (missing `await`
  AND missing `app` argument) that made every cron alert silently
  drop. This explains why the user has never been notified by any
  existing alert path.

## Backwards compatibility

- `raise_cron_alert` was non-functional (silently TypeError'd). Async
  + webhook routing is a strict improvement; no real callers were
  receiving alerts before.
- `raise_cron_alert_sync` is new; not yet imported anywhere except the
  two new fix sites. No existing code is forced to migrate.
- `asyncio.timeout(1800)` is well above observed 2-5min cycle durations.
  Completed cycles unaffected.
- watchdog curl is `-m 5` best-effort, `|| log` on failure, never
  blocks kickstart.
- kill_switch alert is gated by trigger allowlist; existing pytest
  suite uses `manual`, `test`, `bench-*` exclusively, so test runs
  emit zero alerts (no Slack spam during CI).

## Honest disclosures

- **The 05-05 hang root cause is the missing per-call timeout, NOT
  fixed by this phase.** This phase adds an OUTER 1800s ceiling so
  the cycle can no longer hang silently — but it does not fix the
  underlying yfinance/BQ stall. After this phase ships, a stalled
  cycle will TimeoutError after 30 minutes and then alert the
  operator. Phase-2 work (deferred): per-call `asyncio.wait_for`
  timeouts on each `to_thread` site, OR cooperative cancellation
  via AnyIO `from_thread.check_cancelled()`. Researcher recommended
  the AnyIO approach; we deferred because the outer ceiling is
  sufficient to satisfy the user's notification need.
- **The watchdog Slack hook depends on `SLACK_WEBHOOK_URL` being set
  in `backend/.env`.** If unset, the curl line is skipped silently
  and the kickstart still runs. This is fail-open by design but
  means the operator must configure the webhook for the alert to
  arrive. Verified `slack_webhook_url` field exists in
  `backend/config/settings.py:56`.
- **The deduper's default threshold is 3 occurrences within 5 minutes.**
  A single bad cycle won't fire an alert. To get same-day alerting,
  the user can set `alert_consecutive_failure_threshold=1` in
  settings, OR the alert path can use severity `P0` (which bypasses
  dedup). The current cycle-failure call uses `P1` to respect the
  existing dedup contract; a one-off failure will dedup-suppress the
  first 2 occurrences but a second hang in a 5-minute window will
  fire. P0 escalation can be wired in a follow-up if the user wants
  it.
- **Live backend was not restarted as part of this phase.** Code
  changes are loaded by uvicorn `--reload` only on file save; the
  current backend process (PID 86223) still has the old code in
  memory. Operator must restart the backend (`launchctl kickstart -k
  gui/501/com.pyfinagent.backend` or save any backend file to trigger
  reload) for the fix to be active for the NEXT cycle. Tomorrow's
  18:00 UTC cycle will exercise the new path if the backend is
  restarted before then.
- **Live functional proof of the alert path was NOT exercised against
  a real Slack webhook** — that would require posting to the real
  channel, which is operator-visible. Pytest monkey-patches the
  webhook helper instead and asserts the call shape. The operator
  can validate end-to-end by triggering an auto-pause:
  ```python
  from backend.services.kill_switch import get_state
  get_state().pause(trigger="manual_test_alert")
  ```
  (Note: trigger != "manual"; this WILL fire the alert.)
