---
step: phase-25.M
cycle: 87
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.M

## What was built/changed

Closed audit bucket 24.5 F-5(d) by replacing two silent fail-open
paths with fail-loud signaling:

1. **`make_alert_fn_for_budget` factory** -- now validates `app`,
   `loop`, and `channel` at construction time and raises `ValueError`
   if any is missing. Previously a misconfig silently produced a
   non-throwing closure that posted to "" channel.
2. **`_post_slack_sync` runtime catch** -- promoted from
   `logger.warning("alert_fn: Slack post fail-open: %r", exc)` to
   `logger.error("alert_fn: Slack post failed: %r", exc, exc_info=True)`
   so failures surface with traceback in default log views.
3. **`register_phase9_jobs` outer wiring catch** -- promoted from
   `logger.warning(... wiring fail-open ...)` to `logger.error(
   ... wiring failed ..., exc_info=True)`.

## Files changed

| File | Action |
|------|--------|
| `backend/slack_bot/jobs/_production_fns.py` | Validate factory inputs + promote Slack-post error log |
| `backend/slack_bot/scheduler.py` | Promote production-fn wiring error log to ERROR + exc_info |
| `tests/verify_phase_25_M.py` | NEW verifier (5 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_M.py

=== phase-25.M verification ===

[PASS] 1. make_alert_fn_for_budget_raises_loudly_on_wiring_error
        -> Raised ValueError: make_alert_fn_for_budget: channel must be non-empty (got ''); cost-budget alerts would post to empty channel
[PASS] 2. make_alert_fn_for_budget_raises_on_none_loop
        -> Raised ValueError: make_alert_fn_for_budget: loop is required (got None); cost-budget alerts would silently drop
[PASS] 3. scheduler_register_phase9_jobs_logs_error_visibly
        -> Found ERROR-level log with exc_info=True
[PASS] 4. post_slack_sync_logs_error_on_failure
        -> Found logger.error+exc_info=True in _post_slack_sync
[PASS] 5. behavioral_round_trip_logs_error_on_post_failure
        -> Captured ERROR record: alert_fn: Slack post failed: RuntimeError('slack_api_error_for_test')

ALL 5 CLAIMS PASS
```

Backend AST parse: clean.

## Success criteria -> evidence

1. `make_alert_fn_for_budget_raises_loudly_on_wiring_error` -- Claims 1 + 2 PASS:
   factory raises `ValueError` on empty `channel` and on `loop=None`.
2. `scheduler_register_phase9_jobs_logs_error_visibly` -- Claim 3 PASS:
   regex-matched `logger.error(..., exc_info=True)` on the wiring exception path.
   Claim 5 additionally exercises the runtime path end-to-end: a Slack-post
   failure now emits an ERROR record (verified by `_CapturingHandler`).

## Caller safety audit

Production caller (`backend/slack_bot/scheduler.py:663`) always passes a
real `app`, `loop`, and `channel = get_settings().slack_channel_id or ""`.
**If `slack_channel_id` is unset, the factory now raises** -- this surfaces
the misconfig at startup time instead of silently dropping every cost-budget
alert for the lifetime of the process. The outer try/except at
`scheduler.py:670` catches the raised `ValueError` and logs at ERROR with
traceback (criterion 2), so an unconfigured channel is now operator-visible.

No test paths call `make_alert_fn_for_budget` with stubs.

## Out-of-scope / deferred

- Other `make_alert_fn_*` factories (e.g. `make_alert_fn_for_integrity`)
  are not in the masterplan criteria. The same hardening could be applied
  in a follow-up if needed.
- Restoring backwards-compat for callers that intentionally pass empty
  channel: none exist in the codebase.
