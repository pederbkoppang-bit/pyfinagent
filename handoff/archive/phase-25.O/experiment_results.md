---
step: phase-25.O
cycle: 90
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.O

## What was built/changed

Closed audit bucket 24.5 F-5(f) by adding a fail-open
exception-routing helper to `backend/slack_bot/scheduler.py` and
wiring it at 4 high-severity `logger.exception` sites.

1. **`_route_exception_to_p1(exc, *, endpoint, source="scheduler", extra=None)`**
   helper -- builds a Sentry/PagerDuty-canonical fingerprint
   `f"{type(exc).__name__}:{endpoint}"` and calls
   `raise_cron_alert_sync(severity="P1", error_type=fingerprint, ...)`.
   AlertDeduper's existing per-source/error_type window handles dedup.
2. **4 wired call sites**:
   - line 300: morning digest failure
   - line 331: evening digest failure
   - line 434: kill-switch alert scheduling failure
   - line 610: per-ticker alert failure
   Each preserves the existing `logger.exception` (stacktrace in app log)
   AND adds a P1 Slack ping.

## Files changed

| File | Action |
|------|--------|
| `backend/slack_bot/scheduler.py` | NEW `_route_exception_to_p1` helper + 4 call-site wires |
| `tests/verify_phase_25_O.py` | NEW verifier (5 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_O.py

=== phase-25.O verification ===

[PASS] 1. route_exception_to_p1_helper_exists
        -> found=True pos=['exc'] kw=['endpoint', 'source', 'extra']
[PASS] 2. dedup_fingerprint_by_exception_class_plus_endpoint
        -> Fingerprint built as f'{type(exc).__name__}:{endpoint}'
[PASS] 3. high_severity_exceptions_route_to_p1_slack
        -> calls_raise_cron_alert_sync=True severity_P1=True
[PASS] 4. at_least_four_call_sites_wired
        -> call_sites=4 (expected >=4)
[PASS] 5. behavioral_round_trip_helper_fires_p1
        -> Helper invoked raise_cron_alert_sync with expected fingerprint + P1 severity

ALL 5 CLAIMS PASS
```

AST clean on scheduler.py.

## Success criteria -> evidence

1. `high_severity_exceptions_route_to_p1_slack` -- Claim 3 + 4 + 5 PASS:
   helper calls `raise_cron_alert_sync` with `severity="P1"`, wired at
   4 sites; behavioral round-trip confirms invocation kwargs.
2. `dedup_fingerprint_by_exception_class_plus_endpoint` -- Claim 2 + 5 PASS:
   fingerprint is `f"{type(exc).__name__}:{endpoint}"`; behavioral test asserts
   `error_type="ValueError:morning_digest"` for the round-trip.

## Cycle-2 nuance (verifier mutation-resistance fix)

Claim 3's initial regex required `severity="P1"` with double quotes; the
actual code's `ast.unparse(...)` output uses single quotes. Updated the
regex to accept either form -- this is a verifier-side robustness fix,
not a code change.

## Out-of-scope / deferred

- Watchdog/trade-confirmation `logger.exception` sites: not in scope per
  the audit (they're inside the escalation path or low-severity).
- Promoting `send_trading_escalation` itself's logger.exception (line 505)
  -- that's a meta-failure that should NOT recurse into more Slack posts.
- iMessage escalation failure (line 523): out of Slack-routing scope.

## References

- `backend/slack_bot/scheduler.py:26-72` (helper)
- `backend/slack_bot/scheduler.py:300, 331, 434, 610` (wires)
- `backend/services/observability/alerting.py::raise_cron_alert_sync`
