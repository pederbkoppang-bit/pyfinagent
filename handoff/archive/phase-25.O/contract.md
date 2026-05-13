---
step: 25.O
slug: error-escalation-slack-routing
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.O

## Step ID + masterplan reference

`25.O` -- "Error escalation Slack routing (logger.exception promotion)"
(P2, harness_required, no dep).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`. Fingerprint pattern is industry-canonical
(Sentry/PagerDuty: `{type}:{endpoint}`).

## Hypothesis

`backend/slack_bot/scheduler.py` has 4 high-severity `logger.exception`
sites that record stacktraces but do NOT route to P1 Slack escalation.
By adding a thin `_route_exception_to_p1(exc, *, endpoint)` helper that
calls `raise_cron_alert_sync` with a `(exception_class, endpoint)`
fingerprint, the morning/evening digest failures, kill-switch alert
scheduling failures, and per-ticker alert failures all surface to
Slack as P1 with dedup, while the existing logger.exception trace
still lands in the application log.

## Success criteria (verbatim from masterplan.json)

> `high_severity_exceptions_route_to_p1_slack`
>
> `dedup_fingerprint_by_exception_class_plus_endpoint`

## Plan steps

1. **`backend/slack_bot/scheduler.py`** -- add module-level helper
   `_route_exception_to_p1(exc, *, endpoint, source="scheduler")` that
   builds `fingerprint = f"{type(exc).__name__}:{endpoint}"` and calls
   `raise_cron_alert_sync(source=source, error_type=fingerprint,
   severity="P1", title=..., details={endpoint, exception_class,
   exception_repr, ...})`. Fail-open (try/except logger.warning).
2. **Wire at the 4 sites**:
   - morning digest (line 251)
   - evening digest (line 281)
   - kill-switch schedule failure (line 383)
   - per-ticker alert failure (line 558)
   Each site keeps its existing `logger.exception` call and adds a
   `_route_exception_to_p1(exc, endpoint="<name>")` call right after.
3. **Verifier** -- `tests/verify_phase_25_O.py` with 5 claims:
   - Claim 1: `_route_exception_to_p1` exists with the right signature.
   - Claim 2: helper builds fingerprint as `f"{type(exc).__name__}:{endpoint}"`.
   - Claim 3: helper calls `raise_cron_alert_sync` with `severity="P1"`.
   - Claim 4: at least 4 call sites in scheduler.py invoke `_route_exception_to_p1`.
   - Claim 5: behavioral round-trip -- patch `raise_cron_alert_sync`, call
     `_route_exception_to_p1(ValueError("x"), endpoint="morning_digest")`,
     assert it was called with `error_type="ValueError:morning_digest"` and
     `severity="P1"`.

## Files

| File | Action |
|------|--------|
| `backend/slack_bot/scheduler.py` | Add helper + wire at 4 high-severity sites |
| `tests/verify_phase_25_O.py` | NEW verifier |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_O.py
```

## Live-check

`Inject scheduled-job exception; Slack alert delivered with dedup`.
Will write `handoff/current/live_check_25.O.md`.

## Risks + mitigations

- **Risk**: Helper itself raises and breaks the digest path.
  **Mitigation**: Helper is fully fail-open -- internal try/except logger.warning.
- **Risk**: P1 dedup window too narrow; flapping creates noise.
  **Mitigation**: AlertDeduper's default repeat_hours handles dedup -- same
  fingerprint inside the window is suppressed.

## References

- `handoff/current/research_brief.md`
- `backend/slack_bot/scheduler.py:251, 281, 383, 558`
- `backend/services/observability/alerting.py::raise_cron_alert_sync`
- `.claude/masterplan.json::25.O`
