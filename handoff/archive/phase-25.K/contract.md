# Sprint Contract — phase-25.K — Wire kill-switch state to Slack

**Cycle:** phase-25 cycle 5
**Date:** 2026-05-12
**Step ID:** 25.K
**Priority:** P0

## Research-gate
Reuses phase-24.5 cycle 4 + phase-24.8 cycle 8 researcher gates. Fix per audit F-5(b) + F-2.

## Hypothesis
Wiring `pause_signals()` to call `send_trading_escalation` before shutdown + adding `notify_kill_switch_activated/deactivated` helpers will close the F-5(b) silent-pause bug.

## Success criteria (verbatim)
1. kill_switch_activate_emits_p0_slack_escalation
2. kill_switch_deactivate_emits_p1_slack_escalation
3. pause_signals_calls_send_trading_escalation_before_shutdown

## Plan
1. Add `notify_kill_switch_activated(app, trigger, details)` async helper — P0 escalation with operator-actionable details
2. Add `notify_kill_switch_deactivated(app, reason)` — P1 escalation
3. Modify `pause_signals()` to accept `app: AsyncApp | None = None` param; when provided, fire P0 alert via `asyncio.create_task` BEFORE shutdown
4. Backward compatible: pause_signals without app still works as before (preserves existing callers)
5. Verifier `tests/verify_phase_25_K.py` (7 claims)
6. experiment_results.md
7. Q/A
8. harness_log Cycle 61
9. Flip 25.K

## References
- `docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md` F-5(b)
- `docs/audits/phase-24-2026-05-12/24.8-observability-findings.md` F-2
- `backend/slack_bot/scheduler.py:353-366` (pause_signals — modified)
- `backend/slack_bot/scheduler.py:369-423` (send_trading_escalation — reused)
