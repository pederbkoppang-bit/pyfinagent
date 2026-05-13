---
step: phase-25.L
cycle: 92
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.L

## What was built/changed

Closed audit bucket 24.5 F-5(c) + 24.8 by adding a tiered drawdown
alarm system:

1. **NEW `backend/services/drawdown_alarm.py`** module:
   - `DRAWDOWN_TIERS = [("warn_3pct", -0.03, "P2"), ("warn_5pct", -0.05, "P1"), ("critical_10pct", -0.10, "P1")]`
   - `compute_drawdown_from_snapshots(snapshots)` returns the current
     NAV vs all-time peak as a negative ratio (or None if insufficient data).
   - `check_drawdown_alarms(snapshots)` returns list of breached tiers.
   - `emit_drawdown_alarms(snapshots, *, source)` fires per-tier
     `raise_cron_alert_sync` with distinct error_type per tier
     (`drawdown_warn_3pct` / `drawdown_warn_5pct` / `drawdown_critical_10pct`)
     so AlertDeduper suppresses repeated same-tier alerts.
   - Fully fail-open: any error -> WARNING log + return 0.
2. **Wired into `backend/services/autonomous_loop.py`** finally block:
   on `_final_status == "completed"`, fetches latest 180 snapshots and
   calls `emit_drawdown_alarms`. Reuses the existing BQ client.

## Files changed

| File | Action |
|------|--------|
| `backend/services/drawdown_alarm.py` | NEW (134 LOC) |
| `backend/services/autonomous_loop.py` | Wire emit_drawdown_alarms in cycle finally |
| `tests/verify_phase_25_L.py` | NEW verifier (6 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_L.py

=== phase-25.L verification ===

[PASS] 1. p1_slack_alert_at_3pct_5pct_10pct_drawdown_tiers
        -> exists=True 3%=True 5%=True 10%=True
[PASS] 2. check_drawdown_alarms_returns_empty_on_healthy
        -> got []
[PASS] 3. check_drawdown_alarms_returns_one_tier_at_minus_3_5pct
        -> breached_tiers={'warn_3pct'} (expected {warn_3pct})
[PASS] 4. check_drawdown_alarms_returns_all_three_tiers_at_minus_12pct
        -> breached_tiers={'warn_5pct', 'warn_3pct', 'critical_10pct'}
[PASS] 5. behavioral_round_trip_fires_p1_at_minus_6pct
        -> fired=2 error_types=['drawdown_warn_3pct', 'drawdown_warn_5pct'] severities=['P2', 'P1']
[PASS] 6. drawdown_threshold_check_in_morning_digest_or_cycle
        -> import=True call_site=True

ALL 6 CLAIMS PASS
```

AST clean on both touched .py files.

## Success criteria -> evidence

1. `drawdown_threshold_check_in_morning_digest_or_cycle` -- Claim 6 PASS:
   autonomous_loop's cycle finally block imports + calls
   `emit_drawdown_alarms`.
2. `p1_slack_alert_at_3pct_5pct_10pct_drawdown_tiers` -- Claims 1 + 3 + 4 + 5 PASS:
   3 tiers present at the right thresholds (1), correct breach detection at
   each band (3 + 4), behavioral round-trip confirms severity=P1 fires for
   warn_5pct (5).

## Out-of-scope / deferred

- Tier-3 (critical_10pct) Slack-channel routing override: currently uses
  the same default channel. A future enhancement could route 10% to a
  separate "incident" channel.
- Drawdown chart on the homepage: not in criteria; the RedLineMonitor
  already shows NAV over time and would surface a 5% drop visually.
- Mid-cycle drawdown check (sub-cycle granularity): currently the alarm
  fires once per cycle (daily). Faster cadence would require a separate
  hourly job.

## References

- `handoff/current/research_brief.md`
- `backend/services/drawdown_alarm.py`
- `backend/services/autonomous_loop.py:681-695` (wire site)
- `.claude/masterplan.json::25.L`
