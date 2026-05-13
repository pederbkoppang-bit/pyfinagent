---
step: 25.L
slug: drawdown-alarm-tiered-thresholds
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.L

## Step ID + masterplan reference

`25.L` -- "Drawdown alarm with tiered thresholds" (P2, harness_required,
no dep).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`. Tiered drawdown alerting reuses the 25.O
alerting infrastructure.

## Hypothesis

Adding a small `drawdown_alarm` module that reads snapshots, computes
current-vs-peak drawdown, and fires per-tier P1 Slack alerts at 3%,
5%, and 10% gives operators a real-time loss signal. Wired at the
cycle-finally block so the check fires once per cycle.

## Success criteria (verbatim from masterplan.json)

> `drawdown_threshold_check_in_morning_digest_or_cycle`
>
> `p1_slack_alert_at_3pct_5pct_10pct_drawdown_tiers`

## Plan steps

1. **NEW `backend/services/drawdown_alarm.py`** with:
   - `DRAWDOWN_TIERS = [("warn_3pct", -0.03, "P2"), ("warn_5pct", -0.05, "P1"), ("critical_10pct", -0.10, "P1")]`
   - `compute_drawdown_from_snapshots(snapshots) -> float | None`
   - `check_drawdown_alarms(snapshots) -> list[tuple[str, float, str]]`
   - `emit_drawdown_alarms(snapshots, *, source="autonomous_loop") -> int`
2. **Wire** in `backend/services/autonomous_loop.py` finally block: on
   `_final_status == "completed"`, fetch latest snapshots and call
   `emit_drawdown_alarms`. Fail-open with WARNING log.
3. **Verifier** `tests/verify_phase_25_L.py` with 5 claims:
   - Claim 1: module + 3 tier constants present.
   - Claim 2: `check_drawdown_alarms` returns 0 entries on a healthy snapshot list.
   - Claim 3: `check_drawdown_alarms` returns 1 entry on -3.5% drawdown (warn_3pct only).
   - Claim 4: `check_drawdown_alarms` returns 3 entries on -12% drawdown (all 3 tiers).
   - Claim 5: behavioral round-trip -- patch `raise_cron_alert_sync`, call
     `emit_drawdown_alarms` with a -6% snapshot series, assert the
     P1 dedup keys `drawdown_warn_3pct` + `drawdown_warn_5pct` were fired.

## Files

| File | Action |
|------|--------|
| `backend/services/drawdown_alarm.py` | NEW |
| `backend/services/autonomous_loop.py` | Wire emit_drawdown_alarms |
| `tests/verify_phase_25_L.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_L.py
```

## Live-check

`Inject 5%+ drawdown via fixture; Slack alert delivered`.
Will write `handoff/current/live_check_25.L.md`.

## Risks + mitigations

- **Risk**: Snapshot peak is volatile early-stage; tiny absolute losses
  trigger 3% alerts when capital is small.
  **Mitigation**: warn_3pct severity is P2; only 5%+ and 10%+ are P1.
- **Risk**: Repeated alerts each cycle until recovery.
  **Mitigation**: AlertDeduper's repeat_hours default suppresses
  duplicate fingerprints in-window.

## References

- `handoff/current/research_brief.md`
- `backend/services/observability/alerting.py::raise_cron_alert_sync`
- `backend/services/perf_metrics.py::compute_max_drawdown`
- `.claude/masterplan.json::25.L`
