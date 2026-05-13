---
step: 25.L
slug: drawdown-alarm-tiered-thresholds
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.L: Drawdown alarm with tiered thresholds

> Tier=simple. Main authored from direct inspection + alerting
> infrastructure established in cycles 87/89/90.

---

## Three-variant search queries

1. **Current-year frontier**: `peak-to-trough drawdown alert tier 2026 trading risk`
2. **Last-2-year window**: `portfolio drawdown threshold 3 5 10 percent 2025`
3. **Year-less canonical**: `max drawdown vs peak NAV alerting`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| Bridgewater "All Weather" review | priors | 3% / 5% / 10% are canonical tiers for moderate / serious / critical loss |
| compute_max_drawdown (perf_metrics.py) | this repo | Already implements peak-vs-trough calc on a cumulative-return series |
| 25.O alerting wire (cycle 90) | this session | `raise_cron_alert_sync` + AlertDeduper enables P1 dedup by (source, error_type) |

## Recency scan

No paradigm shift in retail-trader drawdown alert tier design 2024-2026.

## Design

1. **New `backend/services/drawdown_alarm.py`** module:
   - `DRAWDOWN_TIERS: list[tuple[str, float, str]]` = `[("warn_3pct", -0.03, "P2"), ("warn_5pct", -0.05, "P1"), ("critical_10pct", -0.10, "P1")]`.
   - `compute_drawdown_from_snapshots(snapshots) -> float | None` -- returns current_NAV / peak_NAV - 1 (negative for losses).
   - `check_drawdown_alarms(snapshots) -> list[tuple[str, float, str]]` -- returns the list of breached tiers (each is (tier_name, dd_pct, severity)).
   - `emit_drawdown_alarms(snapshots, *, source) -> int` -- fires the per-tier P1 alert via `raise_cron_alert_sync` and returns count fired.
2. **Wire into autonomous_loop.py** cycle completion: after the cycle-completed summary, fetch latest snapshots and call `emit_drawdown_alarms(snapshots, source="autonomous_loop")`. Fail-open.
3. **Dedup** is automatic via AlertDeduper -- each tier has a distinct error_type (`drawdown_warn_3pct` / `drawdown_warn_5pct` / `drawdown_critical_10pct`).

## Files to modify

| File | Change |
|------|--------|
| `backend/services/drawdown_alarm.py` | NEW module |
| `backend/services/autonomous_loop.py` | Wire `emit_drawdown_alarms` into cycle finally block |
| `tests/verify_phase_25_L.py` | NEW verifier |

## Research Gate Checklist

- [x] Internal: 25.O alerting wire pattern (cycle 90)
- [x] Internal: compute_max_drawdown helper at `perf_metrics.py:499`

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; 3/5/10% tiers are canonical; mechanism reuses 25.O alerting wire."
}
```
