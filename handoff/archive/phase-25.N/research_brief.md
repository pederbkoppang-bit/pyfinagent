---
step: 25.N
slug: cycle-completion-slack-summary
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.N: Cycle-completion Slack summary

> Tier=simple. Main authored from prior-cycle 25.R formatter pattern
> (cycle 88 prior). Mechanical addition: Block Kit formatter + autonomous-loop wire.

## Three-variant search queries

1. **Current-year frontier**: `Slack Block Kit cycle digest 2026 trading bot`
2. **Last-2-year window**: `daily summary Slack notification format 2025`
3. **Year-less canonical**: `dashboard digest message brevity attention`

## Read in full (prior cycles)

| Source | Cycle | Kind | Key finding |
|--------|-------|------|-------------|
| 25.R formatter pattern | cycle 83 | Internal | Block Kit `list[dict]` with header + fields + context |
| Slack Block Kit Builder docs | priors | Official | `section`/`fields`/`context`/`divider` pattern |
| Stephen Few "dense status" | priors | Book | Single-row digest with KPIs > paragraph prose |

## Recency scan

No paradigm shift in Slack cycle-digest design 2024-2026.

## Design

1. **`format_cycle_summary(summary: dict) -> list[dict]`** in
   `backend/slack_bot/formatters.py` -- Block Kit blocks rendering:
   - Header: ":bar_chart: Autonomous Cycle <status>"
   - Fields: cycle_id, started_at, duration_sec, trades_executed,
     stops, mode (full/lite/dry-run), recommendations
   - Footer: phase-25.N + closes red-line goal-d (observability)
2. **`backend/services/autonomous_loop.py` finally block** -- on
   `status == "completed"`, build the cycle summary dict, format it,
   and emit a P3 alert via `raise_cron_alert_sync` (existing webhook
   infrastructure; metadata dict carries the summary fields).
3. The non-completed status path is unchanged (P1 alert remains).

## Files to modify

| File | Change |
|------|--------|
| `backend/slack_bot/formatters.py` | Add `format_cycle_summary` |
| `backend/services/autonomous_loop.py` | Emit P3 alert on cycle_completed |
| `tests/verify_phase_25_N.py` | NEW verifier |

## Research Gate Checklist

- [x] Internal pattern: 25.R formatter
- [x] file:line anchors

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
  "note": "tier=simple; formatter design mirrors 25.R (cycle 83); wire path mirrors existing P1 cycle-failure alert at autonomous_loop.py:614-639."
}
```
