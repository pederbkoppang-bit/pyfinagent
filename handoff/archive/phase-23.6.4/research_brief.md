---
step: 25.P
slug: weekly-autoresearch-slack-summary
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.P: Weekly autoresearch summary Slack notification

> Tier=simple. Main authored from 25.N (cycle 89) cycle-completion
> summary pattern + inspection of meta_evolution/cron.py.

---

## Three-variant search queries

1. **Current-year frontier**: `weekly cron summary slack 2026`
2. **Last-2-year window**: `meta-evolution weekly digest 2025`
3. **Year-less canonical**: `scheduled job completion notification pattern`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| 25.N cycle 89 | this session | Established `format_cycle_summary` + `raise_cron_alert_sync(severity="P3")` pattern for cycle-completion digests |
| meta_evolution/cron.py:87-152 | this cycle | `run_meta_evolution_cycle` returns a results dict with cron_allocations, provider_allocations, archetype_count, errors, duration |
| 25.O cycle 90 | this session | dedup-fingerprint pattern for distinct alert kinds |

## Recency scan

No paradigm shift in scheduled-cycle Slack digest design 2024-2026.

## Design

1. **`format_autoresearch_summary(results: dict) -> list[dict]`** in
   `backend/slack_bot/formatters.py` -- Block Kit (header + 6-field
   section + divider + context). Fields: started_at, finished_at,
   duration_sec, cron_allocations count, provider_allocations count,
   archetype_count, errors count.
2. **`backend/meta_evolution/cron.py::run_meta_evolution_cycle`** -- after
   the final logger.info, emit a P3 alert via raise_cron_alert_sync with
   error_type `meta_evolution_weekly_summary`. Fail-open.

## Files to modify

| File | Change |
|------|--------|
| `backend/slack_bot/formatters.py` | Add format_autoresearch_summary |
| `backend/meta_evolution/cron.py` | Emit P3 alert on completion |
| `tests/verify_phase_25_P.py` | NEW |

## Research Gate Checklist

- [x] Internal: 25.N pattern (cycle 89)
- [x] Internal: meta_evolution/cron.py results-dict shape

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
  "note": "tier=simple; near-clone of 25.N for the weekly autoresearch cron."
}
```
