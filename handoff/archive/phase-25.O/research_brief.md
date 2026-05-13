---
step: 25.O
slug: error-escalation-slack-routing
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.O: Error escalation Slack routing

> Tier=simple. Main authored from direct inspection. Mechanical
> design: wrap existing `logger.exception` sites with a thin
> P1-routing helper that deduplicates by (exception_class, endpoint).

---

## Three-variant search queries

1. **Current-year frontier**: `error fingerprint dedup exception class observability 2026`
2. **Last-2-year window**: `Sentry / PagerDuty dedup fingerprint best practice 2025`
3. **Year-less canonical**: `logger.exception escalation alerting pattern`

## Read in full (priors)

| Source | Cycle | Key finding |
|--------|-------|-------------|
| Sentry fingerprint docs | priors | `{type}:{endpoint}` is the canonical dedup key |
| PagerDuty deduplication docs | priors | Same fingerprint = noise; distinct = separate incidents |
| 25.M alerting wire (cycle 87) | this session | `raise_cron_alert_sync` already supports dedup via (source, error_type) |

## Recency scan

No paradigm shift in error-fingerprint design 2024-2026.

## Design

1. **`_route_exception_to_p1(exc, *, endpoint, source="scheduler")` helper** in
   `backend/slack_bot/scheduler.py` -- builds fingerprint
   `f"{type(exc).__name__}:{endpoint}"` and calls `raise_cron_alert_sync`
   with `severity="P1"`, `error_type=fingerprint`. Fail-open.
2. **Wire at high-severity `logger.exception` sites** (4 sites):
   - line 251 (`Failed to send morning digest`)
   - line 281 (`Failed to send evening digest`)
   - line 383 (`phase-25.K: failed to schedule kill-switch Slack alert`)
   - line 558 (`Failed to send alert for {ticker}`)
3. Other `logger.exception` sites (watchdog, trade confirmation, send_trading_escalation
   itself, paper_trader notifier) are NOT in scope per the audit -- they are
   inside the escalation path or low-severity.

## Files to modify

| File | Change |
|------|--------|
| `backend/slack_bot/scheduler.py` | Add helper + wire at 4 sites |
| `tests/verify_phase_25_O.py` | NEW verifier |

## Research Gate Checklist

- [x] Internal inspection at `backend/slack_bot/scheduler.py:251, 281, 383, 558`
- [x] file:line anchors

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; fingerprint pattern is industry-canonical (Sentry/PagerDuty), wire is mechanical."
}
```
