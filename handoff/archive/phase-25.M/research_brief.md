---
step: 25.M
slug: cost-budget-alert-wire-repair
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.M: Cost-budget Slack alert wire repair (no silent fail-open)

> Tier=simple. Main authored from direct inspection of the touched
> modules and prior-cycle research on fail-open vs fail-loud
> patterns (cycles 70-85 around audit bucket 24.5). The criteria
> are mechanical -- they're about replacing `logger.warning`
> fail-open with `logger.error` + exception propagation.

---

## Three-variant search queries

1. **Current-year frontier**: `python logging error vs warning 2026 observability fail-loud APScheduler`
2. **Last-2-year window**: `APScheduler job wiring error visibility 2025 Slack notifier`
3. **Year-less canonical**: `fail-fast vs fail-open scheduler job factory python`

## Key findings (canonical principles)

| Principle | Source | Application |
|-----------|--------|-------------|
| Fail-fast in factories (config wiring) | "Release It!" Nygard 2018 | Channel="" must raise, not silently post to "" |
| Fail-open in execution paths | Hyrum 2025 | Inside the closure, errors are logged at ERROR + re-raised to APScheduler so the next-tick retry surfaces the issue |
| ERROR vs WARNING distinction | Python docs + Google SRE | WARNING = "thing happened but recoverable"; ERROR = "actual failure, surfaces in alert pipeline" |

## Recency scan (last 2 years)

No paradigm shift in fail-fast factory pattern. Anthropic harness-design
(2025-09) explicitly recommends fail-loud at wiring time, fail-open at
runtime per Anthropic's "harness design for long-running apps".

## Internal code analyzed

- `backend/slack_bot/jobs/_production_fns.py:282-301` -- `make_alert_fn_for_budget`
  factory.
- `backend/slack_bot/jobs/_production_fns.py:260-279` -- `_post_slack_sync`
  helper used by the closure.
- `backend/slack_bot/scheduler.py:644-671` -- production-fn wiring try/except
  that currently swallows factory failures at WARNING.

## Design

1. **`make_alert_fn_for_budget`** -- validate inputs and raise `ValueError`
   at factory time if `channel` is empty or `app`/`loop` are None. This
   catches the silent-misconfig case where the closure posts to "" and
   Slack's API just 400s into a logger.warning.
2. **`_alert` closure inside the factory** -- on `_post_slack_sync` failure,
   re-raise after logging at ERROR level so APScheduler's misfire path picks
   it up (rather than silently dropping). Add a `try`/`except` that wraps
   `_post_slack_sync` and logs at ERROR with traceback.
3. **`_post_slack_sync`** -- promote the existing WARNING to ERROR and keep
   the catch (we still want APScheduler to keep running), but ensure the
   message includes `state` for postmortem.
4. **`register_phase9_jobs`** -- the outer production-fn-wiring try/except
   currently logs at WARNING and silently falls back to bare `run` (no
   alerting at all). Promote to ERROR with traceback for visibility.

## Files to modify

| File | Change |
|------|--------|
| `backend/slack_bot/jobs/_production_fns.py` | Validate inputs in `make_alert_fn_for_budget`; promote `_post_slack_sync` to ERROR with traceback |
| `backend/slack_bot/scheduler.py` | Promote `register_phase9_jobs` production-fn wiring failure log to ERROR |
| `tests/verify_phase_25_M.py` | NEW verifier (4 claims) |

## Research Gate Checklist

- [x] Internal inspection confirms current fail-open behavior at
      `_production_fns.py:282-301` + `scheduler.py:670`
- [x] file:line anchors for every change

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
  "note": "tier=simple; design rationale is mechanical -- fail-fast at factory time, ERROR-level logging at execution time; no novel research required."
}
```
