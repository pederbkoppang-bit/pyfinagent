# Contract: Phase 4.4.6.3 -- First-week extra monitoring

## Step ID
4.4.6.3

## Checklist item
"First-week monitoring cadence is armed: daily review call scheduled, alert thresholds tightened"

## Scope
Code-side only: add a `FIRST_WEEK_MODE` env-var toggle that tightens two
monitoring thresholds as specified in the checklist:
1. Drawdown de-risk alert: -10% -> -5% (checklist: "drawdown alert at -5% instead of -10%")
2. SLA P3 response: 4 hours -> 1 hour (checklist: "signal miss alert at 1 hour instead of 4")

Calendar scheduling (daily review call days 1-7) is Peder's responsibility
and will be noted as pending, following the pattern of 4.4.5.2 and 4.4.5.5.

## Success criteria
- SC1: `first_week_mode` boolean setting exists in `backend/config/settings.py`, defaults False
- SC2: `sla_monitor.py::get_sla_thresholds()` returns P3 response=3600s (1h) when first_week_mode=True, 14400s (4h) when False
- SC3: `signals_server.py::track_drawdown()` uses derisk_pct=-5.0 when first_week_mode=True, -10.0 when False
- SC4: `get_risk_constraints()` is NOT modified (preserves 4.4.4.4 hardcoded limits)
- SC5: kill switch at -15% is unchanged in both modes
- SC6: drill at `scripts/go_live_drills/first_week_monitoring_test.py` exits 0
- SC7: all existing drills still pass (no regression)

## Files to change
1. `backend/config/settings.py` -- add `first_week_mode` field
2. `backend/services/sla_monitor.py` -- import settings, conditional P3 threshold
3. `backend/agents/mcp_servers/signals_server.py` -- first-week override in track_drawdown
4. `scripts/go_live_drills/first_week_monitoring_test.py` -- new drill (stdlib-only)
5. `docs/GO_LIVE_CHECKLIST.md` -- flip checkbox, add evidence

## Non-goals
- Do NOT modify `get_risk_constraints()` (4.4.4.4 compliance)
- Do NOT implement auto-revert after 7 days (operational procedure, not code gate)
- Do NOT schedule calendar events (Peder's action)
