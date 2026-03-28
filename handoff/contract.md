# Phase 2.6.0 Contract — Operational Resilience

## Hypothesis
If we implement self-healing infrastructure, health monitoring, and proper OpenClaw configuration, Ford will maintain >99.5% uptime with automatic recovery from failures and Peder will always be notified of issues via Slack.

## Success Criteria
1. **Gateway self-healing:** LaunchAgent restarts gateway on crash, watchdog cron verifies every 5 min
2. **Service monitoring:** Backend (8000) and frontend (3000) checked in every heartbeat, auto-restarted if down
3. **Slack availability:** Every heartbeat confirms Slack connectivity first; fallback to iMessage if down
4. **Incident logging:** All failures logged to `memory/incidents.md` with timestamp, cause, action, recovery
5. **OpenClaw config optimized:** All settings reviewed per configuration-reference.md, Slack channel config hardened
6. **Health endpoint verified:** `/api/health` returns status + version, polled by sidebar health dot

## Fail Conditions
- Gateway crash not auto-detected within 10 minutes
- Backend/frontend down >15 minutes without notification
- Slack unreachable without fallback notification
- Any config change breaks existing functionality

## Scope
Sections A through I of Phase 2.6.0 in PLAN.md. Research phase is lightweight (operational, not quant).

## Started
2026-03-28 23:02 Oslo
