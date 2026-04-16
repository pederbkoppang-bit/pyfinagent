# Incident Runbook -- pyfinAgent Go-Live

> Escalation ladder for incidents during trading hours. Covers: Ford auto-alerts -> Slack notification -> iMessage to Peder -> manual intervention.

## Escalation Ladder

| Level | Trigger | Channel | Response Time | Owner |
|-------|---------|---------|---------------|-------|
| L0 Auto-Recover | Stuck task > 15 min | Automatic kill + iMessage | Immediate | Ford (stuck_task_reaper) |
| L1 Slack Alert | Watchdog health check fails | Slack `#all-pyfinagent` | 5 min | Ford (scheduler watchdog) |
| L2 iMessage Alert | P0 ticket breaches resolution SLA (30 min) | iMessage to Peder | 5 min response SLA | Ford (sla_monitor) |
| L3 Model Failover | Agent throttled or hanging | iMessage to Peder | Immediate auto-failover | Ford (queue_notification) |
| L4 Manual Intervention | Rollback trigger (Sharpe < 0.5) | iMessage + Slack | Per ROLLBACK_PLAN.md | Peder |

## Priority Definitions (SLA Thresholds)

Source: `backend/services/sla_monitor.py` lines 23-30 and `backend/db/tickets_db.py`.

| Priority | Response SLA | Resolution SLA | Examples |
|----------|-------------|----------------|----------|
| P0 Critical | 5 minutes | 30 minutes | Backend down, signal pipeline halted, kill switch triggered |
| P1 Urgent | 15 minutes | 2 hours | Signal delivery delayed, paper trader error, data pipeline stale |
| P2 Standard | 1 hour | 8 hours | Dashboard rendering issue, non-critical API timeout |
| P3 Low | 4 hours | 24 hours | Cosmetic bug, documentation update, feature request |

## Automatic Escalation Services

Four background services run inside the Slack bot process (`backend/slack_bot/app.py` lines 49-63):

### 1. Watchdog Health Check (`scheduler.py`)

- Interval: every `watchdog_interval_minutes` (default 15 min)
- Action: HTTP GET `http://backend:8000/api/health`
- On failure: posts `:warning: Watchdog Alert` to Slack channel
- On unreachable: posts `:rotating_light: Backend unreachable` to Slack channel
- Does NOT page Peder directly; relies on SLA monitor for escalation

### 2. SLA Monitor (`sla_monitor.py`)

- Interval: every 5 minutes
- Action: scans non-resolved tickets, computes elapsed time vs SLA thresholds
- On P0 resolution breach: sends iMessage to Peder via `imsg send` CLI
- Alert format: ticket number, priority, breach type, elapsed time, message snippet
- Contact: `+4794810537` (Peder's phone, hardcoded in constructor)

### 3. Stuck-Task Reaper (`stuck_task_reaper.py`)

- Interval: every 60 seconds
- Action: checks for IN_PROGRESS tickets older than 15 minutes
- On stuck task: automatically kills the ticket, sends iMessage notification
- Prevents agent hangs from consuming SLA budget silently

### 4. Queue Failover Notifications (`queue_notification.py`)

- Trigger: model rate-limited or hanging during ticket processing
- Action: sends iMessage to Peder about the failover
- Format: which agent failed, which agent is taking over, ticket number

## Incident Response Procedures

### Backend Down (P0)

1. Watchdog detects failure within 15 min (configurable interval)
2. Slack alert posted to `#all-pyfinagent`
3. If a P0 ticket is created and breaches 30-min resolution SLA, iMessage fires
4. Peder response: SSH to server, check `docker logs` / `systemctl status`
5. Restart: `source .venv/bin/activate && python -m uvicorn backend.main:app --reload --port 8000`
6. Verify: `curl -sf http://localhost:8000/api/health`

### Signal Pipeline Halted (P0)

1. Morning/evening digest fails to post (scheduler error logged)
2. Watchdog may or may not catch this (depends on backend health vs scheduler health)
3. If no signal published by market close, this is a missed trading day (4.4.2.4 violation)
4. Peder response: check Slack for the morning digest, then:
   - `python -c "from backend.slack_bot.scheduler import _scheduler; print(_scheduler.get_jobs())"`
   - Restart Slack bot if scheduler is dead: `pkill -f "backend.slack_bot.app" && python -m backend.slack_bot.app`

### Kill Switch Triggered (P0)

1. `risk_check` blocks all BUY signals when portfolio drawdown >= -15%
2. This is an automatic safety mechanism, not a bug
3. Paper trading continues in SELL/HOLD mode only
4. Peder response: review portfolio on the Paper Trading tab
5. If the drawdown is a data error: investigate data pipeline, fix, and the kill switch auto-clears
6. If the drawdown is real: wait for recovery or invoke ROLLBACK_PLAN.md

### Agent Hanging (P1)

1. Stuck-task reaper detects IN_PROGRESS ticket > 15 min
2. Reaper auto-kills the ticket and sends iMessage
3. If the agent is consistently hanging: check rate limits, model availability
4. Peder response: usually no action needed (reaper handles it)
5. If repeated: check `backend/services/ticket_queue_processor.py` for circuit breaker state

### Data Pipeline Stale (P1)

1. Morning digest shows stale prices or missing macro data
2. Check BigQuery tables: `pyfinagent_data.historical_prices`, `pyfinagent_data.macro_indicators`
3. Check data source APIs: Alpha Vantage, FRED, SEC EDGAR
4. If API key expired: rotate in `backend/.env` and restart
5. If API rate limited: wait for reset window

### Slack Bot Disconnected (P2)

1. Socket Mode WebSocket drops (network issue, Slack outage)
2. Slack Bolt auto-reconnects in most cases
3. If reconnect fails: restart `python -m backend.slack_bot.app`
4. If Slack is down: signals still execute but are not delivered to Slack; check Paper Trading tab directly

## iMessage Bridge Details

The iMessage bridge uses the `imsg` CLI tool (macOS-only) for sending escalation alerts.

- Command: `imsg send --to +4794810537 --text "<alert message>"`
- Timeout: 10 seconds per send attempt (sla_monitor), 5 seconds (queue_notification)
- Fallback if `imsg` unavailable: alerts log to stderr only; Peder must monitor Slack directly
- Incoming: `scripts/imsg_responder.py` listens for iMessage replies and routes to agents

## Contact Information

| Role | Name | Channel |
|------|------|---------|
| System Owner | Peder Koppang | iMessage: +4794810537, Slack: @Peder |
| Autonomous Agent | Ford | Slack: `#ford-approvals`, `#all-pyfinagent` |

## Post-Incident Review

After any P0 or P1 incident:

1. Write a brief entry in `.claude/context/known-blockers.md` under RESOLVED (if resolved) or STILL ACTIVE
2. If the incident triggered a rollback, follow the re-approval gate in `docs/ROLLBACK_PLAN.md`
3. Review SLA compliance stats: `SLAMonitoringService.get_sla_compliance_stats(hours_back=24)`
4. Update this runbook if the incident revealed a gap in the escalation path

## Service Dependency Map

```
Backend (FastAPI :8000)
  |
  +-- Slack Bot (Socket Mode)
  |     +-- Scheduler (APScheduler)
  |     |     +-- morning_digest (cron)
  |     |     +-- evening_digest (cron)
  |     |     +-- watchdog_health_check (interval)
  |     |
  |     +-- Ticket Queue Processor (30s interval)
  |     +-- SLA Monitor (5 min interval)
  |     +-- Stuck-Task Reaper (60s interval)
  |
  +-- iMessage Bridge (imsg CLI)
        +-- imsg_responder.py (inbound)
        +-- sla_monitor.py (outbound alerts)
        +-- queue_notification.py (outbound failover)
        +-- stuck_task_reaper.py (outbound kill notices)
```

## Emergency Contacts and Procedures

If all automated systems fail:

1. Check Slack `#all-pyfinagent` for the last known system status
2. SSH to the server and check process status
3. If the server is unreachable: check cloud provider console (GCP)
4. Revoke `SLACK_BOT_TOKEN` in the Slack admin console to immediately stop all signal delivery
5. This is the emergency stop -- equivalent to pulling the plug
