# Incident Runbook -- pyfinAgent

> Escalation ladder and response procedures for trading-day incidents.
> Referenced by Go-Live Checklist item 4.4.5.2.

## Escalation Ladder

| Level | Trigger | Channel | SLA | Owner |
|-------|---------|---------|-----|-------|
| L1 | Any anomaly detected by Ford | Slack `#pyfinagent-signals` | Immediate (automated) | Ford |
| L2 | P0 incident or no L1 response in 15 min | iMessage to Peder (`+4794810537`) | 5 min response | Ford |
| L3 | No L2 response in 30 min or system-wide outage | Kill signals (`pause_signals()`), await manual intervention | N/A | Ford auto-action |

### L1 -- Slack Alert (Automated)

Ford posts a Block Kit alert to `#pyfinagent-signals` via `send_trading_escalation()` in `backend/slack_bot/scheduler.py`. All incidents start at L1. The alert includes:
- Severity tag (P0/P1/P2)
- Incident title and key details
- Recommended actions

**Code path:** `scheduler.py:send_trading_escalation` -> `formatters.py:format_escalation_alert` -> `app.client.chat_postMessage`

### L2 -- iMessage (P0 Only)

For P0 incidents, `send_trading_escalation()` also sends an iMessage to Peder's phone via the `imsg` CLI tool. This is the "equivalent mobile push" referenced in the checklist.

Two independent iMessage escalation paths exist:
1. **Trading incidents** -- `scheduler.py:send_trading_escalation` (kill switch, drawdown, signal failure)
2. **SLA breaches** -- `sla_monitor.py:send_escalation_alert` (ticket response/resolution SLA)

Both use `subprocess.run(["imsg", "send", "--to", "+4794810537", "--text", ...])`.

### L3 -- Auto-Kill (Last Resort)

If Ford cannot reach Peder after 30 minutes on a P0, Ford calls `pause_signals()` to shut down the APScheduler and stop all signal flow. See `docs/ROLLBACK_PLAN.md` for the full rollback procedure.

## Incident Types

### Kill Switch Triggered (P0)

**What:** The drawdown circuit breaker at -15% fired, blocking all new BUY signals.

**Detection:** `risk_check()` in `signals_server.py` returns `allowed=False` with `drawdown_circuit_breaker` in conflicts.

**Response:**
1. L1+L2 escalation fires automatically
2. Verify the drawdown is real (not a data error) by checking `paper_snapshots` in BigQuery
3. If real: no action needed, kill switch is working as designed. Monitor for recovery.
4. If data error: fix the data source and manually reset the paper trader state

### Drawdown Warning (P1)

**What:** Portfolio drawdown crossed -5% (early warning) or -10% (de-risk threshold).

**Detection:** Watchdog health check or evening digest flags the drawdown level.

**Response:**
1. L1 Slack alert
2. Review open positions for concentration risk
3. Consider reducing position sizes if drawdown worsening

### Signal Generation Failure (P1)

**What:** The autonomous loop failed to produce signals on a trading day.

**Detection:** Morning digest shows no new signals; `signals_log` in BigQuery has a gap.

**Response:**
1. L1 Slack alert
2. Check backend health: `curl http://localhost:8000/api/health`
3. Check orchestrator logs for errors
4. If backend down: restart via `python -m uvicorn backend.main:app --reload --port 8000`
5. If data source down: check Alpha Vantage / FRED / SEC API status

### Backend Unreachable (P0)

**What:** Watchdog health check cannot reach the backend at all.

**Detection:** `_watchdog_health_check` in `scheduler.py` posts `:rotating_light: Backend unreachable`.

**Response:**
1. L1+L2 escalation (P0 because signals cannot flow)
2. Check if the process is running: `ps aux | grep uvicorn`
3. Check for zombie workers: `ps aux | grep python | grep backend`
4. Restart: kill parent AND child workers, then `python -m uvicorn backend.main:app --reload --port 8000`

### SLA Breach (P0/P1)

**What:** A ticket in `#ford-approvals` has exceeded its response or resolution SLA.

**Detection:** `sla_monitor.py:check_active_sla_breaches` runs every 5 minutes.

**Response:**
1. P0 resolution breach: iMessage escalation fires automatically
2. Check the ticket queue: `#ford-approvals` in Slack
3. If Ford is stuck: check `handoff/mas-harness.log` for the blocking reason

## Peder's Response Checklist

When Peder receives an escalation (Slack or iMessage):

1. **Acknowledge** in Slack (reply or react) so Ford knows you've seen it
2. **Assess** -- is this a real incident or a false alarm?
3. **Decide** -- one of:
   - "Monitoring, no action needed" (Ford continues)
   - "Pause signals" (Ford runs `pause_signals()`)
   - "Investigate X" (Ford investigates the specific area)
4. **Follow up** -- check the next morning/evening digest for resolution

## Contact

- **Peder Koppang**: iMessage `+4794810537`, Slack `#ford-approvals`
- **Ford (autonomous agent)**: Slack `#pyfinagent-signals`, runs on main branch
