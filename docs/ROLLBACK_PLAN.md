# Rollback Plan -- pyfinAgent Go-Live

> If live performance degrades below thresholds in the first 2 weeks, stop signals immediately and investigate before resuming.

## Trigger Criteria

| Metric | Threshold | Window | Action |
|--------|-----------|--------|--------|
| Live Sharpe ratio | < 0.5 | Trailing 14-day | Stop signals, investigate |
| Max drawdown | >= -15% | Any point | Kill switch auto-fires (4.4.4.1) |
| Missed trading days | >= 2 consecutive | Rolling | Stop signals, investigate infra |

The primary rollback trigger is **live Sharpe < 0.5 on a trailing 14-day window**. This represents a > 57% degradation from the backtest Sharpe of 1.17 and a > 39% degradation from the paper trading floor of 0.82.

## Stop-Signals Command

### Option A: Graceful pause (preferred)

```bash
# From a Python shell or one-liner on the server running the Slack bot:
source .venv/bin/activate
python -c "from backend.slack_bot.scheduler import pause_signals; pause_signals()"
```

`pause_signals()` shuts down the APScheduler instance, stopping all scheduled jobs (morning digest, evening digest, watchdog) and preventing any new signal dispatches. The Slack bot process stays alive for manual commands but no longer publishes automated signals.

### Option B: Full process kill (if Option A is unavailable)

```bash
# Kill the Slack bot process entirely
pkill -f "backend.slack_bot.app"

# Or if running under systemd/launchd:
# systemctl stop pyfinagent-slack
# launchctl unload ~/Library/LaunchAgents/com.pyfinagent.slack.plist
```

### Option C: Emergency -- disable at source

If neither Option A nor B is reachable (e.g., the server is unresponsive):

1. Revoke the `SLACK_BOT_TOKEN` in the Slack app admin console
2. This immediately disconnects Socket Mode and prevents all message delivery

## Rehearsal Recipe

Run this during launch-week to confirm the pause mechanism works:

```bash
source .venv/bin/activate

# 1. Start the slack bot in background (dry-run context)
python -m backend.slack_bot.app &
SLACK_PID=$!
sleep 3

# 2. Verify scheduler is running
python -c "from backend.slack_bot.scheduler import _scheduler; print('Scheduler running:', _scheduler and _scheduler.running)"

# 3. Pause signals
python -c "from backend.slack_bot.scheduler import pause_signals; result = pause_signals(); print('Pause result:', result)"

# 4. Verify scheduler is stopped
python -c "from backend.slack_bot.scheduler import _scheduler; print('Scheduler running:', _scheduler and _scheduler.running)"

# 5. Clean up
kill $SLACK_PID 2>/dev/null
```

Record the rehearsal date and commit hash as evidence on checklist item 4.4.6.4.

## Investigation Checklist

Before restarting signals after a rollback, check all of the following:

1. **Data pipeline**: Are input signals (prices, fundamentals, macro) fresh and accurate? Check `pyfinagent_data` tables for stale rows.
2. **Model drift**: Compare live signal distribution against backtest signal distribution. Large shifts suggest regime change or feature drift.
3. **Execution quality**: Review paper trades for slippage, missed fills, or timing issues.
4. **External factors**: Was the poor Sharpe driven by an unusual market event (flash crash, circuit breaker, earnings season anomaly)?
5. **Code regression**: Check `git log` for recent changes to `orchestrator.py`, `signals_server.py`, or `portfolio_manager.py`.
6. **Cost model**: Are actual transaction costs materially higher than the $7.14/trade model assumption?

## Re-Approval Gate

After investigation and any fixes:

1. Re-run the paper trading validation (4.4.2) for a fresh 2-week window
2. Confirm paper Sharpe >= 0.82 (70% of backtest)
3. Peder must post a **new** go-live approval in `#ford-approvals` (a fresh 4.4.6.1 sign-off)
4. The previous approval is void -- do not restart signals under the old approval

## Timeline

- **Days 1-14 post-launch**: Monitor daily. Rollback trigger is active.
- **Day 14**: If Sharpe >= 0.5, relax to weekly monitoring. If < 0.5, rollback fires.
- **After rollback**: Minimum 2-week paper re-validation before any restart attempt.
