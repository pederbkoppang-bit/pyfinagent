---
applyTo: "backend/slack_bot/**"
---

# Slack Bot — Bolt + Socket Mode Conventions

## Stack
- Slack Bolt (async), Socket Mode (outbound WebSocket — no public URL needed)
- APScheduler for scheduled jobs (morning digest, anomaly alerts)
- httpx for internal API calls to backend (`http://backend:8000`)
- Block Kit for all message formatting

## Key Modules
- `app.py` — Entry point, `create_app()` + Socket Mode handler. Run via `python -m backend.slack_bot.app`
- `commands.py` — Slash command handlers (`/analyze`, `/portfolio`, `/report`). Registers on the Bolt app.
- `formatters.py` — Block Kit message builders. 3000-char limit per section block, `_truncate()` helper enforces it.
- `scheduler.py` — APScheduler async jobs. Morning digest at configurable hour, proactive anomaly alerts.
- `Dockerfile` — Standalone container for the Slack bot service.

## Conventions
- **Ticker validation**: `ticker.isalpha() and len(ticker) <= 5` on all slash commands before processing
- **Async polling**: `/analyze` starts analysis via backend POST, polls GET every 5s (max 10 min) until complete/failed
- **Block Kit only**: Never use plain text messages. Use `blocks=` param with Block Kit sections, headers, fields.
- **Score emoji**: ≥8 star, ≥6 check, ≥4 yellow circle, <4 red circle
- **Internal API calls**: httpx with 30s timeout to `http://backend:8000`. No auth tokens (internal Docker network).
- **Settings**: `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_CHANNEL_ID`, `morning_digest_hour` from `backend.config.settings`
