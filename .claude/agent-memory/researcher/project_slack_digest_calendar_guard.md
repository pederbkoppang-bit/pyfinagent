---
name: slack-digest-calendar-guard
description: phase-51.3 Slack digest trading-day guard -- insertion points, is_trading_day call shape, slack-bot has NO launchd label (restart via pkill not kickstart), half-day is_session=True
metadata:
  type: project
---

phase-51.3 (operator issue 1): morning/evening Slack digests fire 7 days/week and re-send stale data on weekends/holidays. Fix = gate both digest functions on `is_trading_day` using ET-today, silent early-return.

**Insertion points** (`backend/slack_bot/scheduler.py`): `_send_morning_digest` def :317, guard after `settings = get_settings()` (:319) before `try:` (:321). `_send_evening_digest` def :343, guard after :345 before `try:` (:347). Both register as plain daily crons :199-208 / :211-220 (`hour=` only, NO `day_of_week`), cron tz `ZoneInfo("America/New_York")`.

**The guard call:** `is_trading_day(datetime.now(ZoneInfo("America/New_York")).date(), "US")` from `backend.backtest.markets` (import lazily). `datetime`+`ZoneInfo`+`logger` already imported in scheduler.py. is_trading_day is fail-open (returns True if exchange_calendars missing or on any error) and tz-naive-date-safe (markets.py:147-168, phase-50.4 rewrite to `cal.is_session(ts.normalize())`, exchange_calendars 4.13.2). In-repo precedent for the exact idiom: `autonomous_loop.py:341-355` (`_open_today` entry gate).

**Half-day behaviour (settled):** a half-day / early-close IS a session -> `is_session` returns True (empirically `is_session("2025-11-28")=True`; `early_closes` docstring = "Sessions that close earlier than the prevailing normal close"). So the digest correctly SENDS on half-days (fresh data exists). The guard does NOT over-suppress.

**Resolution:** silent early-return (hard skip, no "market closed" Slack note) -- matches the file's silent-unless-signal watchdog philosophy + external suppress-empty consensus (raising an error to abort is an anti-pattern). Do NOT use `day_of_week='mon-fri'` on the cron as the fix: APScheduler has NO built-in holiday support (issue #520 open since 2021), so a mon-fri cron still fires on July 4th/Christmas. The in-body guard covers weekends AND holidays in one check.

**Restart for live_check (NON-OBVIOUS):** the slack-bot is a SEPARATE process from the backend (port 8000). A scheduler.py edit is bound into the in-memory AsyncIOScheduler at `app.py:56 start_scheduler(app)` and is NOT hot-reloaded. **There is NO launchd label for the slack-bot** (verified 2026-06-01: `launchctl list | grep pyfinagent` has backend/frontend/mas-harness/autoresearch/ablation/backend-watchdog/claude-code-proxy but NO slack-bot; no `com.pyfinagent.slack-bot.plist`). So the frontend `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` idiom does NOT apply -- restart via `pkill -f "backend.slack_bot.app"` + relaunch `python -m backend.slack_bot.app`. Prefer the unit-test live_check artifact over restart-and-observe since it doesn't depend on the restart taking.

**Test file:** `backend/tests/test_phase_slack_digest_71.py` is the canonical digest-regression home (source-grep + formatter tests, no BQ/Slack). Behavioral-assertion precedent: `test_phase_50_4_calendar.py` (real-XNYS-date asserts, e.g. `is_trading_day("2026-06-13","US") is False`).

Multi-market: digests are US-only today; guard tests "US". EU/KR digests (future) need their own market calendar + market-local date (generalizes via `market_for_symbol`+`get_market_config(mk)["timezone"]`). Do NOT scope into 51.3.

Related: [[project_market_calendar_gating]] (phase-50.4 is_trading_day rewrite), [[project_multimarket_scaffolding_disconnected]].
