---
name: slack-digest-calendar-guard
description: phase-51.3 Slack digest trading-day guard -- is_trading_day call shape, half-day is_session=True, digests fire 14:00/23:00 CEST; slack-bot IS launchd-managed since 2026-06-12 (restart via kickstart -k; pkill is hook-BLOCKED)
metadata:
  type: project
---

phase-51.3 (operator issue 1): morning/evening Slack digests fire 7 days/week and re-send stale data on weekends/holidays. Fix = gate both digest functions on `is_trading_day` using ET-today, silent early-return.

**Insertion points** (`backend/slack_bot/scheduler.py`) -- ANCHORS DRIFT, re-derive before citing. As measured 2026-08-07: `_send_morning_digest` def **:539** (guard `_is_us_trading_day_now()` at :545-548), `_send_evening_digest` def **:585** (guard :590-593), the shared helper `_is_us_trading_day_now` at **:345-355**; cron registrations **:227-236 / :239-248** (`hour=` only, NO `day_of_week`), tz `ZoneInfo("America/New_York")`. (The pre-51.3 numbers :317/:343/:199-208 in an earlier revision of this memory are ~220 lines stale.)

**The guard call:** `is_trading_day(datetime.now(ZoneInfo("America/New_York")).date(), "US")` from `backend.backtest.markets` (import lazily). `datetime`+`ZoneInfo`+`logger` already imported in scheduler.py. is_trading_day is fail-open (returns True if exchange_calendars missing or on any error) and tz-naive-date-safe (markets.py:147-168, phase-50.4 rewrite to `cal.is_session(ts.normalize())`, exchange_calendars 4.13.2). In-repo precedent for the exact idiom: `autonomous_loop.py:341-355` (`_open_today` entry gate).

**Half-day behaviour (settled):** a half-day / early-close IS a session -> `is_session` returns True (empirically `is_session("2025-11-28")=True`; `early_closes` docstring = "Sessions that close earlier than the prevailing normal close"). So the digest correctly SENDS on half-days (fresh data exists). The guard does NOT over-suppress.

**Resolution:** silent early-return (hard skip, no "market closed" Slack note) -- matches the file's silent-unless-signal watchdog philosophy + external suppress-empty consensus (raising an error to abort is an anti-pattern). Do NOT use `day_of_week='mon-fri'` on the cron as the fix: APScheduler has NO built-in holiday support (issue #520 open since 2021), so a mon-fri cron still fires on July 4th/Christmas. The in-body guard covers weekends AND holidays in one check.

**Restart for live_check -- CORRECTED 2026-08-07 (the previous text was FALSE and would have been hook-blocked).** The slack-bot is a SEPARATE process from the backend (port 8000). A scheduler.py edit is bound into the in-memory AsyncIOScheduler at `app.py:56 start_scheduler(app)` and is NOT hot-reloaded, so a restart is genuinely required.

The old claim "there is NO launchd label for the slack-bot ... restart via `pkill -f backend.slack_bot.app`" was true on 2026-06-01 and is **wrong now**: `~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist` was installed 2026-06-12 (KeepAlive=true, RunAtLoad=true, ThrottleInterval=5). **The correct verb is `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot`** -- `-k` is required because the job is normally already running (ss64/launchctl: "-k  If the service is already running, kill the running instance before restarting"). `scripts/slack_bot_monitor.sh:28` uses BARE `kickstart` only because it fires solely when the bot is already down.

**`pkill` is now blocked by policy**, not merely inadvisable: `.claude/hooks/pre-tool-use-danger.sh:107-110` blocks any pkill/killall matching `python|uvicorn|next|slack_bot`. `launchctl bootout|unload|remove|disable` on a `com.pyfinagent.*` label is blocked at `:176-177`. Neither is needed -- `kickstart -k` re-execs the interpreter, so code changes land without touching the plist.

**Nothing replays on restart** -- verified against installed apscheduler 3.11.2 (`base.py:1066-1068`): a fresh job's `next_run_time = trigger.get_next_fire_time(None, now)`, i.e. strictly after NOW, and the jobstore is the default in-memory one (`scheduler.py:224`, no jobstore arg). So `misfire_grace_time` does NOT resurrect a cron tick missed while the process was down. The ONE deliberate re-fire is `daily_price_refresh_catchup` (`scheduler.py:330-337`), a one-shot at now+20s, idempotent by `(ticker, date)`.

**Digest timing for evidence capture:** crons are ET-pinned (`scheduler.py:232,:244`) at `morning_digest_hour=8` / `evening_digest_hour=17` (`backend/config/settings.py:626-627`) = **14:00 and 23:00 CEST**. The log lines `Morning digest sent` / `Evening digest sent` (`scheduler.py:578,:632`) in `handoff/logs/slack_bot.log` are the non-spam evidence; pair them with the restart banner (`app.py:72`, `scheduler.py:282`) to bind the digest to the NEW process. **`backend/slack_bot/digest_test.py` is NOT a dry run** -- `_run()` calls `chat_postMessage` (:34) and posts a real message from a standalone WebClient, so it both spams and proves nothing about the bot process.

**Test file:** `backend/tests/test_phase_slack_digest_71.py` is the canonical digest-regression home (source-grep + formatter tests, no BQ/Slack). Behavioral-assertion precedent: `test_phase_50_4_calendar.py` (real-XNYS-date asserts, e.g. `is_trading_day("2026-06-13","US") is False`).

Multi-market: digests are US-only today; guard tests "US". EU/KR digests (future) need their own market calendar + market-local date (generalizes via `market_for_symbol`+`get_market_config(mk)["timezone"]`). Do NOT scope into 51.3.

Related: [[project_market_calendar_gating]] (phase-50.4 is_trading_day rewrite), [[project_multimarket_scaffolding_disconnected]].
