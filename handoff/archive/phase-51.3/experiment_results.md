# experiment_results -- phase-51.3: weekend/holiday Slack digest guard

**Step:** 51.3 | **Date:** 2026-06-01 | **$0 LLM** | no pip | slack-bot only | GENERATE complete

## What was changed
The morning/evening Slack digests fired 7 days/week (plain daily crons, no guard) and re-sent
the prior trading day's data on weekends/holidays (operator-reported: sent Sat 5/16, Sun 5/17,
Sat 5/23, Sun 5/24). Added an early-return guard on the existing XNYS trading-day calendar.

| File | Change |
|------|--------|
| `backend/slack_bot/scheduler.py` | **NEW `_is_us_trading_day_now()`** (`is_trading_day(datetime.now(ZoneInfo("America/New_York")).date(), "US")`, fail-open). Guard added to `_send_morning_digest` (after `settings=get_settings()`, before `try`) + `_send_evening_digest` (same) -> early-return + log when ET today is not a US trading session. +26 lines. |
| `backend/tests/test_phase_51_3_digest_guard.py` | **NEW** 5 tests (helper delegation; SKIP morning+evening with a probe proving the body never runs; PROCEED morning+evening). |

## Why an in-body guard (not `day_of_week='mon-fri'`)
APScheduler has NO holiday support (issue #520) -> a `mon-fri` cron would still fire July 4th /
Christmas. The in-body `is_trading_day` guard covers weekends AND market holidays in one check
(via exchange_calendars XNYS). Half-days send (is_session True -> fresh data exists).

## Verification command output (verbatim)

### Syntax
```
OK scheduler.py
OK test
```

### pytest (phase-51.3 -- 5 tests)
```
$ python -m pytest backend/tests/test_phase_51_3_digest_guard.py -q
.....                                                                    [100%]
5 passed in 0.33s
```

### Scope (slack-bot only)
```
$ git diff --stat backend/
 backend/slack_bot/scheduler.py | 26 ++++++++++++++++++++++++++
```
(the new test file is untracked; no trading-loop / paper-trading / risk-guard change.)

### Real XNYS-calendar smoke -> handoff/current/live_check_51.3.md
helper today (Sun ~20:14 ET) = False (CORRECT -- uses ET date, not UTC); Sat/Sun/Mon + Jul-4 holiday all correctly classified.

## Byte-identity / safety
- Slack-bot only; the trading loop + paper-trading routes are untouched (diff confirmed).
- Fail-open: `is_trading_day` returns True when exchange_calendars is unavailable -> digest sends as before (a calendar-lib error never silently suppresses a digest).
- ET-date correctness: the guard uses the cron tz (America/New_York), so it gates on the actual NYC trading date.

## Artifact shape
- `_is_us_trading_day_now() -> bool`
- both digests: early `return` (no `chat_postMessage`, no HTTP fetch) on a non-trading day.

## Operator note (live activation)
scheduler.py is loaded at slack-bot startup (app.py:56), NOT hot-reloaded, and the slack-bot has
NO launchd label. The live skip activates only after the operator restarts the slack-bot:
`pkill -f "backend.slack_bot.app"` then `python -m backend.slack_bot.app`. The unit + calendar
proof is the $0 gate evidence; the live restart is flagged for the operator.

## Next
51.4 (cron repairs: autoresearch + weekly_data_integrity) finishes the operator's 4 reported
issues. Then calendar_events / 50.6 / MEASURE Monday's first multi-market cycle.
