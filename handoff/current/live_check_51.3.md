# live_check -- phase-51.3: weekend/holiday Slack digest guard

**Step:** 51.3 | **Date:** 2026-06-01 | **Result shape:** unit proof both digests SKIP on a
non-trading day (no chat_postMessage / no HTTP fetch) and PROCEED on a trading day, plus a
real-XNYS-calendar smoke of the helper. Slack-bot only; the live skip takes effect after the
operator restarts the slack-bot (no launchd label -> `pkill -f "backend.slack_bot.app"` + relaunch).

## Unit test (the gate evidence)
```
$ python -m pytest backend/tests/test_phase_51_3_digest_guard.py -q
.....                                                                    [100%]
5 passed in 0.33s
```
- `test_is_us_trading_day_now_delegates_to_is_trading_day` -- the helper returns whatever `is_trading_day` returns.
- `test_digest_skips_on_non_trading_day[_send_morning_digest]` + `[_send_evening_digest]` -- with the helper False, a probe proves the httpx body NEVER runs (`reached=False`) AND no chat_postMessage (`posted==[]`). Criterion #1.
- `test_digest_proceeds_on_trading_day[_send_morning_digest]` + `[_send_evening_digest]` -- with the helper True, execution reaches the httpx body (`reached=True`). Criterion #2.

## Real XNYS-calendar smoke (the helper against the live calendar)
```
helper today (Mon 2026-06-01): False
is_trading_day 2026-05-30 (Sat): False
is_trading_day 2026-05-31 (Sun): False
is_trading_day 2026-06-01 (Mon): True
is_trading_day 2025-07-04 (Independence Day holiday): False
```
**The `helper today = False` is CORRECT and validates the ET-date logic:** the command ran at
2026-06-01 ~00:14 UTC, which is 2026-05-31 ~20:14 ET (EDT, UTC-4) -- i.e. SUNDAY evening in
New York. The helper computes `datetime.now(ZoneInfo("America/New_York")).date()` -> 2026-05-31
(Sunday) -> not a trading day -> False. (Using UTC would have wrongly said Monday.) At the
real cron fire time (8:00 ET Monday) the ET date is 2026-06-01 -> True -> the digest sends.
Weekends AND the Jul-4 holiday are correctly flagged non-trading.

## Criterion-by-criterion

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | both digests early-return (no chat_postMessage) when ET today is not a US trading day, via is_trading_day | the two SKIP tests (probe `reached=False`, `posted==[]`) + the guard at scheduler.py (`_is_us_trading_day_now` before the `try`) | PASS |
| 2 | a regression test proves SKIP when is_trading_day False and SEND when True | the SKIP + PROCEED parametrized tests (morning + evening) | PASS |
| 3 | no change to the trading loop / paper-trading routes; fail-open if exchange_calendars unavailable | diff = `backend/slack_bot/scheduler.py` (+26) + the new test ONLY (git diff --stat); `is_trading_day` returns True when xcals missing -> digest sends as before | PASS |

## Scope / notes
- **Slack-bot only.** No trading-loop / paper-trading / risk-guard change (diff confirmed).
- **Fail-open:** a calendar-lib error -> is_trading_day True -> digest sends (never silently suppressed by a bug).
- **Half-days send** (is_session True for early-close days -> fresh data exists).
- **Restart caveat:** scheduler.py is bound at slack-bot startup (app.py:56), not hot-reloaded, and the slack-bot has NO launchd label -> the live skip activates only after `pkill -f "backend.slack_bot.app"` + relaunch (operator action). The unit + calendar proof above is the gate evidence.
- US-only; EU/KR digest gating is a future step (not scoped).
