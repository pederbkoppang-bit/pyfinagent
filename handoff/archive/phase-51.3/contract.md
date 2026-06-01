# Contract -- phase-51.3: weekend/holiday Slack digest trading-day guard

**Step id:** 51.3 | **Priority:** P2 (integrity; operator-reported issue 1) | **depends_on:** 50.4
**Date:** 2026-06-01 | **harness_required:** true | **$0 LLM** | no pip | slack-bot only (NOT the trading loop)

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher `a7f9f72fde7c51025`: gate_passed=true, tier simple, 6 sources read in full, 16 URLs, recency scan, 6 internal files). Decisive:
- **Insertion points (verbatim, `backend/slack_bot/scheduler.py`):** `_send_morning_digest` def `:317` -> guard AFTER `settings = get_settings()` (:319), BEFORE `try:` (:321). `_send_evening_digest` def `:343` -> guard AFTER `settings = get_settings()` (:345), BEFORE `try:` (:347). New shared helper `_is_us_trading_day_now()` near :316.
- **The call:** `et_today = datetime.now(ZoneInfo("America/New_York")).date()` (both already imported in scheduler.py; cron tz is America/New_York) -> `is_trading_day(et_today, "US")` (`backend/backtest/markets.py:147`, exchange_calendars XNYS `cal.is_session`, **fail-open** when xcals unavailable, tz-naive-safe). In-repo precedent: `autonomous_loop.py:341-355` (`_open_today`).
- **Resolution: silent early-return (hard skip)** -- matches the file's silent-unless-signal philosophy + the "only send if fresh" doctrine; a daily "market closed" note = ~115 noise msgs/yr. Do NOT use `day_of_week='mon-fri'` on the cron (APScheduler has NO holiday support, issue #520 -> would still fire July 4th/Christmas; the in-body guard covers weekends AND holidays in one check).
- **Half-day (early-close):** `is_session` returns True -> the digest SENDS on half-days (fresh data exists). Desired; the guard does not over-suppress.
- **Restart:** scheduler.py is bound at slack_bot startup (`app.py:56`), NOT hot-reloaded. There is NO launchd label for the slack-bot -> restart via `pkill -f "backend.slack_bot.app"` + relaunch. The UNIT TEST is the reliable live_check artifact (the live restart is an operator action).

## Hypothesis
A shared `_is_us_trading_day_now()` helper gating both `_send_morning_digest` and `_send_evening_digest` with an early-return before any HTTP fetch / `chat_postMessage` makes the digests skip weekends AND market holidays (via the existing XNYS calendar), so they only fire on US trading days (when fresh data exists). Fail-open (xcals unavailable -> is_trading_day returns True -> digest sends as before -> no regression). US-only, slack-bot-only -> the trading loop + paper-trading routes are untouched.

## Success criteria (IMMUTABLE -- verbatim from masterplan step 51.3)
1. _send_morning_digest and _send_evening_digest early-return (no chat_postMessage) when today (ET) is not a US trading day, using the existing is_trading_day helper
2. a regression test proves both digests SKIP when is_trading_day is monkeypatched False and still SEND when True
3. no change to the autonomous trading loop or any paper-trading route (digest-path only); fail-open if exchange_calendars is unavailable

**Verification command:** `pytest backend/tests/test_phase_51_3_digest_guard.py` + `ast.parse(scheduler.py)` + `test -f live_check_51.3.md`.
**live_check:** REQUIRED -- unit proof both digests skip on a non-trading day (monkeypatched is_trading_day False) and send on a trading day (True). (Live slack-bot restart is an operator action; flagged, not required.)

## Plan steps (GENERATE)
1. **scheduler.py:** add `_is_us_trading_day_now() -> bool` near :316 -- `et_today = datetime.now(ZoneInfo("America/New_York")).date(); return is_trading_day(et_today, "US")` (lazy import is_trading_day inside the helper; fail-open inherited from is_trading_day).
2. **Guard both digests:** insert at the confirmed insertion points (after `settings = get_settings()`, before `try:`): `if not _is_us_trading_day_now(): logger.info("morning_digest skipped: %s ET is not a US trading day", datetime.now(ZoneInfo("America/New_York")).date()); return` (and the evening analogue with 'evening_digest').
3. **Test:** `backend/tests/test_phase_51_3_digest_guard.py` (mirror test_phase_slack_digest_71.py) -- monkeypatch `backend.slack_bot.scheduler.is_trading_day` (or the helper) to False -> assert both digest functions early-return WITHOUT calling chat_postMessage / the HTTP fetch (use a fake app/client + assert no post); True -> assert they proceed (reach the fetch). Plus a direct `_is_us_trading_day_now` test monkeypatching is_trading_day.
4. **Verify:** pytest; ast.parse(scheduler.py); capture the unit proof (skip-when-False, send-when-True) into `live_check_51.3.md`. Confirm NO change to autonomous_loop / paper_trading routes (grep the diff).
5. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 51.3 -> done.

## Safety / scope notes
- **Slack-bot only.** Diff = scheduler.py (helper + 2 guards) + the new test. NO trading-loop / paper-trading / risk-guard change.
- **Fail-open:** is_trading_day returns True if exchange_calendars is unavailable -> digests send as today (no regression; never silently suppressed by a calendar-lib error).
- **Restart caveat:** the live skip only takes effect after the slack-bot is restarted (no launchd label -> operator pkill+relaunch). The unit test is the gate evidence; the live restart is flagged for the operator.
- US-only (the digests are US-only today); EU/KR digest gating is a future step, NOT scoped here.
- $0 LLM; no pip; no spend; no DROP/DELETE.

## References
- handoff/current/research_brief.md (51.3 gate)
- backend/slack_bot/scheduler.py:316-347 (helper + both digest insertion points), :199-220 (cron registration), app.py:56 (scheduler bind)
- backend/backtest/markets.py:147 (is_trading_day); backend/services/autonomous_loop.py:341-355 (_open_today precedent)
- backend/tests/test_phase_slack_digest_71.py (mirror); test_phase_50_4_calendar.py (real-date precedent)
- APScheduler issue #520 (no holiday support -> in-body guard, not day_of_week)
