# research_brief -- phase-51.3: Slack digest trading-day guard

Tier: simple (depth low; fix is well-scoped). Source floor (>=5 read-in-full, 3-variant queries, recency scan, >=10 URLs) still enforced -- operator overruled all "skip researcher for small fixes" carve-outs. $0 LLM.

Scope: slack-bot scheduler ONLY. NOT the live trading loop, NOT any paper-trading route.

Status: COMPLETE.

---

## Part A -- Internal code audit (file:line)

### A1. Digest functions -- CONFIRMED, line numbers match diagnostic

`backend/slack_bot/scheduler.py`:

- **`_send_morning_digest(app)`** -- def at **:317**, docstring :318, `settings = get_settings()` :319, `try:` :321, first network I/O (`httpx.AsyncClient.get` portfolio) :322-323, `chat_postMessage` :331-335. **INSERTION POINT for the guard: a new block immediately after `settings = get_settings()` (line 319) and BEFORE the `try:` at :321** -- i.e. early-return before any HTTP fetch or post.
- **`_send_evening_digest(app)`** -- def at **:343**, docstring :344, `settings = get_settings()` :345, `try:` :347, first I/O :348-349, `chat_postMessage` :370-373. **INSERTION POINT: immediately after `settings = get_settings()` (line 345), before `try:` at :347.**

Both register as PLAIN daily crons in `start_scheduler`:
- morning_digest: `add_job(_send_morning_digest, "cron", hour=settings.morning_digest_hour, minute=0, timezone=ZoneInfo("America/New_York"), ...)` -- **:199-208**. `hour=` only, NO `day_of_week`, NO trading-day guard. CONFIRMED.
- evening_digest: `add_job(_send_evening_digest, "cron", hour=settings.evening_digest_hour, minute=0, timezone=ZoneInfo("America/New_York"), ...)` -- **:211-220**. Same. CONFIRMED.

Cron TZ is `ZoneInfo("America/New_York")` (DST-aware) for BOTH (:204, :216). So the guard's "today" MUST be ET to align with when the cron actually fires -> `datetime.now(ZoneInfo("America/New_York")).date()`.

`settings.morning_digest_hour` default = 8, `evening_digest_hour` default = 17 (`backend/config/settings.py:458-459`). Note the digest functions also format the message header with a bare `datetime.now().strftime(...)` at :334/:373 (process-local, NOT ET) -- out of scope for 51.3; do NOT change it here.

### A2. `is_trading_day` -- CONFIRMED signature + behavior

`backend/backtest/markets.py:147` `def is_trading_day(date, market: str = DEFAULT_MARKET) -> bool:` (DEFAULT_MARKET = "US", :18).
- phase-50.4 rewrite uses `cal.is_session(ts.normalize())` (:165) on exchange_calendars XNYS. The `.days`-removed latent-no-op bug is fixed.
- **Fail-open**: returns True when `xcals` is None (lib unavailable, :158-159) AND when any exception is raised in the body (:166-168). Never blocks on a calendar error. CONFIRMED.
- **tz handling**: accepts a tz-NAIVE date; internally `pd.Timestamp(date)`, strips tzinfo if present (:163-164), then `.normalize()`. Passing a python `datetime.date` (naive, from `.date()`) is correct and safe -- the strip branch is a no-op. CONFIRMED.
- Import path: `from backend.backtest.markets import is_trading_day`.

`ZoneInfo` and `datetime` are ALREADY imported in scheduler.py (`from datetime import datetime, timedelta, timezone` :8; `from zoneinfo import ZoneInfo` :9). NO new stdlib imports needed for the ET-today call; only `is_trading_day` must be imported (do it LAZILY inside each function / helper to avoid a module-load-time backtest import in the slack process and to mirror the autonomous_loop precedent).

exchange_calendars version installed: **4.13.2** (confirmed via venv).

### A3. In-repo PRECEDENT for the exact guard -- autonomous_loop.py:341-355

`backend/services/autonomous_loop.py:341-355` ALREADY does exactly this pattern for the entry calendar gate (phase-50.4):
```python
from backend.backtest.markets import is_trading_day, market_for_symbol, get_market_config
...
local_date = _dt.now(_tz.utc).astimezone(_ZI(market_tz)).date()
return is_trading_day(local_date, mk)
```
with a fail-open `except` that logs + keeps. The 51.3 guard should mirror this: lazy import, compute ET-local date, call `is_trading_day(et_today, "US")`, on False -> `logger.info(...)` + `return` BEFORE any post. Wrap the guard in its own try/except that falls THROUGH to sending on error (a calendar bug must never silently suppress a legitimate digest -- consistent with markets.py fail-open and the loop's `return True`).

For US, the simplest ET-today is `datetime.now(ZoneInfo("America/New_York")).date()` (both already imported). (autonomous_loop derives the tz from `get_market_config(mk)["timezone"]` because it is market-generic; the digest is US-only so the literal "America/New_York" is fine and matches the cron tz.)

### A4. Test file to mirror -- TWO precedents

1. **`backend/tests/test_phase_slack_digest_71.py`** -- canonical Slack-digest test home. Pure source-grep + formatter tests, NO BigQuery, NO live Slack. `test_evening_digest_scheduler_passes_since_today()` (:216-223) reads `scheduler.py` as text and asserts a substring -- a source-grep guard test. A 51.3 regression test can add a source-grep assertion that BOTH digest paths reference `is_trading_day`.
2. **`backend/tests/test_phase_50_4_calendar.py`** -- directly unit-tests `is_trading_day` with real XNYS dates (`is_trading_day("2026-06-13","US") is False` weekend; `("2026-05-01","EU") is False` holiday). Precedent for the BEHAVIORAL assertion: a 51.3 test can monkeypatch the scheduler's ET-today to a Saturday and assert the AsyncMock Slack client's `chat_postMessage` was NOT awaited.

**Recommendation:** add tests to `test_phase_slack_digest_71.py` (keeps all digest regressions in one file) -- one source-grep guard test PLUS one behavioral test that monkeypatches/forces today to a known weekend (e.g. 2026-06-13 Sat) and asserts no post. The behavioral test is the stronger one; if the guard is factored into a `_is_us_trading_day_now()` helper, the test can call that helper directly with a frozen date OR monkeypatch `markets.is_trading_day` / the helper to return False and assert the digest early-returns without awaiting the mocked Slack client.

### A5. Hard-skip vs "market closed" note -- RESOLUTION: silent early-return (hard skip)

Diagnostic recommended silent skip; matches the project's established "only send if fresh data" doctrine already applied to the SAME digests:
- phase-71 added `since_today=true` so "Today's Trades" only shows fresh rows (scheduler.py:358-359).
- phase-72 added "(as of close YYYY-MM-DD)" labeling because the operator was confused by stale-looking values.

On a weekend/holiday there is NO fresh trading data, and the watchdog philosophy across this file is "silent unless real signal" (`_watchdog_health_check` posts ONLY on transitions, :411-435). A daily "market closed" note would be NOISE (~52 weekends + ~9 holidays = ~115 low-value messages/yr). **RESOLUTION: silent early-return with `logger.info("... digest skipped: <date> ET is not a US trading day")`. No Slack post.** Lowest-noise, lowest-risk, consistent with every other gate in the file. A once-per-transition "closed this week" note would be a separate enhancement -- do NOT scope-creep 51.3. External best-practice (Part B) corroborates: post-only-when-relevant / suppress empty digests is the documented norm for scheduled notifications.

### A6. Restart requirement for live_check -- YES

`backend/slack_bot/app.py:17` imports `start_scheduler`; `:56` calls `start_scheduler(app)` once at slack-bot startup. The cron callables are bound into the in-memory AsyncIOScheduler at that moment. **A scheduler.py edit is NOT picked up by a running slack-bot process -- the standalone `python -m backend.slack_bot.app` process MUST be restarted.** live_check evidence: (a) a unit-test run showing the guard short-circuits on a weekend, and/or (b) after restart, a log line `... digest skipped: <weekend-date> ET is not a US trading day` on the next weekend fire (or a forced manual invocation with a frozen Saturday). The slack-bot is a SEPARATE process from the backend (8000); restarting the backend does NOT restart it.

**Restart mechanism (verified 2026-06-01):** the slack-bot is currently running as `python -m backend.slack_bot.app` (observed PID 42151) but has **NO dedicated launchd label** -- `launchctl list | grep pyfinagent` shows backend / frontend / mas-harness / autoresearch / ablation / backend-watchdog / claude-code-proxy labels but **no slack-bot label**, and there is no `com.pyfinagent.slack-bot.plist` in `~/Library/LaunchAgents/`. So the frontend `launchctl kickstart -k ...` idiom from CLAUDE.md does NOT apply here. The slack-bot restart is: `pkill -f "backend.slack_bot.app"` then relaunch `source .venv/bin/activate && python -m backend.slack_bot.app` (per CLAUDE.md Quick Start). If the process is a child of a supervising script, killing it may auto-respawn; verify the new PID picked up the edit. The unit-test path (a) is the more reliable live_check artifact since it does not depend on the restart actually taking; the restart-and-observe-log path (b) is the live confirmation.

### A7. Multi-market note (DO NOT scope-creep)

Digests are US-only today (pull `/api/paper-trading/portfolio` + `/api/paper-trading/trades`, header "PyFinAgent ... Digest"). The 51.3 guard tests `is_trading_day(et_today, "US")` -- correct for today. FLAG for future: EU/KR paper digests must gate on THAT market's calendar + market-local date (the `_open_today` precedent in autonomous_loop generalizes via `market_for_symbol` + `get_market_config(mk)["timezone"]`). 51.3 is US-only; do not add EU/KR logic now.

### A8. Suggested shape -- single shared guard helper

Both functions share identical guard needs. Recommend a tiny module-level helper `_is_us_trading_day_now() -> bool` (lazy-imports is_trading_day, computes ET-today via `datetime.now(ZoneInfo("America/New_York")).date()`, returns the bool, fail-open `return True` on any exception) called at the top of each digest. DRY, one test target, mirrors the `_open_today` closure from autonomous_loop. Keeps each insertion point to a single guard line:
```python
if not _is_us_trading_day_now():
    logger.info("morning digest skipped: %s ET is not a US trading day",
                datetime.now(ZoneInfo("America/New_York")).date())
    return
```
(evening symmetric). Both insertion points sit right after `settings = get_settings()` and before the `try:`.

---

## Part B -- External research

### Read in full (6; floor is 5; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://pypi.org/project/exchange_calendars/ | 2026-06-01 | official docs (library home) | WebFetch (full) | `xnys.is_session("2022-01-01")` -> `False`. `is_session` is the canonical single-date trading-session check; accepts a date-like arg, returns bool. Inquiry methods: `sessions_in_range`, `is_trading_minute`, `schedule`. v3+ deprecation tables show renamed methods (the 4.x line is the successor to trading_calendars). |
| https://github.com/gerrymanoim/exchange_calendars/blob/master/docs/tutorials/sessions.ipynb (raw) | 2026-06-01 | official docs (tutorial) | WebFetch (raw .ipynb, full) | **LOAD-BEARING.** "In `exchange_calendars` a 'session' is a timezone-naive timestamp that represents a date on which an exchange is open..." and "A timestamp representing a 'session' takes the date in which **most of the session falls** (based on UTC open/close times)." -> confirms (a) sessions are tz-NAIVE (markets.py:165 `cal.is_session(ts.normalize())` is correct), (b) the session date is UTC-derived (why market-LOCAL date matters for KR/EU; US/ET is fine for the digest). |
| https://pandas-market-calendars.readthedocs.io/en/latest/usage.html | 2026-06-01 | official docs (alt library) | WebFetch (full) | Alt idiom for "is the market open on date X": `valid_days(start,end)` (excludes weekends+holidays) + `.isin()`, or a one-day `schedule()` and check non-empty. **`early_closes(schedule=...)` EXPLICITLY distinguishes half-days** -- "identifies dates where market_close occurs earlier than the regular schedule. Early closes appear in schedule DataFrames but can be filtered separately." => half-days ARE trading days/sessions in both libs; they are flagged, not excluded. |
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html | 2026-06-01 | official docs (APScheduler 3.x) | WebFetch (full) | `day_of_week` = "number or name of weekday (0-6 or mon,tue,wed,thu,fri,sat,sun)"; example `add_job(fn,'cron',day_of_week='mon-fri',hour=5,minute=30)`. **"No built-in mechanism exists" for holidays / non-trading-day exclusion** -- cron covers only standard temporal fields. Confirms a `day_of_week='mon-fri'` cron would drop weekends but CANNOT drop holidays -> an in-body calendar guard is still required for holiday coverage. |
| https://github.com/agronholm/apscheduler/issues/520 | 2026-06-01 | official (maintainer repo, issue) | WebFetch (full) | "Scheduling Support for Statutory Holidays" -- a FEATURE REQUEST (opened Jun 2021, still OPEN); holiday-skip is NOT built in. Two patterns: (a) a custom Trigger class (using the `holidays` lib + OrTrigger of DateTriggers) -- "keeps scheduling logic separate from job execution," vs (b) an in-job date check. The issue author prefers the custom trigger ARCHITECTURALLY, but it requires holiday data + more machinery. For a 2-function minimal fix where exchange_calendars already supplies the calendar AND an in-repo in-body precedent exists, the in-body guard is the pragmatic choice. |
| https://community.fabric.microsoft.com/t5/Service/Avoid-to-send-scheduled-report-with-no-data/m-p/1701269 | 2026-06-01 | community (practitioner consensus) | WebFetch (full) | "Avoid sending scheduled report with no data" is a recognized, common requirement. Consensus pattern: a **guard-BEFORE-send** -- "I first check if the query will return any results. If yes, then I run the ... report and forward results to the recipients." Anti-patterns called out: forcing an error (div-by-zero / SQL raise) to abort delivery generates error noise instead of silent suppression. => the correct shape is an explicit pre-send guard that silently early-returns, NOT an exception. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/gerrymanoim/exchange_calendars/blob/master/exchange_calendars/exchange_calendar.py | official source | GitHub HTML view truncates >1000 lines; `is_session` body not in the rendered excerpt. BUT the `early_closes` docstring WAS visible: "Sessions that close earlier than the prevailing normal close" -> half-days are SESSIONS. Behavior confirmed empirically instead (venv probe below). |
| https://docs.datadoghq.com/dashboards/sharing/scheduled_reports/ | official docs | Read in full; Datadog does NOT expose "suppress-if-no-data" -- negative evidence (corroborates that conditional/empty-suppression is an app-level concern, not universally built in). |
| https://help.splunk.com/.../report-management/.../schedule-reports | official docs | Read in full; the SCHEDULE page does not document trigger-conditions (those live in Splunk's Alerting manual -- "trigger when number of results > 0"). Negative evidence; the freshness-gate concept exists in Splunk alerts, just not on this page. |
| https://github.com/agronholm/apscheduler/issues/606 | official (issue) | CronTrigger + DST edge case; tangential (our cron is DST-aware via ZoneInfo already), not load-bearing. |
| https://support.getpause.com/en/article/set-up-slack-notifications-for-public-holidays-qeud8b/ | vendor | Slack holiday-notification tool; confirms "skip on holidays" is a common Slack-bot need but is a 3rd-party product, not a code pattern. |
| https://docs.slack.dev/messaging/sending-and-scheduling-messages/ | official docs | `chat.scheduledMessages.list` / `chat.scheduleMessage` are Slack-side scheduling; our digests post live via `chat_postMessage`, so Slack-side scheduling is not the mechanism here. Snippet confirms the "cancel reminder on a holiday" use case is recognized by Slack. |
| https://medium.com/@wl8380/mastering-trading-periods-in-python-...-market-calendars-... | blog | Practitioner walkthrough of exchange_calendars vs pandas_market_calendars; corroborates is_session idiom; lower-tier than the official docs read in full. |
| https://questdb.com/docs/query/operators/exchange-calendars/ | vendor docs | QuestDB's exchange-calendar operator; confirms is_session semantics are an industry-standard concept; not Python-API-specific. |

### Empirical confirmation (venv probe -- the half-day question, settled)
Ran against the installed `exchange_calendars==4.13.2` / XNYS:
```
2025-11-28 is_session= True    # day after US Thanksgiving = EARLY CLOSE (half day) -> STILL a session
2025-12-25 is_session= False   # Christmas (full holiday)
2025-11-27 is_session= False   # Thanksgiving (full holiday)
2026-07-03 is_session= False   # observed Independence Day holiday
```
=> **a half-day (early-close) IS a session; `is_session` returns True.** This is the DESIRED behavior for the digest: on a half-day the market traded and fresh data exists, so the digest SHOULD send. The guard (`if not is_trading_day(...): return`) will NOT suppress half-days. CONFIRMED both empirically and via the `early_closes` docstring ("Sessions that close earlier than the prevailing normal close").

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026):** "scheduled notification digest only send when fresh data avoid empty report best practice 2026" (-> Power BI/Fabric, Datadog, Splunk, Klaviyo). "exchange_calendars 4.x is_session early close half day session returns True 2025" (-> exchange_calendars GitHub tutorials, QuestDB).
2. **Last-2-year window (2025):** the "...2025" half-day query above; surfaced the XHKG Lunar-New-Year-2025 early-close note (half-day = morning session only on multi-session exchanges) -- see Recency scan.
3. **Year-less canonical:** "exchange_calendars is_session check market open trading day Python pandas_market_calendars"; "APScheduler cron day_of_week vs in-body guard skip job conditionally scheduled task"; "APScheduler skip job holiday non-business day trading calendar conditional return early"; "Slack bot scheduled message skip weekend holiday market closed trading calendar guard" (-> the official APScheduler docs, issue #520, the exchange_calendars sessions tutorial, and the Slack-side scheduling docs -- the year-less queries surfaced the decisive "no built-in holiday support; use an in-body guard or custom trigger" finding).

### Recency scan (2024-2026) -- PERFORMED
Searched the last-2-year window on (a) exchange_calendars 4.x `is_session` / half-day behavior 2025, (b) scheduled-digest empty-report suppression best practice 2026, (c) Slack-bot weekend/holiday skip. **Findings (all COMPLEMENT, none supersede, the approach):**
1. **exchange_calendars is actively maintained on the 4.x line** (installed 4.13.2). No API break to `is_session` in the 4.x window; the tz-naive-session-label contract documented in the sessions tutorial still holds. The 2025 note that an early close on a MULTI-SESSION exchange (e.g. XHKG around Lunar New Year) may run only the morning session is irrelevant to XNYS (single continuous session) and does not change `is_session`'s True-on-half-day behavior. **No new finding overturns the markets.py:165 `cal.is_session(ts.normalize())` implementation.**
2. **2026 best-practice for scheduled digests remains "guard-before-send / suppress-empty"** (Power BI/Fabric thread, Splunk alert trigger-conditions, Amazon QuickSight "schedule snapshot report only if data is present"). The freshness-gate concept is mature and platform-agnostic; pyfinagent's silent-early-return matches it. **No 2024-2026 source argues FOR sending an empty/stale daily digest.**
3. **APScheduler holiday support is STILL absent** (issue #520 open since 2021, confirmed current). The in-body-guard vs custom-trigger debate is unchanged; the project's existing in-body precedent (autonomous_loop phase-50.4) is consistent with the pragmatic recommendation. **No new APScheduler feature obsoletes the guard approach.**

### Key findings (per-claim, cited)
1. **`is_session` is the correct, canonical single-date trading-session check, and it expects a tz-NAIVE session label.** "a 'session' is a timezone-naive timestamp that represents a date on which an exchange is open" (Source: exchange_calendars sessions tutorial, https://raw.githubusercontent.com/gerrymanoim/exchange_calendars/master/docs/tutorials/sessions.ipynb, accessed 2026-06-01); `xnys.is_session("2022-01-01") -> False` (Source: https://pypi.org/project/exchange_calendars/). pyfinagent's `markets.py:165` (`cal.is_session(ts.normalize())` after stripping tzinfo) is exactly aligned -- no change needed to the helper.
2. **A half-day / early-close IS a trading session; `is_session` returns True for it.** `early_closes` = "Sessions that close earlier than the prevailing normal close" (Source: exchange_calendars source docstring, https://github.com/gerrymanoim/exchange_calendars/blob/master/exchange_calendars/exchange_calendar.py); pandas_market_calendars `early_closes()` likewise "appear in schedule DataFrames but can be filtered separately" (Source: https://pandas-market-calendars.readthedocs.io/en/latest/usage.html); empirically `is_session("2025-11-28")=True` (venv probe). => the digest WILL (correctly) send on a half-day; the guard does not over-suppress.
3. **APScheduler has NO built-in holiday/trading-day awareness; the documented options are (a) a custom Trigger class or (b) an in-job guard.** "No built-in mechanism exists for this use case" (Source: https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html); issue #520 "Scheduling Support for Statutory Holidays" is an OPEN feature request (Source: https://github.com/agronholm/apscheduler/issues/520). A `day_of_week='mon-fri'` cron would drop WEEKENDS but NOT holidays -- so even adding `day_of_week` to the cron would still require a calendar guard for the ~9 annual US market holidays. The in-body guard (one mechanism covering both weekends AND holidays via exchange_calendars) is therefore strictly simpler than cron-day-of-week-PLUS-holiday-handling.
4. **Best practice for scheduled digests is a guard-BEFORE-send that silently suppresses when there's nothing fresh; forcing an error to abort is an anti-pattern.** "I first check if the query will return any results. If yes, then I run the ... report" (Source: Power BI/Fabric community, https://community.fabric.microsoft.com/t5/Service/Avoid-to-send-scheduled-report-with-no-data/m-p/1701269); the div-by-zero / SQL-raise workarounds "trigger error notifications rather than preventing delivery" (same source) -- i.e. raising is wrong; an explicit early-return is right. This validates A5 (silent early-return) over any exception-based skip.

### Calendar-gate vs data-freshness-gate (the diagnostic's tradeoff, resolved)
- **Calendar gate (is_trading_day on ET-today):** deterministic, $0, no extra I/O, fail-open, and it answers "is the market even open today?" BEFORE any HTTP fetch. It cannot be fooled by a stale-but-present row. This is the diagnostic's recommendation and the lower-risk choice.
- **Data-freshness gate (only send if the underlying portfolio/trade row is dated == today):** more precise about "is the data actually fresh," but requires fetching first (so the guard runs AFTER I/O), depends on the snapshot-writer's cadence (the portfolio snapshot is a close-of-day persist; on a normal trading day the snapshot may legitimately be "yesterday's close" early in the morning before today's close is written -> a naive date==today freshness gate could SUPPRESS a legitimate morning digest). The project already does a PARTIAL freshness gate (`since_today=true` on trades, phase-71; "as of close" labeling, phase-72) -- those are the right layer for per-section freshness. **For the weekend/holiday problem specifically, the CALENDAR gate is the correct, lower-risk instrument** (it targets exactly the failure mode -- market closed -> no new data -> don't send) without the morning-vs-close cadence ambiguity. RESOLUTION: calendar gate (matches diagnostic).

### Pitfalls (from literature / docs) -> applied to phase-51.3
1. **Do NOT use `day_of_week='mon-fri'` on the cron as the fix.** It would silently regress holiday coverage (cron has no holiday support; issue #520) -- the digest would still fire on July 4th, Christmas, etc. The in-body `is_trading_day` guard covers weekends AND holidays in one check. (If desired, `day_of_week='mon-fri'` could be ADDED to the cron as a cheap pre-filter, but it is NOT a substitute for the guard and adds a second source of truth -- prefer the guard alone for one authoritative gate.)
2. **Do NOT raise/abort to skip.** The empty-report consensus warns that error-based suppression generates noise. Use a plain early-return (A5).
3. **Pass a tz-NAIVE date to is_trading_day.** The session label is tz-naive (sessions tutorial). `datetime.now(ZoneInfo("America/New_York")).date()` yields a naive `date` -> correct. (markets.py also strips tzinfo defensively, so a tz-aware datetime would also work, but `.date()` is the clean idiom and matches the autonomous_loop precedent.)
4. **Use ET 'today', not UTC/process-local 'today'.** The cron fires in America/New_York; a UTC "today" could be the wrong calendar date near midnight ET (e.g. the morning digest at 08:00 ET on a US holiday is still that holiday in UTC, but an evening edge near 00:00 UTC the next day could mis-date). ET-today aligns the guard with the cron's own clock. (Also the documented session date is UTC-open/close-derived, but for XNYS the ET trading date and the session label coincide for any time during ET business hours -- the digest hours 8/17 ET are squarely inside, so ET-today is unambiguous.)
5. **Restart required (A6).** A scheduler.py edit needs a slack-bot process restart (no launchd label -> `pkill -f "backend.slack_bot.app"` + relaunch, NOT `launchctl kickstart`); the live_check must account for this.

---

## SYNTHESIS -- the actionable answer (file:line + exact call)

### S1 -- The fix (minimal, low-risk, matches diagnostic + in-repo precedent + external consensus)
Add a single shared guard helper to `backend/slack_bot/scheduler.py` and call it at the top of BOTH digest functions, silently early-returning on a non-trading ET day. This is endorsed by: the external consensus (guard-before-send / suppress-empty, NOT exception-based), the APScheduler docs (no built-in holiday support -> in-body guard is the documented option for holidays), and the in-repo `autonomous_loop.py:341-355` precedent (lazy-import `is_trading_day`, market-local date, fail-open).

### S2 -- Exact insertion points
- **`_send_morning_digest`**: after `settings = get_settings()` (scheduler.py **:319**), before `try:` (:321).
- **`_send_evening_digest`**: after `settings = get_settings()` (scheduler.py **:345**), before `try:` (:347).
- **Helper**: a new module-level `_is_us_trading_day_now()` (place it near the other module helpers, e.g. after `_route_exception_to_p1` ~:72, or just above `_send_morning_digest` ~:316).

### S3 -- The precise ET-today + is_trading_day call
```python
def _is_us_trading_day_now() -> bool:
    """phase-51.3: True if TODAY (US Eastern) is an XNYS trading session.
    Fail-open: any error -> True (never suppress a digest on a calendar bug).
    Mirrors the autonomous_loop.py:341-355 entry-gate idiom."""
    try:
        from backend.backtest.markets import is_trading_day
        et_today = datetime.now(ZoneInfo("America/New_York")).date()
        return is_trading_day(et_today, "US")
    except Exception as exc:
        logger.warning("phase-51.3 trading-day guard error; assuming open: %r", exc)
        return True
```
Guard at each insertion point:
```python
    if not _is_us_trading_day_now():
        logger.info("morning digest skipped: %s ET is not a US trading day",
                    datetime.now(ZoneInfo("America/New_York")).date())
        return
```
(`datetime`, `ZoneInfo`, `logger` already imported in scheduler.py; `is_trading_day` imported lazily inside the helper. Half-days return True -> digest sends, which is correct.)

### S4 -- Test to mirror
Add to `backend/tests/test_phase_slack_digest_71.py` (the canonical digest-regression file):
- a SOURCE-GREP test (mirror `test_evening_digest_scheduler_passes_since_today`, :216-223) asserting `scheduler.py` text contains `_is_us_trading_day_now` / `is_trading_day` referenced by the digest path; AND
- a BEHAVIORAL test (mirror `test_phase_50_4_calendar.py`'s real-XNYS-date asserts): monkeypatch the helper (or `markets.is_trading_day`) to False, call `_send_morning_digest`/`_send_evening_digest` with an AsyncMock app, assert `app.client.chat_postMessage` was NOT awaited; then monkeypatch to True and assert it WAS (or at least that the early-return did not fire). Known weekend anchor for a non-mocked variant: `is_trading_day("2026-06-13","US") is False` (Saturday).

### S5 -- Resolutions (explicit answers to the prompt's 5 Part-A questions)
1. Insertion points: scheduler.py :319 (morning) / :345 (evening), before the `try:`. CONFIRMED, lines unchanged from diagnostic.
2. `is_trading_day(date, market="US")` -- fail-open, tz-naive-date-safe, XNYS via exchange_calendars 4.13.2; cron tz America/New_York; ZoneInfo+datetime already imported. CONFIRMED.
3. Existing test file: `backend/tests/test_phase_slack_digest_71.py` (+ behavioral precedent `test_phase_50_4_calendar.py`).
4. Hard-skip vs note: **silent early-return (hard skip)** -- matches "only send if fresh" doctrine + external suppress-empty consensus + the file's silent-unless-signal watchdog philosophy. Slack-bot DOES need a restart (no launchd label; `pkill -f backend.slack_bot.app` + relaunch).
5. Multi-market: guard tests "US" only; EU/KR digests (future) need their own market calendar + market-local date. FLAGGED, not scoped into 51.3.

---

## GATE ENVELOPE

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```

`gate_passed: true` -- 6 external sources read in full via WebFetch (floor 5: exchange_calendars PyPI, exchange_calendars sessions tutorial raw .ipynb, pandas_market_calendars usage docs, APScheduler cron docs, APScheduler issue #520, Power BI/Fabric empty-report thread); recency scan performed + reported (2024-2026); 3-variant query discipline visible (current-year / last-2-year / year-less, queries listed); 16 unique URLs total; internal audit pinned to file:line across 6 files (backend/slack_bot/scheduler.py, backend/backtest/markets.py, backend/services/autonomous_loop.py, backend/config/settings.py, backend/slack_bot/app.py, backend/tests/test_phase_slack_digest_71.py; plus the venv exchange_calendars==4.13.2 probe). The half-day question is settled both empirically (is_session("2025-11-28")=True) and via the early_closes docstring.
