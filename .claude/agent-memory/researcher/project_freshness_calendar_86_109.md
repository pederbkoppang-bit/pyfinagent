---
name: freshness-calendar-86-109
description: Step 86.109 research -- the 51.3 digest guard DOES exist (grep undercounted 7 files as 2); LIVE backend.log is at repo root; 8/8 vendors have NO business-day config; the severity whitelist is INAPPLICABLE because the false positive is already P1
metadata:
  type: project
---

Step 86.109 (calendar-aware freshness alarming). Five findings that cost real
time to establish and will not be re-derivable from the code alone.

**1. A criterion premise that looked FALSE was TRUE -- the grep undercounted.**
The spawn prompt reported "exactly ONE production consumer of
`is_trading_day`". Measured: **7 files**, and the missing one was the whole
criterion -- `backend/slack_bot/scheduler.py:365-375` `_is_us_trading_day_now()`
gates both digests at `:565` / `:610`. It was invisible to the grep because the
call sits behind a **wrapper function name** and the import is
**function-local** (`from backend.backtest.markets import is_trading_day`
inside the body).

**Why:** greping a symbol finds the symbol, not its consumers. A wrapper +
function-local import hides a call site from both a module-import scan and a
literal-callsite scan.

**How to apply:** when asked "who calls X", grep X to find the DEFINITION and
any wrappers, then grep each wrapper name too. And quote your `--include`
globs -- unquoted `--include=*.py` in zsh raises `no matches found` and prints
NOTHING (I hit this on my first attempt; it is the most likely cause of the
original undercount). Related: [[test-isolation-leak-86-110]].

**2. The live backend log is NOT under `handoff/logs/`.** Per
`~/Library/LaunchAgents/com.pyfinagent.backend.plist`, `StandardOutPath` and
`StandardErrorPath` both point at repo-root `backend.log` (~30MB). Only the
ROTATED logs (`backend.log.*.gz`) live in `handoff/logs/`. A grep scoped to
`handoff/logs/*.log` returns **ZERO** for backend content and looks like a
clean negative. Always `gzcat` the rotated set AND read repo-root
`backend.log`.

**How to apply:** the rotated logs also change FORMAT mid-history -- older
lines are ANSI-coloured plain text with `HH:MM:SS` and **no date**, newer ones
are JSON with a full timestamp. A timestamp-regex census silently drops the
old half (here: 430 dated of 1,149 total). Report the split, don't average
over it.

**3. `cycle_health.py` already carries THREE calendar notions and two
disagree.** `is_weekday_et` (`:262`, raw `weekday() < 5`, **holiday-blind**),
the 96h `_CYCLE_COMPLETED_STALE_SEC` threshold (`:80`, absorbs Fri->Mon
instead of testing the day), and the calendar-blind `_TABLE_MAX_AGE_SEC`
ceilings. `markets.is_trading_day` is imported **nowhere** in that file. Adding
a day term to `_band` makes a fourth.

**4. The weekend problem is a MONDAY problem.** Measured over 430 dated
alerts: Sat+Sun = 30.0%, but **Monday alone = 38.4% (165)**, 4.3x the Tue-Fri
mean. A Sat/Sun suppression does not touch the Monday spike, because a Friday
write is still >2.0x a 24h interval on Monday morning.

**5. Industry census: 8 of 8 data-observability tools ship NO business-day or
holiday configuration** (dbt, Datadog, Monte Carlo, Anomalo, Great
Expectations, Tecton, Databricks, Sifflet -- all read in full). Two searches
specifically for a calendar-aware freshness monitor found none. The
cross-vendor answer instead is **demote/mute the NOTIFICATION, never the
DETECTION** (Grafana mute timings "suppress notifications but do not interrupt
alert evaluation"; PagerDuty support hours change urgency, not incident
creation; Alertmanager muted routes "otherwise act normally").

**6. The standard "never suppress critical" carve-out is INAPPLICABLE here --
the false positive is already P1.** Every source that endorses calendar
suppression pairs it with a severity whitelist. It cannot work on this alarm:
`freshness_cron.py:162-166` emits with a literal `severity="P1"`, and
`alerting.py:54` has `_CRITICAL_SEVERITIES = frozenset({"P0","P1","critical",
"CRITICAL"})` -- so all ~991 delivered pages took the critical branch
(`:83-93`), bypassing the consecutive-occurrence threshold with only the 1h
repeat window applying. **The weekend noise and a genuinely dead writer are
emitted at the same severity by the same call site**, so no severity rule can
separate them. Either discriminate on the CONDITION (expected-next-write
miss), or re-grade routine staleness below P1 first.

**7. The strongest CITED objection to a weekend mute is not "it hides a dead
writer" -- it is "it never expires."** The hides-a-failure claim stays weakly
sourced (community tier only; the sharp form is uncited anywhere I looked
across 23 rounds). But every source endorsing suppression pairs it with a
**finite duration + expiry obligation**, and a recurring calendar mute
re-arms itself forever with no human re-approval. Lead with that.

**8. `is_trading_day` has TWO production callers, not one** --
`slack_bot/scheduler.py:365-375` AND `services/autonomous_loop.py:663,674`.
**Both use a function-local import**, which is the house style for this
symbol and the reason any module-level import scan finds neither. Of the 7
files mentioning it: 1 definition, 2 production callers, 2 comment-only (one
is `news/fetcher.py:162` explicitly REFUSING it because it fails open), 2
tests.
