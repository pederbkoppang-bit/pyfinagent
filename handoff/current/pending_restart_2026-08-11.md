# ~~Pending~~ COMPLETED restart -- 2026-08-11 (session `pyfinagent-06`)

> **DONE 2026-08-11 22:26:48 CEST, after the book cycle completed at 21:21.**
>
> ```
> launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend
> listener pid 66306 -> 99231      (PID CHANGED, so the restart took)
> ps lstart: tir. 11 aug. 22.26.48 2026
> launchctl list: 99231  com.pyfinagent.backend
> health: {"status":"ok", version 6.93.216, mcp data/backtest/signals all ok}
> paper_cycle_max_seconds = 10800.0   paper_analyze_top_n = 5   (both unchanged)
> ```
>
> **D2 VERIFIED IN FORCE, measured on the running process:**
> `GET /api/backtest/harness/log` returns **1226** cycles; `grep -c "^## Cycle "` on
> disk returns **1226**. **Match -- the parser is lossless in the running process.**
>
> **One correction from doing it**: the pre-restart reading was **1066**, not the
> 1064 this file predicted. My own two log appends during the evening added headers,
> so the pre-fix number was a moving target. The right invariant is
> **endpoint == on-disk**, not a frozen integer -- and that is what was checked.

**NOT YET IN FORCE.** Recorded per the standing rule that backend restarts batch to
session end and never run near the 20:00 CEST book cycle.

## Running process

```
$ ps -o pid,lstart -p 66306
  66306   man. 10 aug. 21.33.01 2026
$ lsof -nP -iTCP:8000 -sTCP:LISTEN  ->  listener pid 66306
```

**pid 66306 started 2026-08-10 21:33:01** and has not re-read any module since.

## What is committed but not active

| commit | file | change | effect once restarted |
|---|---|---|---|
| `fe9a6dad` (2026-08-11T17:13:05+02:00) | `backend/api/backtest.py` | phase-86.44 D2: harness-log parser accepts any cycle token, not just `\d+` | `GET /api/backtest/harness/log` goes from **1064** to **1224** cycles |

**Measured, not assumed** -- this is the live state as of 2026-08-11 ~17:50 CEST:

```
$ curl -s http://127.0.0.1:8000/api/backtest/harness/log | jq '.cycles | length'
1064          <- the PRE-FIX number
```

The fixed code returns 1224 when imported fresh. **So the Harness tab is currently
mis-attributing 160 cycle bodies to the preceding cycle**, and will keep doing so
until the process is restarted.

## Why it is not being restarted now

> **THE SCHEDULER LIVES INSIDE THE PROCESS BEING RESTARTED.** I had assumed the book
> cycle was a crontab entry. It is not -- `crontab -l` has exactly **one** line (the
> Slack mention checker). The cycle is an **APScheduler cron job registered inside
> the backend process**:
>
> ```
> backend/api/paper_trading.py:1436  _scheduler.add_job(
>     _scheduled_run, "cron",
>     hour=settings.paper_trading_hour,   # live value: 14
>     minute=0, day_of_week="mon-fri",
>     timezone=ZoneInfo("America/New_York"),
>     replace_existing=True)
> ```
>
> **14:00 ET = 20:00 CEST**, and today is a Tuesday, so it fires. **Restarting the
> backend tears down and re-registers that job.** That makes "no restart near the
> cycle" a hard requirement rather than a courtesy: a restart at the wrong minute
> does not merely interrupt the process, it can drop the firing. APScheduler's
> misfire grace was widened in phase-44.2.X for exactly this class of problem.

1. The book cron fires **20:00 CEST**; from 19:30 no restarts.
2. The standing rule batches restarts to session end regardless.
3. The defect is a **display misattribution on a read-only tab**. Nothing trades on
   it, and it has been present for the entire life of the parser -- one more evening
   is not a new risk.

## Restart verb

`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` is sufficient here --
this is a **code** change, not a plist `EnvironmentVariables` change, so the
`bootout`+`bootstrap` path (reserved for the operator) is not required.

**After restarting, verify by reading the running process, not the file:**

```
curl -s http://127.0.0.1:8000/api/backtest/harness/log | jq '.cycles | length'   # expect 1224
ps -o pid,lstart -p <new pid>                                                    # expect a pid AFTER fe9a6dad
```

A pid that has not changed means the restart did not take -- `launchctl list` and a
check for `EADDRINUSE` in the log are the next step, because an orphaned server can
answer on the port while launchd shows no pid.
