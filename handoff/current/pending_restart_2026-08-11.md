# Pending restart -- 2026-08-11 (session `pyfinagent-06`)

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
