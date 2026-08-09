# Pending backend restart -- 2026-08-09

Per CLAUDE.md, restarts are batched to SESSION END. Everything below is
**COMMITTED BUT NOT IN FORCE**: the running process imported these modules
before the change landed.

## Running process, measured

```
$ ps -p 84494 -o lstart=
søn.  9 aug. 17.08.05 2026        (= 15:08:05Z)
```

## Owed by this session

| Change | Commit | Commit time | In force? |
|---|---|---|---|
| **36.17** exit-only stop-loss pass in the halt branch (`autonomous_loop.py`, +73) | `e98ca260` | 17:31:45 | **NO** -- 23 min after the process started |
| 36.17 cycle-2 comment corrections (comment-only, no behaviour) | `6ca17793` | 17:47 | NO (irrelevant -- comments only) |

**Consequence while un-restarted, stated plainly:** a `paused` or `blocked`
cycle running under pid 84494 still returns before Step 5.6 and still does NOT
enforce stop-losses. The defect 36.17 fixes is live until the restart.

Mitigating fact: the kill switch currently reads `paused: false`, so the halt
path is not being taken right now. The exposure is conditional on a halt
occurring before the restart.

## Already done this session (do NOT repeat)

| Change | Action taken | Verified |
|---|---|---|
| `PAPER_CYCLE_MAX_SECONDS 7200 -> 10800` (`backend/.env`) | `launchctl kickstart -k` at 15:08Z | YES -- `/api/settings/` on the RUNNING backend returns `10800.0` |

## Restart procedure

`.env`-only changes need `kickstart -k` (it restarts the process, which re-reads
`.env`). Only a **plist `EnvironmentVariables`** change needs `bootout`+`bootstrap`
-- and `bootout` is blocked by the 62.0 guard (away-ops rail 9, operator-reserved).

```
# 1. NEVER restart into a live cycle -- read the LOCK, never last_result:
cat handoff/.autonomous_loop.lock        # require "state": "released"

# 2. Restart:
launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend

# 3. Prove it took effect on the RUNNING process, not a fresh interpreter:
launchctl list | grep com.pyfinagent.backend    # new pid
curl -s localhost:8000/api/health
```

## Post-restart check specific to 36.17

The fix is only exercised on a halted cycle, so a normal restart proves only
that it imports. To prove it is IN FORCE, compare the running process's start
time against `git log -1 --format=%cd e98ca260` -- start time must be LATER.
