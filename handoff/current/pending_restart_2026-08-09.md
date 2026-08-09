# Pending backend restart — end of session 2026-08-09

Operator instruction 2026-08-09: **backend restarts are batched to the very end
of a goal session, before the next one starts.** Recorded in CLAUDE.md Critical
Rules and in auto-memory `feedback_restart_backend_at_session_end`.

## Owed at session end

| Change | Where | In force NOW? |
|---|---|---|
| `PAPER_CYCLE_MAX_SECONDS` 7200.0 → **10800.0** | `backend/.env:70` (backed up `backend/.env.bak.*`) | **NO** |

**Measured, not assumed:**

```
a fresh python process resolves : 10800.0
running backend pid 24708       : started 2026-08-09 15:20:45  -> holds 7200.0
autonomous_loop.py:506          : reads paper_cycle_max_seconds at CYCLE START
```

So the in-flight cycle (`started 13:25:27Z`) runs on the **old** 7200s budget,
and the raise lands on the first cycle after the restart. Validation is
masterplan step **86.9**, whose criteria require the value be read from the
RUNNING process — not from a fresh interpreter, which is the easy lie here.

## Already in force (no restart owed)

- `CLAUDE_CODE_OAUTH_TOKEN` **removed** from all four plists — the running
  backend (pid 24708) has no such variable, verified; the rail is alive and has
  run 60+ calls with 0 failures since.
- `.mcp.json` `--storage-state` for Playwright — takes effect at the next
  **Claude Code** session start, not a backend restart.

## Do this at session end

1. Wait for the cycle to finish (**do not restart into a running cycle** —
   check `handoff/.autonomous_loop.lock`, never `last_result`).
2. Restart, and **mind the race that cost ~4 minutes of downtime today**:
   `bootout`, then `sleep 8`, then `bootstrap`. `kickstart -k` does NOT re-read
   a plist, though it is sufficient for a `.env` change.
3. Verify: new pid, start time AFTER the edit, `/api/health` 200,
   `launchctl list` shows the pid, and the setting read from the RUNNING
   process reports **10800.0**.
