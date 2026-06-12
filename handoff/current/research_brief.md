# Research Brief — phase-62.1 (goal-away-ops): Slack bot under launchd + restart on current code

Tier: simple-to-moderate (caller-stated; treated as moderate). Date: 2026-06-12. Agent: researcher (Layer-3, merged Explore).
Tool calls: ~19 vs 18 moderate budget (5-topic internal audit + 5 fetches); disclosed. All gate floors met.

## Sources read in full (>=5 required; counts toward gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://launchd.info/ | 2026-06-12 | Official-grade reference (canonical launchd tutorial) | WebFetch, full page | KeepAlive=true: "run the job as soon as the job definition is loaded and restart it should it ever go down." Dict forms: `SuccessfulExit` true = restart until it fails / false = restart until it succeeds; `Crashed` true = restart after crash. RunAtLoad: "for agents execution at login." ThrottleInterval = seconds to wait between invocations. EnvironmentVariables: no shell globbing/expansion. Agents run as the logged-in user from `~/Library/LaunchAgents`. |
| 2 | https://ss64.com/mac/launchctl.html | 2026-06-12 | Official man-page mirror | WebFetch, full page | `bootstrap` is "equivalent to Load in the legacy syntax" with modern domain targets (`gui/UID`); `bootout` = modern unload; `kickstart -k` = "kill the running instance before restarting"; `launchctl list` column 2 = last exit status, "negative... represents the negative of the signal" (our backend shows `-15` = SIGTERM — hard-kill lifecycle is already the norm here). `disable` persists across boots. |
| 3 | https://docs.slack.dev/apis/events-api/using-socket-mode/ | 2026-06-12 | Official docs (Slack) | WebFetch, full page | Up to 10 simultaneous connections per app; with multiple connections "each payload may be sent to *any* of the connections. It's best not to assume any particular pattern" — i.e. **events are load-balanced to ONE connection, NOT replicated to all**. Caller's "(each gets events!)" premise is wrong; see Risks for the corrected duplicate-send mechanics. Unacked envelopes (ack by `envelope_id`) are redelivered; links recycle every few hours. |
| 4 | https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html | 2026-06-12 | Official docs (APScheduler 3.x) | WebFetch, full page | `shutdown(wait=True)` "Shuts down the scheduler, along with its executors and job stores. Does not interrupt any currently running jobs"; `wait=True` blocks until executing jobs finish. Nothing persists for in-memory jobstores — a process kill without shutdown() loses only the in-RAM schedule (rebuilt at next start; this codebase already relies on that, scheduler.py:316-342 catch-up). |
| 5 | https://ss64.com/mac/caffeinate.html | 2026-06-12 | Official man-page mirror | WebFetch, full page | `-i` = prevent **idle** sleep; `-s` = "prevent the **system** from sleeping. This assertion is valid only when system is running on AC power." With a utility argument the assertion holds "as long as that process is running." Power assertions are system-wide → the backend's always-on `caffeinate -i -s uvicorn...` already keeps the whole Mac awake; **the bot plist does NOT need caffeinate**. |

## Snippet-only sources (do NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/openclaw/openclaw/issues/20257 | Bug tracker (2026) | "KeepAlive: true causes restart loop when App and launchd both manage gateway lifecycle" — exactly our cron-monitor-vs-launchd dual-supervisor class; captured in Risks |
| https://github.com/openclaw/openclaw/issues/60885 | Bug tracker (2026) | ThrottleInterval=1 caused unrecoverable downtime after auto-update — argues for >=5s throttle |
| https://github.com/openclaw/openclaw/issues/4632 | Bug tracker | EPIPE crash + exponential throttle on LaunchAgent restart |
| https://www.manpagez.com/man/5/launchd.plist/ | Man mirror | Redundant with source 1 |
| https://www.real-world-systems.com/docs/launchdPlist.1.html | Reference | Redundant with source 1 |
| https://developer.apple.com/forums/thread/88223 | Apple dev forums | SuccessfulExit usage; redundant with source 1 |
| https://github.com/tjluoma/launchd-keepalive | Example repo | KeepAlive demo plists; redundant |
| https://api.slack.com/apis/socket-mode | Official docs (legacy host) | Same content as source 3 (docs migrated to docs.slack.dev) |
| https://docs.slack.dev/apis/events-api/comparing-http-socket-mode/ | Official docs | Adjacent topic; not needed in full |
| https://github.com/agronholm/apscheduler/issues/567 | Bug tracker | `shutdown(wait=False)` may leave threads — reason NOT to add a fancy SIGTERM handler for a stateless bot |
| https://apscheduler.readthedocs.io/en/master/userguide.html | Official docs | Method contract already covered by source 4 |
| https://www.npmjs.com/package/@slack/socket-mode | Official SDK page | JS SDK angle; redundant with source 3 |

## Search queries run (three-variant discipline)

1. Year-less canonical: `launchd KeepAlive SuccessfulExit Crashed ThrottleInterval plist semantics`
2. Current-year: `Slack Socket Mode multiple connections event delivery which connection receives 2026`
3. Last-2-year: `APScheduler shutdown wait running jobs SIGTERM 2025`

## Recency scan (last 2 years, 2024-2026)

Performed via queries 2-3 above. Findings: (a) Slack developer docs migrated to docs.slack.dev in 2025 — source 3 is the current canonical and confirms load-balanced (not broadcast) delivery; (b) APScheduler 3.11.2 is the current 3.x line; issue #567 (wait=False thread leak) still open — supports "keep shutdown simple" for a stateless bot; (c) 2026 openclaw issues (#20257, #60885) document the dual-supervisor restart-loop and too-low-ThrottleInterval failure modes — directly applicable new practitioner evidence. launchd plist semantics themselves are unchanged in the window.

## Internal code audit (file:line anchors)

**1. How the bot runs today (live, 2026-06-12 08:05 UTC):** PID 26147, PPID 1, state `SN` (detached AND niced — cron-spawned), started **2026-06-11 05:03:44 CEST**, cmd `python -m backend.slack_bot.app`. Started by the cron monitor `scripts/slack_bot_monitor.sh:18-24` (`cd` repo → `source .venv/bin/activate` → `nohup python -m backend.slack_bot.app >> backend_slack.log 2>&1 &`), installed in the user crontab at `*/5`. **Correction to the step premise:** the process is running code-as-of 2026-06-11 05:03, not 2026-06-05 — but that still predates phase-60.1 (commit fa62b5fe, 2026-06-11 13:06) and phase-60.4 (b0fe1983, 16:30), so it is missing the ingestion-silence alarm (scheduler.py:600-633), busy-vs-down line (:120-141) and meta-scorer digest line (:380-398). Restart justified.
- Env loading: `backend/config/settings.py:12` `_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"` + `:536` `model_config={"env_file": str(_ENV_FILE)...}` — **absolute path, CWD-independent; the bot loads backend/.env itself. No secrets needed in the plist.**
- `app.py:45-73`: token guard, `start_scheduler(app)` (:56), background tasks queue-processor/SLA/reaper (:60-68), `AsyncSocketModeHandler.start_async()` (:71-73), `asyncio.run(main())` (:77).

**2. Plist template** (`~/Library/LaunchAgents/com.pyfinagent.backend.plist`, read verbatim): EnvironmentVariables `PATH` (venv first, then `/opt/homebrew/bin:...`), `PYTHONUNBUFFERED=1`, `DEV_LOCALHOST_BYPASS=1` (backend-auth flag — bot does not need it); `KeepAlive=true`; `RunAtLoad=true`; `ThrottleInterval=5`; `ProcessType=Interactive` + `LegacyTimers=true` (timer-coalescing matters for APScheduler cron precision — today's bot runs *niced*, which is worse); `ProgramArguments` wraps uvicorn in `caffeinate -i -s`; `WorkingDirectory=<repo>`; both log paths → `backend.log`. **Caffeinate decision: NOT needed in the bot plist** — the backend job is KeepAlive-permanent and its `-s` assertion is system-wide (source 5). Bot plist: `ProgramArguments = [<repo>/.venv/bin/python, -m, backend.slack_bot.app]` with `WorkingDirectory=<repo>` (REQUIRED: `-m` resolves the `backend` package from CWD). Keep the backend's PATH verbatim — `imsg` (scheduler.py:845) needs `/opt/homebrew/bin`; bare `python` subprocess at scheduler.py:898 needs venv-first PATH.

**3. Duplicate-process risk (corrected mechanics):** Slack does NOT broadcast to all connections (source 3). The real double-send engine is that **each process runs its own AsyncIOScheduler** (scheduler.py:224) plus its own ticket-queue processor / SLA monitor / stuck-task reaper (app.py:60-68) → two processes = duplicate digests/alerts/jobs deterministically, and inbound slash-commands route to a RANDOM one of the two (stale-vs-new code nondeterminism). **The monitor is the hidden second supervisor:** its liveness check `ps aux | grep -E "python.*backend.slack_bot.app"` (monitor.sh:13-15) would match the launchd instance in steady state, but the check-then-start is non-atomic — if cron fires in the kill→bootstrap gap, OR during any future crash window racing launchd's 5s restart, it nohup-starts a second bot (openclaw #20257 class). **The crontab line must be removed BEFORE the cutover.** (Keep the `*/2` `slack_mention_checker.sh` line — API-based, not a process supervisor.)
- Job idempotency at kill time: phase-9 jobs are idempotency-keyed daily/weekly/hourly via `backend/slack_bot/job_runtime.py` (scheduler.py:928-930); `daily_price_refresh` dedups on (ticker,date) at BQ + has a +20s catch-up-on-start (scheduler.py:316-342) — expect one benign catch-up fire right after bootstrap. All nightlies run 01:00-06:00 UTC (scheduler.py:1015-1030) — all past for today.

**4. Graceful shutdown:** NONE. `app.py` has no signal handlers, no try/finally; `pause_signals()` (scheduler.py:679-715, calls `_scheduler.shutdown(wait=False)`) is never wired to SIGTERM. SIGTERM kills the process instantly. **Risk: LOW** — in-memory jobstore (nothing to corrupt, source 4), idempotent jobs, KeepAlive makes hard-kill the operative lifecycle anyway (backend's last-exit is already `-15`). Do NOT add a SIGTERM handler in this step (scope creep + issue #567 hang risk); just cut over outside the nightly window.

**5. Logs:** today → `backend_slack.log` (repo root, gitignored via `*.log` .gitignore:24) via nohup redirect. Recommended `StandardOutPath`/`StandardErrorPath`: `<repo>/handoff/logs/slack_bot.log` — `handoff/logs/` is gitignored (.gitignore:72, phase-4.16.2 runtime-log convention) and a FRESH file makes "digest came from the NEW process" attributable (the old process can never have written there).

## Risks & gotchas

- **Cutover sequence (exact, in order):**
  1. `crontab -l | grep -v 'slack_bot_monitor.sh' | crontab -` then `crontab -l` to verify (monitor removed FIRST — kills the race permanently).
  2. Write `~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist` (shape per §2; KeepAlive=true, RunAtLoad=true, ThrottleInterval=5, no caffeinate).
  3. Import smoke-test before bootstrap (restart-loop guard): `cd <repo> && .venv/bin/python -c "from backend.slack_bot.app import create_app"` — a bad import + KeepAlive = 5s crash-loop (openclaw #60885 class).
  4. `kill 26147` (SIGTERM), then CONFIRM dead: `pgrep -f backend.slack_bot.app || echo DEAD` (must print DEAD).
  5. `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist` (modern syntax, source 2; RunAtLoad starts it immediately).
  6. Verify exactly ONE bot: `launchctl list | grep slack-bot` (PID column populated) + `pgrep -fl backend.slack_bot.app | wc -l` == 1; tail `handoff/logs/slack_bot.log` for "Slack bot starting in Socket Mode...", "Scheduler started: morning digest at 8:00, evening digest at 17:00, watchdog every 15 min", "phase-9 jobs registered", and the expected idempotent "daily_price_refresh catch-up" line.
- **Digest collision windows (ET crons, scheduler.py:227-248; settings.py:530-531 defaults 8/17):** morning 08:00 ET = 14:00 Oslo, evening 17:00 ET = 23:00 Oslo. Other ticks: hourly_signal_warmup at :05 UTC every hour; nightlies 01:00-06:00 UTC; redteam 03:15 ET. NOTE: 8/17 are code defaults — Main must confirm backend/.env doesn't override (`grep -i digest backend/.env`; researcher sandbox is denied that file).
- **Now = 08:05 UTC Fri:** all nightlies passed; next tick is hourly warmup 09:05 UTC. Window **08:10-08:55 UTC (10:10-10:55 Oslo) is clean** — avoid the :05 UTC minute.
- **KeepAlive + crashing bot = restart loop:** ThrottleInterval=5 means a boot-crash loops every ~5s, each loop re-opening a Socket Mode connection (and at 12:00 UTC each successful partial boot could re-fire startup tasks). The import smoke-test (step 3) is the cheap guard; if a loop happens anyway: `launchctl bootout gui/$(id -u)/com.pyfinagent.slack-bot` stops it (KeepAlive does not resurrect a booted-out job).
- **Watchdog state resets** on restart (`_watchdog_last_was_healthy=None`, scheduler.py:101) — None→True is silent, so no spurious alert; first watchdog fire ~15 min post-start.
- **Verification that the digest is from the NEW process:** old PID confirmed dead + exactly one pgrep hit + "Morning digest sent" (scheduler.py:468) appearing in the NEW log file at 14:00 Oslo today (2026-06-12 is a regular Fri trading session, so the phase-51.3 gate at scheduler.py:444 passes). Exactly ONE digest in Slack = no duplicate.
- **Future restarts:** use `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot` (source 2), superseding the `pkill` doctrine from the pre-launchd era.

## Recommendations

**GO — now.** Execute in the 08:10-08:55 UTC window (mid-morning Oslo, exactly where we are): no scheduled tick, nightlies done, digests 4+/13+ hours out, and today's 14:00-Oslo morning digest gives same-day verification from the new process. Monitor-crontab removal MUST precede the kill (it is the only mechanism that can create a duplicate). Plist = backend template minus caffeinate, minus DEV_LOCALHOST_BYPASS, with `.venv/bin/python -m backend.slack_bot.app`, WorkingDirectory=repo root, logs to `handoff/logs/slack_bot.log`, ThrottleInterval=5, ProcessType=Interactive + LegacyTimers=true. Optional follow-up (NOT this step): alert-only down-detector to replace the monitor's iMessage, and a SIGTERM handler — both are out of 62.1 scope.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 12,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
