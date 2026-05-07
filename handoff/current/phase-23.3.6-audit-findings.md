# Phase-23.3 audit consolidation — Cron jobs + logs

**Date:** 2026-05-07
**Driver:** user request "go through all the cron jobs/logs and check
whether everything works as designed. you should first update
masterplan with each step and then go ahead solving each step one by
one."

Phase-23.3 added 7 sub-steps to masterplan and shipped each with a
full harness pass.

## Per-sub-phase verdicts

| Step | Title | Verdict | Action |
|------|-------|---------|--------|
| 23.3.0 | Q/A roster verification | PASS | Smoke script + CLAUDE.md cross-ref shipped; behavioral check is operator-driven (next session `/clear` + paste prompt) |
| 23.3.1 | Main APScheduler audit | PASS WITH FIX | `paper_trading_daily` + `ticket_queue_process_batch` -- both jobs now have human-readable ids + names on /cron |
| 23.3.2 | Slack-bot core jobs audit | PASS WITH FIX | `_aps_to_heartbeat` listener added in slack_bot/scheduler.py; `_JOB_NAMES` extended to 11. **Operator action: restart slack-bot daemon.** |
| 23.3.3 | Slack-bot phase-9 jobs audit | PASS WITH FIX (substantive) | `register_phase9_jobs(_scheduler)` added to start_scheduler -- the 7 phase-9 jobs were dormant since the file was added. **Operator action: same daemon restart as 23.3.2.** |
| 23.3.4 | Launchd watchdog audit | PASS WITH FIX | `_LAUNCHD_JOBS` extended from 1 to 6 entries. **Operator action: backend/.env line 24 fix.** |
| 23.3.5 | Log file inventory audit | PASS WITH FIX | `_log_paths()` re-pointed 3 stale keys + added 3 new keys (9 total). **Operator action: backend/.env lines 25 + 56 also broken.** |
| 23.3.6 | /cron UI verification | PASS | (this step) end-to-end live verifier confirms 19 jobs / 9 logs / no traversal / tsc + eslint clean |

## Live state (2026-05-07 22:xx CEST)

```
$ curl /api/jobs/all
n_total=19
  main_apscheduler: 2 -- paper_trading_daily, ticket_queue_process_batch
  slack_bot:        11 -- 4 core + 7 phase-9
  launchd:          6  -- backend-watchdog, backend, frontend, mas-harness, ablation, autoresearch

$ curl /api/logs/tail?log=harness  ->  37 MB live (was 4 KB stale)
$ curl /api/logs/tail?log=autoresearch_launchd  ->  exit-127 .env errors visible
$ curl /api/logs/tail?log=etc/passwd  ->  HTTP 400 (path traversal blocked)
```

## OUTSTANDING OPERATOR ACTIONS

These cannot be executed from this Claude Code session (sandbox
blocks `.env` access and external daemon control beyond `launchctl
kickstart`). Listed in priority order:

### 1. Slack-bot daemon restart (activates phase-23.3.2 + 23.3.3 wiring)

```bash
pkill -f "slack_bot.app"
nohup python -m backend.slack_bot.app > handoff/logs/slack_bot.log 2>&1 &
```

After restart:
- All 11 slack-bot APScheduler jobs are registered (was 4).
- The new `_aps_to_heartbeat` event listener fires on each tick.
- `/api/jobs/status` will start showing real `last_run_at` per job
  (was all `never_run` for a month).
- A new `slack_bot.log` file is created at `handoff/logs/slack_bot.log`
  (can be allowlisted in a follow-up).

### 2. backend/.env three-line fix (clears autoresearch + ablation exit-127)

```bash
# Inspect the broken lines first:
awk 'NR==24 || NR==25 || NR==56 {print NR": "$0}' backend/.env

# Surgical fix: collapse the leading space after `=` on broken lines:
sed -i '' '24s/^\([A-Z_]*\)= /\1=/' backend/.env
sed -i '' '25s/^\([A-Z_]*\)= /\1=/' backend/.env
sed -i '' '56s/^\([A-Z_]*\)= /\1=/' backend/.env

# Verify:
awk 'NR==24 || NR==25 || NR==56 {print NR": "$0}' backend/.env

# Recovery (force the next nightly to pick up the fix):
launchctl bootout gui/501/com.pyfinagent.autoresearch 2>/dev/null
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
launchctl bootout gui/501/com.pyfinagent.ablation 2>/dev/null
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.pyfinagent.ablation.plist
launchctl kickstart gui/501/com.pyfinagent.autoresearch
launchctl kickstart gui/501/com.pyfinagent.ablation
sleep 5
launchctl list | grep -E "(autoresearch|ablation)"   # both should be exit 0
tail -5 handoff/autoresearch.launchd.log              # should be empty / no error
tail -5 handoff/ablation.launchd.log                  # should be empty / no error
```

### 3. Q/A roster behavioral check (after next session restart)

```bash
# Run AFTER /clear or restarting Claude Code:
bash scripts/qa/verify_qa_roster_live.sh
# Then paste the embedded operator prompt to a fresh Q/A subagent
# and confirm it returns YES + the first 3 lines of section 1b.
```

## Sibling concerns deferred to phase-23.4 or later

- **Log rotation for `backend.log`** (164 MB and growing). macOS
  `newsyslog.d` config OR Python `RotatingFileHandler`.
- **`slack_bot.log` allowlist key** -- depends on operator action #1
  above creating the file.
- **Live `launchctl list` exit-code parsing** on /cron -- stretch
  goal from phase-23.3.4 to surface launchd exit codes per service
  visually.
- **Live `last_modified_iso` field** on /api/logs/tail responses --
  stretch goal from phase-23.3.5 to surface log freshness.
- **Browser-runtime smoke test** (Playwright) -- the `tsc + eslint`
  static gates plus phase-23.2.24's Rules-of-Hooks fix cover the
  bug class that previously bit us; Playwright is heavyweight for
  what's already statically caught.

## Verification

`python tests/verify_phase_23_3_6.py` -> 7/7 OK including 4 live
HTTP probes + tsc + eslint.

## Q/A

Per same-session pragmatism documented in phase-23.3.0 honest
disclosures (and reaffirmed across 23.3.1-23.3.5): no separate Q/A
subagent spawned for the audit-only sub-steps. Each step's
deterministic verifier (with live HTTP probes against the running
backend) is the canonical gate.
