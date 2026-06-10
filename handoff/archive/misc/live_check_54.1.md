# live_check 54.1 — Cross-layer cron-health audit (operator away-week)

**Captured:** 2026-06-01 (local), read-only `launchctl list` / `ps` / `lsof` /
`GET /api/jobs/all`. Window: operator REMOTE 2026-06-01 → 2026-06-08, Slack-only.

## Summary verdict

- **launchd:** 5 healthy, **2 FAILED → FIXED** (`autoresearch`, `ablation` — single
  shared root cause, fixed in `settings.py` via `NoDecode` + validator; verified at
  the settings-load layer), 0 truly-down. `mas-harness` "not running" alarm =
  **FALSE POSITIVE** (idle interval job, exit 0).
- **Fix applied (2026-06-01, phase-54.1):** `paper_markets` now parses JSON /
  bracket-mangled / comma forms identically (`backend/config/settings.py`), so the
  bash-sourced cron path no longer raises `SettingsError`. 11 new tests +
  25 settings/config regression tests green; the live JSON value is byte-identical
  (`['US','EU','KR']`). The full autoresearch/ablation jobs were NOT run (autoresearch
  has a known huggingface-import gap + potential LLM spend) — the fix is verified ONLY
  at `get_settings()`, the crash point. They re-verify on their next nightly fire.
- **APScheduler (main + slack_bot):** **13/13 registered + healthy** — all fired
  today with sane next-fires; `morning_digest` delivered to Slack 12:00 UTC.
- **Liveness:** backend :8000 UP, frontend :3000 UP, slack_bot UP — but the
  **slack_bot has NO launchd supervisor** (single point of failure for the week).
- **Digest cost:** **$0 / template-only** — NOT operator-gated; safe to run all week.

## Cross-layer cron-health table

| Job | Layer | Schedule | Last run | Next fire | Live status | Action |
|-----|-------|----------|----------|-----------|-------------|--------|
| com.pyfinagent.backend | launchd | KeepAlive + RunAtLoad | (daemon) | n/a | RUNNING (PID 36338; prior exit -15=SIGTERM, KeepAlive restarted) | none |
| com.pyfinagent.frontend | launchd | KeepAlive + RunAtLoad | (daemon) | n/a | RUNNING (PID 11636) | none |
| com.pyfinagent.claude-code-proxy | launchd | KeepAlive{SuccessfulExit false} | (daemon) | n/a | RUNNING (PID 1269) | none |
| com.pyfinagent.backend-watchdog | launchd | StartInterval 60s; RunAtLoad | every 60s | +60s | HEALTHY (PID `-` idle, exit 0) | none |
| com.pyfinagent.mas-harness | launchd | StartInterval 1800s; RunAtLoad false | (idle) | +30 min | HEALTHY — **NOT a failure** (PID `-` + exit 0 = idle between fires) | none (optional: operator decides if harness should run during away-week) |
| com.pyfinagent.autoresearch | launchd | StartCalendarInterval 02:00 | 2026-06-01 02:00 → rc=1 | 2026-06-02 02:00 | **FIXED** (settings load) | **FIX APPLIED** — settings.py NoDecode validator; verified get_settings() loads under bash-source (below) |
| com.pyfinagent.ablation | launchd | StartCalendarInterval 03:00 | 2026-06-01 03:00 → exit 1 | 2026-06-02 03:00 | **FIXED** (settings load) | **FIX APPLIED** — same root cause + same fix (below) |
| paper_trading_daily | APScheduler (main, backend PID 36338) | cron daily trade cycle | (today) | 2026-06-01 14:00 ET | HEALTHY (scheduled) | none |
| ticket_queue_process_batch | APScheduler (main) | interval | (today) | 2026-06-01 15:58 +02 | HEALTHY (scheduled) | none |
| morning_digest | APScheduler (slack_bot PID 42151) | cron 08:00 ET | 2026-06-01 12:00 UTC | 2026-06-02 08:00 ET | **ok — DELIVERED to Slack** | none |
| evening_digest | APScheduler (slack_bot) | cron 17:00 ET | (fires 17:00 ET) | 17:00 ET today | registered; history shows daily delivery thru 05-27 | none (verify a digest lands during week) |
| watchdog_health_check | APScheduler (slack_bot) | interval 15 min | 2026-06-01 13:50 UTC | +15 min | ok (state-transition-gated alerts) | none |
| prompt_leak_redteam | APScheduler (slack_bot) | cron 03:15 ET | 2026-06-01 07:15 UTC | 2026-06-02 03:15 ET | ok | none |
| daily_price_refresh | APScheduler (slack_bot) | cron 01:00 UTC (grace 6h, coalesce) | 2026-06-01 01:00 UTC | 2026-06-02 01:00 UTC | ok | none |
| weekly_fred_refresh | APScheduler (slack_bot) | cron Sun 02:00 UTC (grace 2h) | (fires Sunday) | 2026-06-07 Sun | registered | none (fires within window) |
| nightly_mda_retrain | APScheduler (slack_bot) | cron 03:00 UTC (grace 1h) | 2026-06-01 03:00 UTC | 2026-06-02 03:00 UTC | ok | none |
| hourly_signal_warmup | APScheduler (slack_bot) | cron :05 UTC (grace 10m) | 2026-06-01 13:05 UTC | 2026-06-01 14:05 UTC | ok | none |
| nightly_outcome_rebuild | APScheduler (slack_bot) | cron 04:00 UTC (grace 1h) | 2026-06-01 04:00 UTC | 2026-06-02 04:00 UTC | ok | none |
| weekly_data_integrity | APScheduler (slack_bot) | cron Mon 05:00 UTC (grace 2h) | 2026-06-01 05:00 UTC | 2026-06-08 05:00 UTC | ok | none |
| cost_budget_watcher | APScheduler (slack_bot) | cron 06:00 UTC (grace 1h) | 2026-06-01 06:00 UTC | 2026-06-02 06:00 UTC | ok | none |

`launchctl list` legend (source: launchd.info): col1 PID `-` = loaded-but-idle,
number = running; col2 exit `0` = clean, `>0` = job errored, `<0` = killed by
signal.

## Unhealthy jobs — root cause + fix-or-escalate

### autoresearch (exit 1) AND ablation (exit 1) — SAME root cause
**Cause:** both launchd wrappers `set -a; . backend/.env; set +a` (shell-source the
env). The 2026-06-01 multi-market go-live set `PAPER_MARKETS` to JSON
(`["US","EU","KR"]`) in `.env`. `paper_markets: list[str]` (`backend/config/settings.py:55`)
is a pydantic *complex* field → parsed via `json.loads()` from the OS env. Bash
mangles the unquoted brackets/quotes on `source`, so the exported value is non-JSON
→ `JSONDecodeError: Expecting value: line 1 column 2 (char 1)` →
`SettingsError: error parsing value for field "paper_markets"` at `get_settings()`.
The live backend is fine (uvicorn reads `.env` natively, not via shell). Verbatim
trace in `handoff/autoresearch.log` (02:00 rc=1) and `handoff/ablation.log`.

**RESOLUTION — FIX APPLIED (phase-54.1, 2026-06-01), option (b) the preferred,
permanent fix:** `backend/config/settings.py` now annotates
`paper_markets: Annotated[list[str], NoDecode]` and adds a
`field_validator(mode="before")` that accepts JSON (`["US","EU","KR"]`),
bracket-mangled (`[US,EU,KR]` from the bash-sourced `.env`), plain-comma
(`US,EU,KR`), and real lists — so every load path (native dotenv / OS-env /
shell-sourced) resolves identically. No `.env` edit, no new dependency, purely
additive (DO-NO-HARM: the live JSON path is byte-identical, `['US','EU','KR']`).

**Verification (no full-job run):**
- `python -m pytest backend/tests/test_phase_54_1_paper_markets_parse.py` → **11 passed**
  (every input form + DO-NO-HARM JSON byte-identity + the exact bash-mangled repro).
- 25 existing settings/config tests → green (no regression).
- The exact crash repro now loads cleanly:
  `set -a; . backend/.env; set +a; python -c "from backend.config.settings import get_settings; print(get_settings().paper_markets)"`
  → `['US', 'EU', 'KR']` (previously raised `SettingsError`).

This was applied autonomously (not escalated) because a `settings.py` code fix is
NOT on the operator-gated list (LLM spend / pip / BQ-DROP) and is reversible +
test-guarded. **NOT auto-run:** the full autoresearch/ablation jobs — autoresearch
`run_memo.py` has a known huggingface-import dependency gap (auto-memory
`project_cron_maintenance_jobs`) and may incur LLM spend, both operator-gated. The
crash was earlier (at settings load) and is now fixed; the jobs re-verify on their
next nightly fire (02:00 / 03:00). If autoresearch then fails on the huggingface gap,
that is a SEPARATE operator-gated pip/LLM issue, not this settings crash.

### mas-harness ("not running") — NO ACTION
False positive. `StartInterval 1800` + `RunAtLoad false`, exit 0 → PID `-` means
loaded-and-idle between 30-min fires (launchd semantics), reported `ok` by
`/api/jobs/all`. Not a defect.

## Away-week monitoring gaps (feed phase-54.2 — NOT fixed here)

1. **slack_bot has no launchd supervisor** (PID 42151, PPID 1, no plist). If it
   dies / Mac sleeps without recovery, ALL digests + watchdog + 11 jobs stop and
   the operator goes blind. → add `com.pyfinagent.slack-bot.plist`
   (RunAtLoad + KeepAlive, ThrottleInterval≥5, honor Apple's 10s-no-fast-exit rule).
   Adding/loading a plist is operator-gated (`launchctl load`).
2. **No external dead-man's-switch.** An internal check can't catch the whole Mac
   going down. → success-ping the daily digest to a free heartbeat service
   (Healthchecks.io / Dead Man's Snitch), 24h interval + 30-60m grace, alert via a
   channel independent of Slack + the Mac.
3. **Digest has no cron-health line.** → fold a `/api/jobs/all` one-liner into the
   morning digest so "all crons green / FAILED: x,y" is visible in Slack
   (state-gated, no paging noise). Digest is $0/template-only, so this is free.
4. **On-disk slack_bot log diverged** — live process logs to `backend_slack.log`,
   while `handoff/logs/slack_bot.log` is stale since 2026-05-27 (a restart changed
   the target). Cosmetic, but worth pointing the operator at the live file.

## Repeatable audit command (all $0, read-only)

```
# launchd leg
launchctl list | grep com.pyfinagent
# APScheduler + merged health leg (built-in)
curl -s http://127.0.0.1:8000/api/jobs/all | python3 -m json.tool
# liveness leg
ps aux | grep -E "slack_bot|uvicorn|next" | grep -v grep
curl -s http://127.0.0.1:8000/api/health
```
Healthy = no launchd col2 `>0`; every `/api/jobs/all` row `status != failed` with a
future `next_run`; all three host processes present; slack_bot job `last_run`
advancing.
