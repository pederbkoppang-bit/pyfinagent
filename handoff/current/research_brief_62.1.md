# Research Brief — phase-62.1: Slack bot under launchd + restart on current code

**Tier:** moderate
**Date:** 2026-08-07 (measurements taken 20:21 CEST / 14:21 ET / 18:21 UTC)
**Status:** IN PROGRESS (write-first; internal half complete, external half pending)

## Scope

Step 62.1 criteria (immutable, `.claude/masterplan.json:15266-15277`):
1. `com.pyfinagent.slack-bot` launchd agent exists with `KeepAlive=true`, mirroring
   `com.pyfinagent.backend.plist`'s environment shape; old manual PID dead.
2. `ps lstart` of the launchd-managed bot process is LATER than the newest git
   commit touching `backend/slack_bot/` (verbatim paste of both).
3. A morning or evening digest observed in Slack from the NEW process.

---

## HEADLINE

**This is a restart + evidence cycle, not a plist-repair cycle.** The plist is
correct and criterion 1 is already satisfied. Criterion 2 fails by 9 days, and the
staleness is **not cosmetic** — the running process carries three defects fixed
after it started, two of which are in a nightly job that runs *inside this very
process*. Criterion 3 has a **zero-spam** evidence path: tonight's naturally
scheduled evening digest at 23:00 CEST.

---

## 1. Verified live state (I re-measured; did not trust the handoff)

| Fact | Measured value | Source |
|------|----------------|--------|
| launchd job | `658  0  com.pyfinagent.slack-bot` | `launchctl list` |
| state | `state = running`, `runs = 1`, `last exit code = (never exited)` | `launchctl print gui/501/com.pyfinagent.slack-bot` |
| program | `<repo>/.venv/bin/python -m backend.slack_bot.app` | same |
| process start | `tir. 28 jul. 18.39.22 2026` (etime 10-01:40) | `ps -o lstart= -p 658` |
| newest slack_bot commit | `2026-08-06 19:42:47 +0200` — `18659bc3` | `git log -1 --format=%ci -- backend/slack_bot/` |
| plist mtime | `12 jun. 10:21` | `ls -la ~/Library/LaunchAgents/` |

**Criterion 2 delta: the process predates the newest commit by 9 days 1 hour.**

Sibling PIDs 650 (anthropic-bridge), 658 (slack-bot), 668 (claude-code-proxy) are
low and sequential — consistent with a **machine reboot on 2026-07-28** starting
everything via `RunAtLoad`, not a deliberate restart. This is why the situation
partially self-healed: the masterplan's OPERATOR-ACTION entry
(`.claude/masterplan.json:19584`) recorded pid 83982 started **13 June** as of
2026-07-25. That 42-day-stale process is gone. The debt is now 9 days, not 42, and
the operator-action entry's MEASURED STATE block is stale — it must be re-measured
before anyone acts on it.

---

## 2. Criterion 1 — plist comparison (already satisfied, with two notes)

Read in full: `~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist` and
`~/Library/LaunchAgents/com.pyfinagent.backend.plist`.

**Identical in both** (the "environment shape" criterion 1 names):
`PATH` (byte-identical, `.venv/bin` first), `PYTHONUNBUFFERED=1`, `KeepAlive=true`,
`RunAtLoad=true`, `LegacyTimers=true`, `ProcessType=Interactive`,
`SoftResourceLimits.NumberOfFiles=8192`, `ThrottleInterval=5`, `WorkingDirectory`,
`StandardOutPath`==`StandardErrorPath`.

**Keys in backend, absent from slack-bot:**

| Key | Matters? | Why |
|-----|----------|-----|
| `CLAUDE_CODE_OAUTH_TOKEN` | **No** | Backend-only (Max rail for the autonomous loop). The bot has no CC-rail call path. *(Value deliberately not reproduced here — it is a live credential sitting in a world-readable-by-owner plist. Worth a separate hygiene step; out of 62.1 scope.)* |
| `DEV_LOCALHOST_BYPASS=1` | **No — and it is correctly absent** | Per `.claude/rules/security.md`, this is the *server-side* half of the localhost auth rail; it belongs on the process being called, not the caller. The bot is the *client* (`_LOCAL_BACKEND_URL = http://127.0.0.1:8000`, `scheduler.py:95`). |
| `HardResourceLimits.NumberOfFiles=16384` | **Marginal** | Bot holds one Socket Mode WebSocket + short-lived httpx clients. Soft cap 8192 is ample. |
| `caffeinate -i -s` wrapper in `ProgramArguments` | **Worth knowing, not a defect** | Backend prevents idle sleep for the whole machine; because backend and bot co-reside, the bot inherits wake coverage transitively. Adding `caffeinate` to the bot would be redundant. |

**Verdict on criterion 1: SATISFIED as written.** The criterion says "mirroring
`com.pyfinagent.backend.plist`'s environment shape", and the shape that matters
(PATH/venv/PYTHONUNBUFFERED/KeepAlive/RunAtLoad/WorkingDirectory) mirrors exactly.
The three missing keys are each correctly missing. The old manual PID (26147, and
its successor 83982) is dead — `launchctl print` shows `runs = 1` against a
launchd-owned pid 658, and there is exactly one `backend.slack_bot.app` process.

---

## 3. Criterion 2 — what the stale process is actually missing

`git log --since=2026-07-28 --oneline -- backend/slack_bot/` returns **three
commits**, all landed 2026-08-06:

| Commit | Step | What the running process is doing wrong RIGHT NOW |
|--------|------|---------------------------------------------------|
| `d10188ef` | 82.39 | `nightly_outcome_rebuild`'s BigQuery fetch selects `timestamp` and `realized_pnl` — **neither column exists** (`paper_trades` has `created_at` STRING and `realized_pnl_pct` FLOAT). BQ answers 400 `Unrecognized name: timestamp`; the fail-open `except` returns `[]`. The job "succeeds" nightly over ZERO trades. Dead since 2026-05-11. |
| `bb41eb96` | 82.48 | The outcome **write** emits a 5-column schema that never existed, so `outcome_tracking` receives nothing. Also adds the dedup that stops one SELL landing ~30x, and a P1 alert on a REJECTED write. |
| `18659bc3` | 82.59 | Two Bolt assistant listeners (`thread_started`, `thread_context_changed`) are called with a kwarg set that fits a *third* handler. **Every invocation raises TypeError.** Bolt's `AsyncioListenerRunner` catches it into `logger.exception` and acks *before* the listener, so Slack always saw 200. Symptom: blank assistant panel — no welcome message, no suggested prompts. Broken since April. |

**Answer to the lead's critical question: not merely missing features — the
running process has three live defects.** And they are not in dormant code:

- `nightly_outcome_rebuild` is registered **in this process** by
  `register_phase9_jobs` (`scheduler.py:306`, table at `:1166-1178`), firing daily
  at **hour=3 UTC** with `misfire_grace_time=3600, coalesce=True`. So 82.39 and
  82.48 are broken *inside the process the restart replaces*. Every night since
  2026-07-28 that job has run against a 400 and written nothing.
- The 82.59 listeners fire on any Slack assistant-panel interaction.

There is a **fourth**, subtler consequence: `_alert_fetch_failure` and
`_alert_write_rejected` (the P1 pagers 82.39/82.48 added so this class stops being
silent) do not exist in the running process. The restart is what arms them.

---

## 4. Criterion 3 — observing a digest with ZERO unsolicited messages

**The digests are already firing on a natural schedule, and tonight's has not
happened yet.** Measured from `handoff/logs/slack_bot.log` (6.4 MB, live):

```
2026-08-06 14:00:03,001 INFO backend.slack_bot.scheduler: Morning digest sent
2026-08-06 23:00:07,878 INFO backend.slack_bot.scheduler: Evening digest sent
2026-08-07 14:00:17,338 INFO backend.slack_bot.scheduler: Morning digest sent
```

- Cron is pinned to `America/New_York` (`scheduler.py:232,:244`), hours from
  `settings.morning_digest_hour=8` / `evening_digest_hour=17`
  (`backend/config/settings.py:626-627`). Observed fires at **14:00 and 23:00
  CEST** confirm the defaults are in force (ET+6 in summer) — no `.env` override
  is active. *(I am sandbox-denied on `backend/.env`; the log is the stronger
  evidence anyway because it is what the process actually did.)*
- Weekend/holiday guard `_is_us_trading_day_now()` (`scheduler.py:345-355`) is
  working — 2026-08-01/02 logged `skipped: not a US trading day`.
- **2026-08-07 is a Friday and a US trading session.** Current time 20:21 CEST.
  **Tonight's evening digest fires at 23:00 CEST, ~2h39m from now.**

**Recommended evidence path (no message I originate):** restart now, let the
23:00 digest fire from the new process, then capture:

1. `handoff/logs/slack_bot.log` line `Evening digest sent` (emitted at
   `scheduler.py:632`) with a timestamp **after** the new process's start.
2. The restart banner lines that precede it in the same log
   (`Slack bot starting in Socket Mode...`, `app.py:72`; `Scheduler started: ...`,
   `scheduler.py:282`) — these bind the digest to the *new* process, which is
   exactly what criterion 3 asks and what a permalink alone would not prove.
3. Optionally the operator's Slack permalink for the 23:00 post (operator-only;
   the log pair above is sufficient and honest without it).

**What NOT to use.** `backend/slack_bot/digest_test.py` exists and looks like a
dry-run, but it is not one: `_run()` calls `client.chat_postMessage` (`:34`) —
it **posts a real message**, just to `SLACK_TEST_CHANNEL_ID`. Worse for our
purpose, it is a standalone `slack_sdk.WebClient` script, so it proves nothing
about the bot process. Using it would both spam a channel and produce evidence
that does not satisfy criterion 3.

**There is no true dry-run/preview path for the digest.** `_send_morning_digest`
and `_send_evening_digest` post unconditionally once the trading-day guard passes.
Honest conclusion: the naturally-scheduled 23:00 digest is the only non-spam
evidence, and it is available tonight.

---

## 5. Restart mechanics

**`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot` is correct here**,
and the auto-memory intuition transfers — but for a *different and stronger* reason
than the frontend case, and one prior memory is now wrong.

- **`pkill` is blocked by policy, not just inadvisable.**
  `.claude/hooks/pre-tool-use-danger.sh:107-110` blocks any `pkill`/`killall`
  whose command string matches `python|uvicorn|next|slack_bot`, with the message
  pointing at `launchctl kickstart -k`. It would also race `KeepAlive=true`:
  launchd restarts within `ThrottleInterval=5`s, so a kill-then-start sequence can
  produce a second instance — the exact duplicate-digest hazard
  `scripts/slack_bot_monitor.sh:3-9` was rewritten to avoid.
- **Removal verbs are blocked; `kickstart` is explicitly allowed.**
  `pre-tool-use-danger.sh:176-177` blocks `launchctl bootout|unload|remove|disable`
  on any `com.pyfinagent.*` label, with the rail message "kickstart is the allowed
  restart path". So no `bootout`/`bootstrap` reload is available to an agent
  session — and none is needed, because the plist is unchanged.
- **`-k` is required.** `scripts/slack_bot_monitor.sh:28` uses bare `kickstart`
  because it only fires when the bot is already **down**. Our process is **up**, so
  we need `-k` (kill the running instance, then restart).
- **STALE MEMORY — correct it.** `.claude/agent-memory/researcher/
  project_slack_digest_calendar_guard.md:18` states "There is NO launchd label for
  the slack-bot ... restart via `pkill -f "backend.slack_bot.app"` + relaunch".
  That was true on 2026-06-01 and is **false now** (plist installed 2026-06-12) and
  the advice is now hook-blocked. The sibling memory
  `project_slack_bot_supervision_topology.md` already has it right.

### What a restart loses (bounded, ~5-20s)

| State | Lost? | Consequence |
|-------|-------|-------------|
| Socket Mode WebSocket | Yes | Slack buffers and redelivers unacked envelopes. Commands issued during the gap may be delayed, not dropped. |
| APScheduler jobs | Yes — `AsyncIOScheduler()` at `scheduler.py:224` takes **no jobstore**, so it is the default in-memory store | **Nothing replays.** Jobs are re-added at startup and `next_run_time` is computed from *now*. Today's already-sent 14:00 morning digest cannot re-fire. |
| `operator_tokens` `_seen_events` dedupe set | Yes — documented process-lifetime (`operator_tokens.py:61`) | A Slack redelivery straddling the restart could double-append one token line. Explicitly accepted in the module docstring: "sessions treat identical raw+slack_ts as one token". |
| `handoff/away_ops/tokens_cursor` | **No** | On disk, and **the bot never writes it** (`operator_tokens.py:174-176`, "consumed by away SESSIONS, never by the bot"). |
| Watchdog transition state (`_watchdog_last_was_healthy`, `_cycle_heartbeat_last_was_stale`, `_ingestion_silence_last_was_stale`) | Yes — reset to `None` | Intentional per `scheduler.py:99-100`: "first-fire state is the post-restart baseline". Worst case is one extra P1 if the first post-restart probe finds the backend unhealthy. |
| Ticket queue / SLA monitor / stuck-task reaper | Restarted as asyncio tasks (`app.py:60,64,68`) | Backing store is BQ/tickets.db, not memory. |

### Does a restart RE-FIRE anything? (the lead's 62.2 concern)

**The "stale RESUME re-fires" concern does not apply to a bot restart.** Traced in
full: the bot's only token action is `append_operator_token`
(`operator_tokens.py:119-172`) — it parses, dedupes, and **appends a JSONL line**.
It takes no action on the token's meaning. Acting on a token (including any
`.env` write or kill-switch API call) is done by away *sessions*, which gate on
`unapplied_tokens()` vs the on-disk cursor. Restarting the bot re-reads nothing
and re-applies nothing.

**One thing does deliberately re-fire:** `daily_price_refresh_catchup`
(`scheduler.py:330-337`), a one-shot `date` job at `now + 20s` on every startup.
This is phase-47.1 catch-up-on-start, and the inline comment documents why it is
safe: `ingest_prices` is idempotent at the BQ level (dedup on `(ticker, date)`), so
a redundant same-day run inserts ~0 rows. Expect a price-refresh burst ~20s after
restart. It costs a yfinance pull and a BQ round trip; it is not a hazard, but it
is the one observable side effect and should be disclosed in the live_check rather
than discovered.

---

## 6. What the bot registers at startup (restart blast radius)

`app.py::main()` → `create_app()` registers `register_commands`,
`register_assistant_lifecycle`, `register_governance` (`app.py:32-34`), then:

- `start_scheduler(app)` (`app.py:56`) → 4 core jobs: `morning_digest`
  (`scheduler.py:227`), `evening_digest` (`:239`), `watchdog_health_check`
  (`:251`, interval `watchdog_interval_minutes=15`), `prompt_leak_redteam`
  (`:263`, 03:15 ET).
- `register_phase9_jobs` (`:306`) → 7 jobs, all UTC-pinned with
  `misfire_grace_time` + `coalesce=True` (`:1166-1178`): hourly signal warmup
  (`minute=5`), daily price refresh (`hour=1`), nightly MDA retrain (`hour=2`),
  **nightly outcome rebuild (`hour=3`)**, nightly (`hour=4`), weekly FRED
  (`mon hour=5`), weekly data integrity (`sun hour=2`) — plus `hour=6`.
- `_seed_next_run_registry()` (`:314`) → one `status="scheduled"` heartbeat POST
  per job so `/api/jobs/all` has `next_run` before anything fires.
- Three background asyncio tasks: ticket queue processor (30s),
  SLA monitor (300s), stuck-task reaper (60s) (`app.py:60,64,68`).
- An APScheduler listener pushing every terminal job event to
  `/api/jobs/heartbeat` (`:276-279`).

**Blast radius: ~5-20 seconds.** Nothing in the list is stateful across restart
except the in-memory items already tabled above. `KeepAlive=true` +
`ThrottleInterval=5` means launchd brings it straight back.

---

## 7. Per-criterion remaining work

| # | Criterion | Status now | Remaining work |
|---|-----------|-----------|----------------|
| 1 | plist exists, `KeepAlive=true`, mirrors backend env shape; old manual PID dead | **SATISFIED** | Paste the plist diff table (§2) into `experiment_results.md`. No plist edit needed. |
| 2 | `ps lstart` LATER than newest `backend/slack_bot/` commit | **FAILS** (28 Jul vs 06 Aug) | Run the kickstart (§8). Paste both verbatim. |
| 3 | digest observed in Slack from the NEW process | **NOT YET** | Wait for tonight's 23:00 CEST evening digest; capture the log triple (§4). |

Note on criterion 2's durability: it is a **moving target**. Any future
`backend/slack_bot/` commit re-breaks it. The restart must be the *last* action
before the evidence capture, and no slack_bot file may be touched between the
restart and the flip (cf. `feedback_freeze_the_tree_during_evaluate`).

---

## 8. Restart runbook (recommended; I did NOT execute — read-only)

**Pre-flight — import smoke test first.** The bot is currently *up*; if HEAD has
an import error, `kickstart -k` kills a working process and `KeepAlive` crash-loops
against `ThrottleInterval=5`. This check is cheap and is the difference between a
restart and an outage.

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
.venv/bin/python -c "import backend.slack_bot.app; print('IMPORT OK')"
```

**Record the before-state** (criterion-2 evidence needs both sides):

```bash
OLD=$(launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | awk '/pid =/{print $3}')
echo "OLD pid=$OLD"; ps -o lstart= -p "$OLD"
git log -1 --format=%ci -- backend/slack_bot/
```

**Restart:**

```bash
launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot
```

**Verify (allow ~10s):**

```bash
sleep 10
launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | grep -E 'state|pid|runs|last exit'
NEW=$(launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | awk '/pid =/{print $3}')
ps -o lstart= -p "$NEW"
kill -0 "$OLD" 2>/dev/null && echo "OLD STILL ALIVE -- INVESTIGATE" || echo "OLD pid dead (expected)"
tail -40 handoff/logs/slack_bot.log
```

**Expected observables:**

- `runs` increments 1 → 2; `pid` changes from 658; `state = running`.
- `ps -o lstart=` on the new pid shows today, **later than 2026-08-06 19:42:47**.
- Log shows, in order: `Slack bot starting in Socket Mode...` (`app.py:72`),
  `Scheduler started: morning digest at 8:00, evening digest at 17:00, watchdog
  every 15 min` (`scheduler.py:282`), `phase-9 jobs registered: [...]` (`:307`),
  `phase-47.1: scheduled daily_price_refresh catch-up (+20s, idempotent by day)`
  (`:339`), then ~20s later the price-refresh run.
- At 23:00 CEST: `Evening digest sent` (`scheduler.py:632`).

**Rollback if it crash-loops:** `launchctl print` will show `runs` climbing with a
non-zero `last exit code`. `git log` the last slack_bot commit and revert it; do
**not** reach for `bootout` (hook-blocked) — fix forward and `kickstart -k` again.

---

## 9. External research

### Search-query composition (three-variant discipline)

- **Current-year frontier:** `launchctl kickstart vs bootout bootstrap restart
  LaunchAgent 2026`; `supervising long-running Slack Socket Mode bot process
  restart production 2026`
- **Last-2-year window:** `launchd changes macOS Tahoe 26 LaunchAgent KeepAlive
  2025 2026`
- **Year-less canonical:** `launchd KeepAlive RunAtLoad launchctl kickstart -k
  restart semantics`; `launchd KeepAlive restart running job correct verb`;
  `APScheduler misfire_grace_time coalesce memory jobstore restart replay missed
  jobs`; `slack bolt socket mode reconnect graceful shutdown SIGTERM close_async`

### Read in full (9; counts toward the gate)

| URL | Accessed | Kind | Key finding |
|-----|----------|------|-------------|
| https://keith.github.io/xcode-man-pages/launchd.plist.5.html | 2026-08-07 | official man page | `KeepAlive` true = "unconditionally keep the job alive"; it "implicitly implies `RunAtLoad`". `ThrottleInterval`: "jobs will not be spawned more than once every 10 seconds" by default (ours overrides to 5). `LegacyTimers` true opts out of timer coalescing — "less efficient but more precise". |
| https://ss64.com/mac/launchctl.html | 2026-08-07 | command reference | **The decisive quote for the restart verb:** `kickstart [-kp] service-target` — "**-k   If the service is already running, kill the running instance before restarting the service.**" `bootout` = "deactivates a domain or service (functionally equivalent to the legacy unload)". |
| https://www.launchd.info/ | 2026-08-07 | authoritative tutorial | KeepAlive true "run[s] the job as soon as the job definition is loaded and restart it should it ever go down". Modern verbs (10.10+) are kickstart / bootstrap / bootout. "Remember to always (re)load a job definition after changing it. Just saving it will leave launchd oblivious to the change." |
| https://developer.apple.com/library/archive/.../CreatingLaunchdJobs.html | 2026-08-07 | Apple official | `KeepAlive` true "indicates that this job needs to be running at all times... so launchd should always try to keep this job running." Apple discourages non-on-demand daemons generally, but a Socket Mode client has no on-demand trigger, so KeepAlive is correct here. |
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-08-07 | official docs | Misfire: "the scheduler will then check each missed execution time against the job's `misfire_grace_time`". Coalescing: "if coalescing is enabled... it will only trigger it once". Job stores: "The default job store simply keeps the jobs in memory". **Does NOT state what happens to memory-store jobs across a restart** — see the adversarial note below. |
| https://github.com/agronholm/apscheduler/issues/1095 | 2026-08-07 | upstream issue | Reports that missed runs do NOT execute immediately; next fire time resets to the original schedule even with `misfire_grace_time=None`. No maintainer response in the thread. Corroborates "APScheduler does not aggressively replay". |
| https://docs.slack.dev/apis/events-api/using-socket-mode/ | 2026-08-07 | Slack official | "Connections refresh regularly." Disconnect warning ~10s ahead. "Your app still needs to acknowledge receiving _each event_ so that Slack knows whether to retry." For zero-downtime restarts: "you can use multiple connections for temporary active-active redundancy". |
| https://docs.slack.dev/tools/bolt-python/reference/adapter/socket_mode/async_base_handler.html | 2026-08-07 | Slack official API ref | `start_async()` "establishes a new connection and then starts infinite sleep"; `close_async()` "disconnects... and cleans the resources this instance holds up". No documented `auto_reconnect_enabled` on this page. |
| https://oneuptime.com/blog/post/2026-02-02-websocket-graceful-shutdown/view | 2026-08-07 | industry blog (pub. 2026-02-02) | On an abrupt kill: "clients receive no warning", "in-flight messages get dropped". Recommends SIGTERM → stop accepting → close frames → drain (5s per client, 30s overall). Client side: exponential backoff with jitter. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://eclecticlight.co/2025/10/03/welcome-to-tahoes-launch-angels/ | expert blog | Recency-scan hit; Tahoe Launch Angels + RunningBoard, tangential to user LaunchAgents |
| https://developer.apple.com/forums/thread/768741 | vendor forum | "Can we disable KeepAlive temporarily" — no bearing on our restart |
| https://developer.apple.com/forums/thread/22824 | vendor forum | KeepAlive restart-loop anecdote |
| https://www.manpagez.com/man/5/launchd.plist/ | man mirror | Duplicate of the keith.github.io man page already read |
| https://www.real-world-systems.com/docs/launchdPlist.1.html | 3rd-party mirror | Duplicate content, lower authority |
| https://gist.github.com/masklinn/a532dfe55bdeab3d60ab8e46ccc38a68 | community gist | launchctl cheat sheet; superseded by ss64 |
| https://inventivehq.com/knowledge-base/macos/how-to-manage-launchagents-launchdaemons-macos | vendor KB | Introductory, no new semantics |
| https://apscheduler.readthedocs.io/en/3.x/modules/job.html | official docs | Job attribute reference; the userguide covered the semantics |
| https://github.com/slackapi/bolt-js/issues/1906 | upstream issue | bolt-**js**, not bolt-python |
| https://github.com/slackapi/node-slack-sdk/issues/1243 | upstream issue | node SDK, wrong runtime |
| https://docs.slack.dev/tools/bolt-python/reference/adapter/socket_mode/index.html | official docs | Module index; the base-handler page carried the API |
| https://github.com/tjluoma/launchd-keepalive | community repo | Example plists only |
| https://github.com/dfeuerbach/slack_bot_ws | community repo | Alternative framework, not in use here |
| https://medium.com/@chetcorcos/a-simple-launchd-tutorial-9fecfcf2dbb3 | blog | Introductory duplicate of launchd.info |
| https://discussions.apple.com/thread/256146694 | user forum | Tahoe LaunchAgent breakage anecdote, unverified |

**Source-quality caveat worth recording.** Several searches returned GitHub issues
under `openclaw/openclaw` and `NousResearch/hermes-agent` whose titles matched this
topic almost too precisely (Socket Mode restart loops, `launchd_restart` missing
`bootout`, KeepAlive restart loops). I did not fetch or cite them: I could not
establish that those repositories/issues are genuine, and the pattern is consistent
with low-quality or generated SEO content. The same applies to `aifreeapi.com`,
`apiscout.dev`, `markaicode.com`, and `digitalapplied.com` hits. **No conclusion in
this brief rests on them.** Everything load-bearing comes from Apple, the man page,
Slack's own docs, APScheduler's docs, or the installed source on this machine.

### Recency scan (last 2 years, 2024-2026) — PERFORMED

Three findings, one of which is materially important:

1. **macOS Tahoe 26 / Darwin 25 launchd changes** (eclecticlight.co, 2025-10-03):
   Tahoe adds "Launch Angels" with RunningBoard lifecycle-management keys. These
   are *system* jobs under `/System/Library/LaunchAngels`; they do not change user
   `LaunchAgent` semantics. Some reports of long-working LaunchAgents breaking under
   Tahoe exist but are unverified user-forum anecdotes.

2. **The conditional-KeepAlive first-spawn regression — and this project already
   hit it.** A recurring 2025-2026 report is that on Darwin 25, launchd evaluates a
   KeepAlive **dictionary** clause as a precondition for the *initial* spawn, where
   older macOS only applied the dict after `runs >= 1`. This machine is Darwin
   25.5.0. **Independent corroboration from this repo's own measured history:**
   `.claude/masterplan.json:19419` records that
   `com.pyfinagent.anthropic-bridge.plist` shipped with
   `KeepAlive = <dict><key>SuccessfulExit</key><false/></dict>`, and a measured
   `kill -9` left it **dead** (`runs = 1`, `state = not running`, still dead after
   21s); the recorded fix was changing KeepAlive to `<true/>`. That is the same
   failure class, observed locally.
   **Consequence for 62.1: the slack-bot plist uses the unconditional `<true/>`
   form and is therefore NOT exposed to this regression.** This raises confidence
   that `kickstart -k` will bring it straight back.

3. **2026 WebSocket graceful-shutdown practice** (oneuptime, 2026-02-02): confirms
   the drain expectation for an abrupt kill — in-flight messages dropped, clients
   should reconnect with backoff. Slack's own docs already cover the client half
   (auto-reconnect + retry-on-unacked), so the practical exposure here is a few
   seconds, not data loss.

**No 2024-2026 source supersedes the canonical guidance** that `kickstart -k` is
the correct restart verb for a loaded, running, KeepAlive LaunchAgent.

### Key findings (external → this decision)

1. **`kickstart -k` is exactly the documented primitive for this situation.**
   "If the service is already running, kill the running instance before restarting
   the service" (ss64 launchctl). Our service *is* running, so the bare `kickstart`
   used by `scripts/slack_bot_monitor.sh:28` is insufficient — that script only
   fires when the bot is down.

2. **No reload is needed because the plist is unchanged.** launchd.info: "always
   (re)load a job definition after changing it. Just saving it will leave launchd
   oblivious." We are changing *Python code*, not the plist, and the plist re-execs
   the interpreter — so `kickstart -k` picks up new code with no `bootout` (which
   is hook-blocked anyway).

3. **`KeepAlive=true` is the unconditional form** — "unconditionally keep the job
   alive" (man page), "restart it should it ever go down" (launchd.info). Combined
   with `ThrottleInterval=5`, the process returns within ~5s.

4. **Slack's client-side machinery absorbs the disconnect.** Slack requires
   per-event acknowledgement "so that Slack knows whether to retry", and Bolt's
   Socket Mode handler reconnects automatically. The exposure during a
   kill-and-restart is a brief window where an event may be redelivered — which is
   precisely the case `operator_tokens.py`'s dedupe set is documented to absorb,
   and whose cross-restart hole is documented and accepted.

5. **Slack's own zero-downtime recipe does not apply.** "Use multiple connections
   for temporary active-active redundancy" would mean running two bot instances —
   which for this project means **duplicate digests and duplicate phase-9
   schedulers**, the exact hazard `scripts/slack_bot_monitor.sh:3-9` was rewritten
   to prevent. Reject it; accept the ~5-20s gap instead.

### Consensus vs debate

**Consensus:** `kickstart -k` for a loaded running job; `bootout`+`bootstrap` only
when the plist itself changed or the job is not registered; `KeepAlive` true means
unconditional restart; APScheduler needs a persistent job store for jobs to survive
a restart.

**Genuine debate / gap — and I resolved it against source, not docs.** The
APScheduler user guide never states what happens to a **MemoryJobStore** job across
a process restart; it only says persistent stores serialize jobs. My brief asserts
"nothing replays", so I verified it in the installed package rather than inferring:

`apscheduler` **3.11.2**, `.venv/lib/python3.14/site-packages/apscheduler/schedulers/base.py:1066-1068`:

```python
# Calculate the next run time if there is none defined
if not hasattr(job, "next_run_time"):
    now = datetime.now(self.timezone)
    replacements["next_run_time"] = job.trigger.get_next_fire_time(None, now)
```

`get_next_fire_time(previous_fire_time=None, now)` computes the next fire strictly
**after `now`**. Since a memory store starts empty, every job at startup takes this
branch. **A cron tick that elapsed while the process was down cannot be replayed** —
`misfire_grace_time` (`scheduler.py:1166-1178`) only governs runs that were already
queued *within a live scheduler*, evaluated in `_process_jobs` (`base.py:1190-1228`).

**This is the adversarial check that matters**, because it cuts against the
intuition that `misfire_grace_time=21600` (6 hours, on the hourly warmup job) would
replay the morning. It will not. Confirmed by upstream issue #1095's independent
report that missed runs reset to the original schedule.

---

## 10. Recommendation

**(a) This is a restart + evidence cycle.** Not a plist repair (criterion 1 is
already satisfied and the plist is correct in every way that matters), and not
already-closeable (criterion 2 fails by 9 days and criterion 3 has no artifact).

**The restart is worth doing on its merits, independent of the criterion.** The
running process silently writes nothing to `outcome_tracking` every night and has a
dead Slack assistant panel. Three real defects, all fixed on disk, all dark.

**Sequence for tonight (unattended-safe):**

1. Import smoke test (§8) — **do not skip**; it is the only thing standing between
   a bad HEAD and a crash-loop against a currently-working bot.
2. Record before-state (old pid + lstart + newest commit).
3. `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot`
4. Verify new pid + `runs = 2` + startup log banner; confirm old pid dead.
5. **Wait for 23:00 CEST** and capture `Evening digest sent` from
   `handoff/logs/slack_bot.log`, together with the restart banner above it in the
   same file. That pair is the honest criterion-3 artifact.
6. Freeze `backend/slack_bot/` between step 3 and the masterplan flip — any new
   commit there re-breaks criterion 2.

**Do not** send a test message, do not run `digest_test.py`, do not start a second
instance for zero-downtime, and do not use `pkill` or `bootout` (both hook-blocked,
and both wrong here).

**Timing note.** It is 20:21 CEST; the digest fires at 23:00 CEST. If the restart
slips past 23:00, the next opportunity is Monday 2026-08-10 at 14:00 CEST — Saturday
and Sunday are non-trading days and the digests will log `skipped` (proven by the
2026-08-01/02 log lines). **Restarting before 23:00 tonight saves three days.**

### Out-of-scope defects discovered (queue as their own steps per the standing rule)

1. **Plaintext `CLAUDE_CODE_OAUTH_TOKEN` in `~/Library/LaunchAgents/com.pyfinagent.backend.plist`**
   (`EnvironmentVariables`). A live Anthropic OAuth credential in a config file.
   Value deliberately not reproduced in this brief.
2. **Plaintext `SLACK_BOT_TOKEN` in the user crontab** — the `*/2 * * * *`
   `slack_mention_checker.sh` line `export`s the bot token inline, so it is visible
   to `crontab -l` and in the command string of the spawned shell every 2 minutes.
3. **The masterplan OPERATOR-ACTION entry at `.claude/masterplan.json:19584` carries
   a stale MEASURED STATE block** (pid 83982 / 13 June / "42 days"). Reality is pid
   658 / 28 July / 9 days. Anyone acting on it without re-measuring would write a
   false live_check.
4. **Stale researcher memory** `.claude/agent-memory/researcher/project_slack_digest_calendar_guard.md:18`
   claims there is no launchd label for the slack-bot and recommends `pkill` —
   false since 2026-06-12 and now hook-blocked. *(Corrected in this session.)*

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**9**)
- [x] 10+ unique URLs total incl. snippet-only (**24**)
- [x] Recency scan (last 2 years) performed + reported (§9, three findings)
- [x] Full pages read, not abstracts, for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (plists, `app.py`,
      `scheduler.py`, `operator_tokens.py`, `digest_test.py`, `jobs/`, the two
      hooks, `slack_bot_monitor.sh`, the live log, crontab, installed apscheduler)
- [x] Contradictions noted (APScheduler docs gap resolved against installed source;
      Slack's active-active advice explicitly rejected for this project)
- [x] All claims cited per-claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 15,
  "urls_collected": 24,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Criterion 1 is already satisfied: the slack-bot plist exists with KeepAlive=true and mirrors the backend plist's PATH/venv/PYTHONUNBUFFERED/RunAtLoad/WorkingDirectory shape; the three backend-only keys (CLAUDE_CODE_OAUTH_TOKEN, DEV_LOCALHOST_BYPASS, HardResourceLimits) are each correctly absent. Criterion 2 fails: pid 658 started 2026-07-28 18:39 (a reboot, not a deliberate start) versus newest slack_bot commit 2026-08-06 19:42. The staleness is not cosmetic -- the running process carries three defects fixed since: 82.39 and 82.48 leave nightly_outcome_rebuild fetching non-existent BQ columns and writing a schema that never existed, and that job runs INSIDE this process at hour=3 UTC; 82.59 leaves two Bolt listeners raising TypeError. Criterion 3 has a zero-spam path: digests fire 14:00 and 23:00 CEST and tonight's evening digest is still ahead. digest_test.py is NOT a dry run (it posts). Restart verb is launchctl kickstart -k; pkill and bootout are both blocked by pre-tool-use-danger.sh. Nothing replays on restart -- verified against installed apscheduler 3.11.2 base.py:1066-1068, not just docs. Recommendation: restart + evidence cycle, before 23:00 tonight or it slips to Monday.",
  "brief_path": "handoff/current/research_brief_62.1.md",
  "gate_passed": true
}
```
