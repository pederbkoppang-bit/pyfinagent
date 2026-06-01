# Research Brief — phase-54.2: Reliable daily Slack digests for the away week

**Tier:** moderate
**Step:** phase-54.2 (guarantee operator receives daily Slack status updates for the 1-week remote window 2026-06-01 → 2026-06-08)
**Author:** researcher subagent
**Date:** 2026-06-01

---

## 0. Bottom line

**THE LOAD-BEARING UNKNOWN IS RESOLVED — and it overturns the 54.1 framing.** The
slack_bot (PID 42151) is **NOT unsupervised**. 54.1 checked only launchd and
concluded "no auto-restart"; that was incomplete. The bot has a **working
restart supervisor since 2026-04-01**: `scripts/slack_bot_monitor.sh` runs **every
5 minutes from the user crontab** (`*/5 * * * *`), greps for
`python.*backend.slack_bot.app`, and if absent `nohup`-restarts it from the venv
**and fires an iMessage to the operator's phone** (+4794810537, via the `imsg` CLI
which IS installed at `/opt/homebrew/bin/imsg`). The `nohup` is exactly why the bot
shows PPID 1 / no launchd plist. A second crontab line (`*/2 * * * *`)
`slack_mention_checker.sh` is a separate concern (does NOT start the bot).

**This makes a launchd KeepAlive plist the WRONG move for the away week — it is a
NET-NEGATIVE that risks the lifeline.** A `RunAtLoad`-true plist would create a
**SECOND bot instance** alongside the cron-managed one, and the cron monitor's
grep (`ps | grep ...slack_bot.app`) would then *always* see "a bot is running" and
never notice if the launchd one died — i.e. the two supervisors would mask each
other. Two instances means **two independent APScheduler schedulers**, so the
operator gets **DOUBLE-FIRED morning/evening digests AND a 2x nightly_mda_retrain /
hourly_signal_warmup / etc.** (Slack confirms a single app token allows up to 10
concurrent Socket Mode connections and routes each inbound payload to *one random*
connection — so slash-commands are not duplicated — but the **scheduler jobs are
per-process and fire in BOTH**.) That is the real double-fire risk, not Socket Mode
event duplication.

**The Mac will NOT sleep** during the away week: `pmset -g` shows
`sleep 1 (sleep prevented by caffeinate)` — the backend launchd job runs under
`caffeinate -i -s` (PID 36340), asserting a system-sleep-prevention assertion. So
54.1's "if the Mac sleeps" risk is already mitigated by existing infra.

**The digest is confirmed $0 / template-only** (formatters.py imports only `math` +
`datetime`; grep for any LLM/anthropic/openai/gemini import → NONE) → **NOT
operator-gated; safe to run + safe to send a live confirmation now.** Today
(2026-06-01) IS a US trading day, so the `_is_us_trading_day_now()` guard will not
suppress a confirmation digest.

**Recommended 54.2 (does NOT touch the running bot):** (1) send ONE live
confirmation digest via a **standalone one-shot script** that builds a bare
`AsyncWebClient(bot_token)` and calls the existing `format_morning_digest()` +
`chat_postMessage(channel)` — a Web-API POST that opens **NO Socket Mode
connection** and never touches PID 42151; (2) **harden the existing cron monitor**
(it is the right supervisor — keep it, don't replace it) + fold a **cron-health
one-liner** into the digest from `/api/jobs/all`; (3) optionally add an external
dead-man's-switch on the digest. **Do NOT add a launchd KeepAlive plist for the bot
during the away week** — it duplicates the supervisor and double-fires every job.
`gate_passed: true`.

---

## 1. External sources — READ IN FULL (floor ≥5; **7** read)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.slack.dev/apis/events-api/using-socket-mode/ | 2026-06-01 | official doc (Slack) | WebFetch (full) | "An app can maintain up to **10 concurrent WebSocket connections**." "When multiple connections are active, **each payload may be sent to _any_ of the connections**." "It's best **not to assume any particular pattern** for how payloads will be distributed." Multiple connections are FOR graceful-restart / load-balance / active-active redundancy. Refresh every few hours; 10-sec disconnect warning + `refresh_requested`. |
| 2 | https://github.com/slackapi/bolt-python/issues/462 | 2026-06-01 | official repo issue | WebFetch (full) | Confirms: "**events are not duplicated to all connections** … routed to a single connection, but there's no guaranteed pattern for which one." For guaranteed cross-process handling, **a job queue is needed, not multiple Socket Mode connections**. Issue itself unresolved (no maintainer promise of broadcast). |
| 3 | https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html | 2026-06-01 | official doc (Apple) | WebFetch (full) | **Verbatim 10-sec rule:** "If your daemon shuts down too quickly after being launched, launchd may think it has crashed. Daemons that continue this behavior may be **suspended and not launched again** … **do not shut down for at least 10 seconds after launch.**" KeepAlive = "always running" vs on-demand. Plists in `~/Library/LaunchAgents`. (Apple's doc omits ThrottleInterval/SuccessfulExit detail → source 5.) |
| 4 | https://github.com/MoonBoi9001/apple-juice/issues/25 | 2026-06-01 | practitioner issue (migration playbook) | WebFetch (full) | The canonical **nohup→launchd migration is exactly pyfinagent's situation.** nohup's flaw: "If the process dies unexpectedly (SIGKILL, OOM, crash), **nothing restarts it**." **Safe steps (verbatim intent): "Kill the existing nohup process first; detect and remove the old LaunchAgent; … Verify no duplicate processes exist before loading."** `KeepAlive{SuccessfulExit:false}` + `RunAtLoad:true`. Rollback = `launchctl bootout` + plist removal. |
| 5 | https://www.launchd.info/ | 2026-06-01 | authoritative ref (de-facto launchd manual) | WebFetch (full) | `KeepAlive` subkeys; **`SuccessfulExit:false` = "restarted until it succeeds"** (i.e. restart only on crash). **`ThrottleInterval` = "Time in seconds to wait between program invocations"** (min respawn gap). `RunAtLoad` = start at load/login. Load (10.10+): `launchctl bootstrap gui/$(id -u) <plist>`; unload: `launchctl bootout gui/$(id -u) <plist>`. **launchctl list col1 PID `-`=loaded-not-running / number=running; col2 0=clean / >0=errored / <0=signal-killed.** Always set StandardOut/ErrorPath. |
| 6 | https://docs.slack.dev/reference/methods/chat.postMessage/ | 2026-06-01 | official doc (Slack) | WebFetch (full) | Needs `chat:write`. Success → `ok:true` + `ts`. Errors: `channel_not_found`, `not_in_channel`, `missing_scope`, `token_revoked`, `rate_limited`, `invalid_blocks`. "**generally allow an app to post 1 message per second to a specific channel**"; on 429 read `Retry-After`. **"highly recommended that you include `text` to provide a fallback when using `blocks`"** (screen readers). **NO idempotency/dedup key** — retries CAN create duplicate messages (caller must guard). |
| 7 | https://docs.slack.dev/apis/events-api/comparing-http-socket-mode/ | 2026-06-01 | official doc (Slack) — **recency-scan full read** | WebFetch (full) | "**To have the highest possible reliability for application connectivity, we recommend using HTTP for production applications.**" WebSocket "subject to a network partition … the socket server backend **recycles containers serving connections every now and then, leading to occasional reliability issues**." Socket Mode = dev / on-prem-firewalled. WebSocket "Challenging to scale … limits the number of concurrent WebSocket connections to 10 per app." |

## 2. External sources — snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/slackapi/bolt-python/issues/579 | repo issue | "Does socket mode support broadcasting event to multiple instances?" — title alone confirms NO-broadcast; #462 read in full covers it |
| https://github.com/slackapi/bolt-js/issues/1263 | repo issue | "run multiple instances … receive ALL events?" — confirms you CANNOT (events split); corroborates source 1/2 |
| https://github.com/slackapi/bolt-js/issues/2188 | repo issue | "Receiving multiple events for simple messages" — the retry/duplicate-handler nuance (distinct from multi-connection); not load-bearing here |
| https://github.com/slackapi/python-slack-events-api/issues/93 | repo issue | "Socket mode is duplicating the messages" — duplication from app-level retries, not multi-connection; AlertDeduper-adjacent |
| https://api.slack.com/apis/socket-mode | official doc | Socket Mode index; specifics covered by source 1 |
| https://dev.to/.../openclaw-slack-in-2026-the-managed-vs-self-hosted-decision-dm1 | blog (2026) | "Socket Mode connections drop … Missed events during that window are gone. Slack doesn't replay them." Reinforces source 7; snippet sufficient |
| https://github.com/NousResearch/hermes-agent/issues/14326 (2026) | repo issue | **2026 real failure mode**: "Slack gateway can remain 'running' while Socket Mode is dead, requiring manual restart" — directly relevant; captured via search snippet (see §4) |
| https://github.com/openclaw/openclaw/issues/31287 (2026) | repo issue | "Socket Mode … channel events silently dropped … auto-reconnect" — 2026 silent-deafness failure mode; snippet |
| https://github.com/slackapi/bolt-js/issues/1151 | repo issue | "Socket mode is unreliable" — corroborates source 7 production stance |
| https://github.com/tjluoma/launchd-keepalive | repo | KeepAlive plist examples incl SuccessfulExit; corroborates sources 4/5 |
| https://keith.github.io/xcode-man-pages/launchd.plist.5.html | man page | launchd.plist(5) — ThrottleInterval default (10s) + KeepAlive subkeys; corroborates source 5 |
| https://www.manpagez.com/man/5/launchd.plist/ | man page | launchd.plist(5) mirror; snippet |
| https://slack.engineering/migrating-millions-of-concurrent-websockets-to-envoy/ | eng blog (Slack) | Slack's own Socket-Mode backend scale story; confirms container recycling (source 7) |
| https://docs.alertlogic.com/.../network-health-digest.htm | vendor doc | "daily issues … comparison of health statuses, top-ten, total open remediations" — digest-content pattern for §10 cron-health line |
| https://oxmaint.com/article/remote-maintenance-manager-site-performance | blog | Remote operator "structured daily/weekly cadence of dashboard reviews + exception responses" — signal-set for §10 |
| https://medium.com/@nisheet110/healthchecks-io-the-ultimate-guide-...-fd1b6bf311fc | blog | Healthchecks.io dead-man's-switch how-to — corroborates the 54.1 external-heartbeat option |

**Unique URLs collected: 23** (7 read-in-full + 16 snippet-only). Floor is 10.

## 3. Search-query variants run (3 per topic)

- **Topic 1 (Socket Mode multiple connections):** year-less canonical → "Slack Socket Mode multiple concurrent connections same app duplicate events" (surfaced the official Using-Socket-Mode doc + bolt-python #462 + bolt-js #1263/#2188); current-year → "...multiple connections load balancing **2026**" (Slack eng Envoy post, autokitteh private-socket, dev.to 2026). **Decisive answer obtained from the official doc + #462.**
- **Topic 2 (launchd migrate-from-manual):** last-2-year → "macOS launchd KeepAlive migrate from manual nohup process single instance daemon **2025**" (apple-juice #25 migration playbook + Apple dev doc + tjluoma repo); year-less canonical → launchd.info + launchd.plist(5).
- **Topic 3 (chat.postMessage reliability):** year-less canonical → official `chat.postMessage` method doc (read in full); rate-limit specifics already canonical from 54.1's archived read.
- **Topic 4 (remote-operator daily status):** year-less canonical → "daily status report remote operator minimum signal trading system health check digest" (AlertLogic health digest, oxmaint remote-cadence). SRE-handoff specifics (active-incidents / silenced-alerts / runbook-links; Google 2-3 actionable/shift) are canonical from 54.1's archived incident.io + Google SRE Workbook reads — re-cited, not re-fetched (same window).
- **Recency current-year → "Slack Socket Mode reliability production unattended reconnection 2026"** (HTTP-vs-Socket comparison read in full; hermes-agent #14326 + openclaw #31287 2026 silent-deafness issues).

## 4. Recency scan (last 2 years, 2024–2026) — PERFORMED

Findings in the 2024-2026 window, with explicit relevance to the away-week:

1. **Slack officially recommends HTTP over Socket Mode for production reliability**
   (source 7, current Slack docs): WebSocket "recycles containers serving
   connections every now and then, leading to occasional reliability issues." This
   is a real away-week consideration — **but migrating to HTTP mode requires a
   public HTTPS endpoint**, which pyfinagent (local-only Mac, per
   `project_local_only_deployment`) does NOT have and should NOT acquire for a
   1-week window. **Verdict: stay on Socket Mode; mitigate with the cron monitor +
   the external dead-man's-switch, NOT an architecture change.** This is a finding,
   not an action item.
2. **2026 "silent-deafness" failure mode** (hermes-agent #14326, openclaw #31287 —
   both 2026, snippet): a bot can report `running` / "DMs work, heartbeats fire,
   cron runs" **while Socket Mode is dead and group-channel events are silently
   dropped.** For pyfinagent this means: the **digests (outbound `chat.postMessage`)
   keep working even if Socket Mode is half-dead** — so digest delivery is a WEAKER
   liveness signal than slash-command responsiveness. The away-week monitor should
   therefore prove **outbound posting works** (the live confirmation digest) and
   not assume the bot is fully healthy just because the process is alive.
3. **No idempotency on `chat.postMessage`** (source 6, current): retries can
   duplicate. Relevant if the away-week adds any retry/backoff around the digest —
   the AlertDeduper covers alerts, but a naive digest retry loop could double-post.
   Keep digest sends single-shot (no auto-retry), or guard with a daily key.
4. **Apple 10-sec throttle rule unchanged** (source 3): still the governing
   constraint on ANY KeepAlive plist. Reinforces that a fast-crash-on-bad-config
   bot would be *suspended* by launchd — another reason the cron monitor (which has
   no such throttle) is the safer supervisor for the away week.

No source contradicts the recommended plan. The strongest *tension* is Slack's
HTTP-for-production guidance (source 7) — explicitly evaluated and rejected as
out-of-scope for a 1-week local window.

## 5. Key external findings

1. **Two bot instances do NOT duplicate slash-command events, but DO double-fire
   every scheduled job** (sources 1, 2). Slack routes each inbound payload to one
   random connection (no broadcast), so the *command* side degrades gracefully —
   BUT each process runs its own `AsyncIOScheduler`, so morning/evening digests +
   all 11 phase-9/core crons fire **twice**. This is the precise, load-bearing
   reason a `RunAtLoad` KeepAlive plist alongside the existing cron monitor is
   harmful, not helpful.

2. **The documented safe nohup→launchd migration is "kill the manual process FIRST,
   verify no duplicate processes exist, THEN load"** (source 4). If a launchd plist
   is ever adopted (post-away-week), the transition MUST stop the cron monitor +
   `pkill -f backend.slack_bot.app`, confirm zero instances, then
   `launchctl bootstrap`. Rollback = `launchctl bootout gui/$(id -u) <plist>`
   (source 5). Doing this DURING the away week is high-risk (a botched plist could
   leave the operator with zero bots and no console access).

3. **A launchd KeepAlive job that crashes within 10s gets suspended permanently**
   (source 3, verbatim). The bot calls `get_settings()` at import; a bad-config
   crash would be <10s → launchd stops retrying. The cron monitor has **no such
   throttle** — it retries every 5 min regardless — so for resilience against a
   transient bad state, the cron is actually MORE forgiving than launchd here.

4. **`chat.postMessage` is rate-safe for the digest cadence (1 msg/sec/channel) but
   has no dedup** (source 6). 2 digests/day + state-gated alerts is far under the
   limit. The `text=` fallback is required with `blocks` and is already present
   (scheduler.py:354, :399). Any away-week retry logic must be single-shot or
   daily-keyed to avoid duplicate posts.

5. **Slack says Socket Mode is less reliable than HTTP for production, and a bot can
   be silently "deaf" while appearing healthy** (sources 7 + 2026 issues). The
   away-week liveness check must verify **outbound delivery** (the confirmation
   digest), because a dead inbound socket won't stop digests but WILL stop
   slash-commands — and the operator's only window is the digest, which is the
   resilient half. Net: digests are the right thing to lean on; prove they post.

6. **A remote-operator daily status should be signal-dense: state + exceptions +
   links** (54.1 sources incident.io 2026 / Google SRE Workbook, re-cited; +
   AlertLogic/oxmaint snippets). The cron-health one-liner ("Crons: 17/19 green;
   FAILED: x,y") turns "all green" into the visible all-clear and surfaces a newly
   failed job in the one channel the operator watches — without paging noise.

---

## 6. Internal code inventory — file:line anchored (12 files)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/app.py` | 1-77 | Entry; **async Bolt + Socket Mode** (`AsyncSocketModeHandler`, :71); `start_scheduler(app)` at :56; `asyncio.run(main())` at :77; needs both bot+app tokens (:49) | live |
| `backend/slack_bot/scheduler.py` | 187-893 | `start_scheduler` (:187); `_send_morning_digest` (:330); `_send_evening_digest` (:363); `_is_us_trading_day_now` guard (:317); digest POST `app.client.chat_postMessage(channel=settings.slack_channel_id, blocks, text=...)` (:351, :396); base URL `_LOCAL_BACKEND_URL=http://127.0.0.1:8000` (:95) | live |
| `backend/slack_bot/formatters.py` | 1-1023 | `format_morning_digest` (:323), `format_evening_digest` (:391); imports ONLY `math`+`datetime` (:6-7) → **$0 template** | live |
| `backend/config/settings.py` | 1-75+ | `slack_channel_id` / `slack_bot_token` / `slack_app_token` / `morning_digest_hour` / `evening_digest_hour` / `watchdog_interval_minutes`; `paper_markets` NoDecode+validator (:62-75, the 54.1 fix) | live |
| `scripts/slack_bot_monitor.sh` | 1-40 | **THE SUPERVISOR**: 5-min cron; grep-guarded restart (:13-15, :27); `nohup … &` (:22) sourcing venv (:20-21); iMessage alert on restart (:32) | live (cron) |
| `scripts/slack_mention_checker.sh` | — | Separate 2-min cron; does NOT start the bot | live (cron) |
| `backend/api/cron_dashboard_api.py` | 410-527 | `GET /api/jobs/all` (:410) → `{jobs:[{id,source,schedule,next_run,last_run,status,description,controllable}], generated_at, n_total}`; merges main-APScheduler + slack_bot manifest + launchd; `/jobs/{id}/trigger` (:511) **only `paper_trading_daily`** (:517), else 400/404 | live |
| `backend/main.py` | 262-309 | `init_scheduler` + `_register_cron_scheduler("main", scheduler)` (:262-263); ticket-queue interval (:309) | live |
| `backend/slack_bot/commands.py` | — | slash handlers; **no digest-trigger command** (grep: none) | live |
| `backend/slack_bot/jobs/*` | — | 7 phase-9 job modules registered via `register_phase9_jobs` (scheduler.py:784) | live |
| `backend/tests/test_phase_slack_digest_71.py` | — | existing digest test home (where 54.2 tests should live) | live |
| `backend/tests/test_phase_51_3_digest_guard.py` | — | existing trading-day-guard test home | live |

### 6a. Live state snapshot (read-only `ps` / `launchctl` / `pmset` / `curl`)

- **slack_bot:** PID 42151, PPID 1, `python -m backend.slack_bot.app`, started
  2026-05-28 23:20. **Exactly 1 instance** (`ps | grep …slack_bot.app | wc -l` = 1).
  NO launchd plist (`launchctl list | grep slack` → none).
- **Supervisor:** user crontab `*/5 * * * * scripts/slack_bot_monitor.sh` (added
  2026-04-01, 2 months stable). `imsg` CLI present at `/opt/homebrew/bin/imsg` →
  the restart-alert + P0-escalation iMessage paths actually work.
- **Mac sleep:** `pmset -g` → `sleep 1 (sleep prevented by caffeinate)`. Backend
  runs under `caffeinate -i -s` (PID 36340). **Mac will not system-sleep** the
  away week. (displaysleep 10 min is screen-only, harmless.)
- **`/api/jobs/all` live:** 19 jobs — 11 ok, 2 scheduled, 2 never_run
  (evening_digest, weekly_fred_refresh — registry artifact, fire on schedule), 2
  running (backend, frontend), **2 failed (autoresearch, ablation — the 54.1
  settings fix; they show the *last* fire's failure, next-fire scheduled 02:00 /
  03:00; re-verify on next nightly fire)**.
- **Trading-day guard:** `is_trading_day(2026-06-01, "US")` = **True** (Monday) → a
  confirmation digest sent today is NOT suppressed by the `_is_us_trading_day_now`
  guard.
- **Settings resolved:** `slack_channel_id` set, **11 chars, starts `C`** (a real
  Slack channel id); `bot_token` `xoxb-…`; `app_token` `xapp-…`; morning 08:00 ET,
  evening 17:00 ET, watchdog 15 min. (Resolved via `get_settings()` — `.env` itself
  not read; values masked.)

---

## 7. THE LOAD-BEARING UNKNOWN — RESOLVED

**Q: How is the bot launched + kept alive today?**

**A: A user-crontab process-monitor, not launchd, not nohup-bare, not tmux/screen.**

1. **Launch + restart:** `scripts/slack_bot_monitor.sh`, scheduled in the **user
   crontab** as `*/5 * * * *` (confirmed via `crontab -l`). Every 5 minutes it runs
   `ps aux | grep -E "python.*backend.slack_bot.app" | grep -v grep`; if no match it
   `cd`s to the repo, `source .venv/bin/activate`, and `nohup python -m
   backend.slack_bot.app >> backend_slack.log 2>&1 &`. The `nohup &` detaches the
   child → it reparents to PID 1 → **explains the observed PPID 1 / no-plist /
   orphaned state.** It was started ~2026-05-28 by exactly this path.
2. **Is there a restart mechanism today?** YES — the 5-min cron monitor (since
   2026-04-01). Max recovery gap after a crash/reboot ≈ 5 min. It ALSO sends the
   operator an iMessage on every restart (independent of Slack + the Mac console).
3. **CRITICAL supervisor-decision answer — would a launchd KeepAlive `RunAtLoad`
   plist create a SECOND instance?** **YES, and it would be actively harmful:**
   - At `launchctl bootstrap` (RunAtLoad), launchd starts bot instance #2 while the
     cron-managed instance #1 is alive → **two processes**.
   - Slack (sources 1, 2): two Socket Mode connections on one app token → inbound
     payloads routed to ONE at random → slash-commands NOT duplicated (degrade
     gracefully) — **but each process runs its own `AsyncIOScheduler`
     (scheduler.py:196)** → **morning/evening digests fire TWICE, and so do all 11
     phase-9/core crons** including `nightly_mda_retrain` (03:00 UTC) and
     `hourly_signal_warmup`. (These phase-9 jobs are template/data ML jobs, not
     Claude-routine LLM spend — but `nightly_mda_retrain` retrains a model and a 2x
     fire is still wasteful + could race BQ writes; treat double-fire as a real
     defect regardless of $ cost.)
   - **The two supervisors would also mask each other:** the cron monitor's grep
     would always find "a" bot, so it could never detect if the launchd one died,
     and vice-versa. Worst-of-both.
   - **SAFE transition (if a plist is EVER adopted — NOT recommended for the away
     week):** per source 4 — (a) comment out the `*/5` cron line, (b)
     `pkill -f backend.slack_bot.app`, (c) confirm `ps … | wc -l` = 0, (d)
     `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist`
     with `KeepAlive{SuccessfulExit:false}` + `RunAtLoad:true` + `ThrottleInterval:5`
     + StandardOut/ErrorPath, (e) ensure the bot does not crash within 10s of launch
     (Apple's rule — guard `get_settings()` so a bad value logs+exits-slow, not
     instant-crash). **Rollback:** `launchctl bootout gui/$(id -u) <plist>` + delete
     the plist + re-enable the cron line. This is a multi-step, console-required
     operation — **exactly what you do NOT want to attempt the day the operator
     leaves.**

**Conclusion:** the lifeline already has a 2-month-stable supervisor. The right
54.2 move is to **harden what exists**, not bolt on a competing supervisor.

---

## 8. Supervisor options: safe vs risky (with rollback)

| Option | Risk to running bot | Double-fire? | Verdict |
|--------|---------------------|--------------|---------|
| **A. Keep the cron monitor; harden it** (verify it's the right grep; ensure `backend_slack.log` rotates; confirm `imsg` path) | **None** (read/verify only) | No | **RECOMMENDED** — zero-risk, the supervisor is already proven |
| **B. Add launchd KeepAlive plist alongside the cron** | **HIGH** — 2nd instance, double-fired digests + 11 crons, supervisors mask each other | **YES** | **REJECT for away-week** |
| **C. Migrate cron→launchd (replace, not add)** | Medium — requires kill-first + verify-zero + bootstrap; console-required; 10s-crash-suspend trap | No (if done right) | **DEFER to post-away-week** (source 4 safe steps + bootout rollback) |
| **D. External dead-man's-switch on the digest** (success-ping Healthchecks.io/Cronitor; alert via email/SMS) | **None** (additive `curl` on a working path) | No | **OPTIONAL add-on** — catches whole-Mac-down, which no internal check can; operator-gated only for the account/pip |
| **E. Send ONE live confirmation digest now** (standalone one-shot script, §9) | **None** — Web-API POST, no Socket Mode connection, never touches PID 42151 | No | **RECOMMENDED** — proves outbound delivery (the resilient half per source 2/7) |

**Rollback for the recommended path (A+E+cron-health line):** all changes are
either read-only verification (A), a new standalone script that doesn't run in the
bot (E), or a purely-additive formatter line behind the existing $0 digest (§10) —
**none of them can take down PID 42151.** If the cron-health line errors, it must
fail-open (wrap in try/except, omit the line) so a `/api/jobs/all` hiccup never
suppresses the digest the operator depends on.

---

## 9. Exact code path to send ONE confirmation digest to the operator channel

**There is NO in-process "trigger digest now" endpoint.** `/api/jobs/{id}/trigger`
(cron_dashboard_api.py:511) is scoped to `paper_trading_daily` ONLY (:517);
everything else → 404/400. So the safe path is a **standalone one-shot script**
(`scripts/...` or a `python -c`) that reuses the existing formatter + a bare
Web-API client — **without importing/instantiating `AsyncSocketModeHandler`**, so it
opens **zero** Socket Mode connections and is invisible to PID 42151:

```python
# one-shot; opens NO socket-mode connection -> cannot create a 2nd bot instance
import asyncio, httpx
from slack_sdk.web.async_client import AsyncWebClient
from backend.config.settings import get_settings
from backend.slack_bot.formatters import format_morning_digest

async def main():
    s = get_settings()
    async with httpx.AsyncClient(timeout=30.0) as c:
        pf = (await c.get("http://127.0.0.1:8000/api/paper-trading/portfolio")).json()
        rp = (await c.get("http://127.0.0.1:8000/api/reports/?limit=5")).json()
    blocks = format_morning_digest(pf, rp)              # $0, template-only
    client = AsyncWebClient(token=s.slack_bot_token.get_secret_value())
    await client.chat_postMessage(                       # Web API, not Socket Mode
        channel=s.slack_channel_id, blocks=blocks,
        text="PyFinAgent CONFIRMATION digest (away-week readiness check)",
    )

asyncio.run(main())
```

- Mirrors `_send_morning_digest` (scheduler.py:330-355) exactly, minus the
  trading-day guard (today IS a trading day anyway, so either way it posts) and
  minus the bot/Socket-Mode handler.
- `AsyncWebClient` import verified present. `slack_channel_id`/`bot_token` verified
  set + real. One POST = 1 msg, well under 1/sec/channel (source 6).
- **Why this is safe:** the bot's *scheduler* is what fires digests; this script is
  a separate process making a single Web-API call. It does NOT call
  `start_scheduler`, does NOT open a WebSocket, and does NOT register with launchd
  or cron. PID 42151 is untouched.
- The operator will receive a labeled "CONFIRMATION" digest → proves the
  token+channel+formatter+Block-Kit path end-to-end (the resilient outbound half).

---

## 10. Where/how to fold in the cron-health line

**Data source:** `GET /api/jobs/all` → `{jobs:[{id,status,...}], n_total}` (live:
19 jobs). Derive a one-liner:

```python
# inside _send_morning_digest, after the portfolio fetch (scheduler.py ~:347)
jobs = (await client.get(f"{_LOCAL_BACKEND_URL}/api/jobs/all")).json().get("jobs", [])
bad = [j["id"] for j in jobs if j.get("status") == "failed"]
n_green = sum(1 for j in jobs if j.get("status") in ("ok", "scheduled", "running"))
cron_line = (f":white_check_mark: Crons: {n_green}/{len(jobs)} healthy"
             if not bad else
             f":warning: Crons: {n_green}/{len(jobs)} healthy — FAILED: {', '.join(bad)}")
```

**Where it renders — exact Block Kit insertion point:** `format_morning_digest`
(formatters.py:323) builds `blocks` = header (:325) → optional Portfolio section
(:362) → optional Recent Analyses (:377) → `divider` (:382) → `context` footer
(:383). **Insert a new `section` block immediately before the `divider` at line
382** (so the cron-health line sits under the portfolio/analyses, above the footer).
Two clean ways:
- (a) **Pass `cron_health: str | None = None` as a new kwarg** to
  `format_morning_digest` and append `{"type":"section","text":{"type":"mrkdwn",
  "text": cron_health}}` when non-None. The scheduler computes the line (it already
  has the httpx client open) and passes it in. **Keeps formatters.py $0 + I/O-free
  (it stays a pure template builder)** — the cleanest separation and matches the
  existing pattern (formatters never do I/O).
- (b) Compute inside the formatter — REJECT: would make formatters.py do an HTTP
  call, breaking its pure-template contract and the $0 guarantee shape.

**Recommend (a).** Mirror the same kwarg into `format_evening_digest` (:391, insert
before its `divider` at :443) if the operator wants it on both; morning-only is
sufficient (one all-clear/day). **Must fail-open:** the scheduler wraps the
`/api/jobs/all` fetch in try/except and passes `cron_health=None` on any error, so a
jobs-endpoint hiccup never blocks the digest.

**DO-NO-HARM byte-identity:** with the kwarg defaulting to `None`, every existing
caller and test produces byte-identical blocks → the change is purely additive.

---

## 11. Is the digest $0 / template? (operator-gating decision) — DECISIVE: YES, $0

- `backend/slack_bot/formatters.py` imports ONLY `math` + `datetime` (:6-7). A grep
  for `anthropic|openai|google|llm|llm_client|generate(|.complete(|messages.create`
  → **NONE**. `format_morning_digest` / `format_evening_digest` are pure Block Kit
  builders over two backend HTTP GETs (`/api/paper-trading/portfolio`,
  `/api/reports`, `/api/paper-trading/trades`). **Zero token spend.**
- The proposed cron-health line adds one more **internal** GET (`/api/jobs/all`) —
  also $0 (no LLM).
- **Implication:** the daily digest + the live confirmation send + the cron-health
  line are **NOT operator-gated** (LLM spend / pip / BQ-DROP are the only gated
  axes per `active_goal.md`). All three can proceed autonomously. (Adding an
  external dead-man's-switch *account/pip* would be the only operator-gated item,
  and is optional.)

---

## 12. Recommended phase-54.2 plan (does NOT risk the running lifeline)

Ordered; every item is zero-risk to PID 42151 and $0:

1. **VERIFY the supervisor is sound (Option A).** Confirm (read-only) the `*/5`
   cron monitor is present + its grep matches the live process name + it sources the
   venv + `imsg` is on PATH. Document it in `live_check_54.2.md` so the operator
   knows the bot IS supervised (correcting the 54.1 "unsupervised" framing). Note
   the Mac won't sleep (caffeinate). **No code change.**

2. **Send ONE live confirmation digest (Option E, §9).** Standalone one-shot script
   → posts a labeled CONFIRMATION digest to the real operator channel via
   `AsyncWebClient` (no Socket Mode connection). Capture the returned `ts` / a
   screenshot as the `live_check_54.2.md` evidence that delivery works end-to-end.
   **This is the single most important deliverable** — it proves the operator's only
   window actually receives a message.

3. **Fold the cron-health one-liner into the morning digest (§10, Option a).** New
   `cron_health` kwarg on `format_morning_digest`; scheduler computes it from
   `/api/jobs/all` (fail-open → `None`); insert a `section` before the `divider`
   (formatters.py:382). Byte-identical when the kwarg is `None`. Add a test in
   `test_phase_slack_digest_71.py` asserting (i) byte-identity with kwarg absent and
   (ii) the line renders with a synthetic failed job. **No restart of the bot needed
   to ship the code** — but the running bot won't pick up the new formatter until it
   restarts; the 5-min monitor will NOT restart a healthy bot, so either (i) accept
   that the cron-health line goes live on the next natural bot restart, or (ii)
   do a controlled single-restart: `pkill -f backend.slack_bot.app` and let the
   monitor respawn within 5 min (ONE instance guaranteed — the monitor greps first).
   Document the chosen path; option (ii) is the clean way to make it live for the
   away week.

4. **(OPTIONAL, operator-gated) External dead-man's-switch (Option D).** If the
   operator wants belt-and-suspenders against a whole-Mac-down, success-ping a free
   Healthchecks.io/Cronitor check from the morning digest (24h interval + 30-60 min
   grace) routed to email/SMS. The `curl` is $0; the account/signup is the only
   operator-gated bit. Internal checks cannot catch the host vanishing (54.1 sources
   1/6/9) — this is the only true external safety net. **Escalate the service
   choice; don't force.**

5. **DO NOT add a launchd KeepAlive plist for the bot this week (Option B/C).**
   It double-fires every scheduled job and the two supervisors mask each other
   (§7-8). If the operator wants launchd long-term, schedule it as a post-away-week
   step using the kill-first→verify-zero→bootstrap migration (source 4) with
   `launchctl bootout` rollback (source 5).

6. **Note (not a 54.2 blocker):** the 2 `failed` launchd jobs (autoresearch,
   ablation) are the 54.1 settings fix awaiting their next nightly fire to flip to
   `ok`. The cron-health line WILL show them as FAILED until ~02:00/03:00 the next
   night — either (a) accept the one-night "FAILED: autoresearch, ablation" line as
   accurate (they genuinely last-failed), or (b) trigger a manual re-fire before the
   operator leaves to clear them. Flag this so the line doesn't cry wolf.

### Application to pyfinagent (external finding → file:line)

| External finding | pyfinagent anchor / action |
|---|---|
| 2 instances → split commands but DOUBLE-fire schedulers (src 1,2) | reject launchd plist alongside cron (scheduler.py:196 per-process scheduler); keep single instance |
| Safe migration = kill-first/verify-zero/bootstrap; bootout rollback (src 4,5) | defer cron→launchd to post-away-week; documented steps in §7 |
| 10-sec crash-suspend rule (src 3) | any future bot plist must not crash <10s; `get_settings()` (settings.py) must fail slow; cron monitor has no such trap (safer now) |
| chat.postMessage 1/sec + text fallback + NO dedup (src 6) | confirmation send + digests are single-shot; `text=` already present (scheduler.py:354,:399); no auto-retry |
| Socket Mode less reliable than HTTP; bot can be silently deaf (src 7 + 2026 issues) | prove OUTBOUND delivery via the confirmation digest (§9); don't migrate to HTTP for a 1-week local window |
| Remote status = state+exceptions+links, signal not noise (54.1 src incident.io/SRE) | cron-health one-liner in morning digest (formatters.py:382 insert); state-gated, $0 |

---

## 13. Research Gate Checklist

Hard blockers — `gate_passed` false if any unchecked:
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (**7** read; Slack official + Apple + launchd.info lead the set)
- [x] 10+ unique URLs total incl. snippet-only (**23** collected)
- [x] Recency scan (last 2 years) performed + reported (§4 — incl. the decisive 2026 HTTP-vs-Socket + silent-deafness findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (app.py:49/56/71/77; scheduler.py:95/187/196/317/330/351/354/363/396/399/784; formatters.py:6-7/323/382/391/443; settings.py:62-75; cron_dashboard_api.py:410/511/517; main.py:262-263/309; slack_bot_monitor.sh:13-15/20-22/27/32; + live `ps`/`launchctl`/`pmset`/`crontab -l`/`curl /api/jobs/all` output)

Soft checks:
- [x] Internal exploration covered every relevant module (app entry, scheduler full, formatters digests, settings, the monitor + mention-checker crons, /api/jobs/all endpoint, main lifespan, commands, test homes; live process/launchd/pmset/crontab/jobs-endpoint state)
- [x] Contradictions / consensus noted (Slack's HTTP-for-production guidance is the one tension — explicitly evaluated + rejected as out-of-scope for a 1-week local window; the 54.1 "bot is unsupervised" framing is REFUTED)
- [x] All claims cited per-claim (URL for external, file:line / command output for internal)

---

```json
{"tier":"moderate","external_sources_read_in_full":7,"snippet_only_sources":16,"urls_collected":23,"recency_scan_performed":true,"internal_files_inspected":12,"report_md":"handoff/current/research_brief.md","gate_passed":true}
```
